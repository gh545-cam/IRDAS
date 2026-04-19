"""
Recurrent neural network for residual dynamics learning.
Learns model residuals in reduced 7-state dynamics space.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class StateNormalizer:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.mean = np.zeros(state_dim)
        self.std = np.ones(state_dim)
        self.fitted = False

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std[self.std < 1e-6] = 1.0
        self.fitted = True

    def normalize(self, data):
        if not self.fitted:
            return data
        return (data - self.mean) / self.std

    def denormalize(self, data):
        if not self.fitted:
            return data
        return data * self.std + self.mean

    def save(self):
        return {'mean': self.mean.copy(), 'std': self.std.copy()}

    def load(self, state):
        self.mean = state['mean'].copy()
        self.std = state['std'].copy()
        self.fitted = True


class RecurrentResidualDynamicsNetwork(nn.Module):
    """
    GRU-based residual dynamics network with sequence memory.
    """

    def __init__(self, state_dim=7, control_dim=3, hidden_dim=128, num_layers=2, dropout=0.2, output_dim=7):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim

        input_dim = state_dim + control_dim
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state_seq, control_seq, hidden=None):
        # state_seq/control_seq: [B,T,*]
        x = torch.cat([state_seq, control_seq], dim=-1)
        x = self.input_layer(x)
        rnn_out, hidden_out = self.rnn(x, hidden)
        residual = self.head(rnn_out)
        return residual, hidden_out


# Backward-compatible name
ResidualDynamicsNetwork = RecurrentResidualDynamicsNetwork


class ResidualDynamicsLearner:
    """
    Recurrent residual dynamics learner with:
      - sequence training
      - stateful inference
      - PINN-inspired regularizers (traction-circle and steering symmetry)

    Args:
        state_dim: reduced dynamics-state dimension (default 7)
        control_dim: control dimension (default 3)
        learning_rate: optimizer learning rate
        l2_reg: Adam weight decay
        device: torch device string
        sequence_length: sequence window length used for recurrent training
        pinn_traction_weight: weight of traction-circle regularizer
        pinn_symmetry_weight: weight of steering-symmetry regularizer
        max_residual_accel: soft cap (m/s^2) for residual acceleration magnitude
        model_dt: discretization step used to convert residual velocity increments
                  into residual accelerations in traction regularization
    """

    def __init__(self, state_dim=7, control_dim=3, learning_rate=1e-3,
                 l2_reg=1e-4, device='cpu', sequence_length=20,
                 pinn_traction_weight=0.05, pinn_symmetry_weight=0.05,
                 max_residual_accel=18.0, model_dt=0.05):
        self.device = device
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = state_dim
        self.sequence_length = max(2, int(sequence_length))

        self.network = RecurrentResidualDynamicsNetwork(
            state_dim=state_dim,
            control_dim=control_dim,
            output_dim=self.output_dim
        ).to(device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=l2_reg)
        self.loss_fn = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )

        self.state_normalizer = StateNormalizer(state_dim)
        self.control_normalizer = StateNormalizer(control_dim)
        self.residual_normalizer = StateNormalizer(state_dim)

        self.training_history = {'train_loss': [], 'val_loss': [], 'epochs': []}

        self.pinn_traction_weight = float(pinn_traction_weight)
        self.pinn_symmetry_weight = float(pinn_symmetry_weight)
        self.max_residual_accel = float(max_residual_accel)
        self.model_dt = float(model_dt)

        self._stateful_hidden = None

    def reset_stateful_inference(self):
        self._stateful_hidden = None

    def _build_sequence_dataset(self, states, controls, residuals, seq_len):
        n = len(states)
        if n < seq_len:
            raise ValueError(
                f"Insufficient samples for sequence dataset: got {n}, need at least {seq_len}."
            )
        Xs, Xu, Yr = [], [], []
        for i in range(n - seq_len + 1):
            Xs.append(states[i:i + seq_len])
            Xu.append(controls[i:i + seq_len])
            Yr.append(residuals[i:i + seq_len])
        return np.array(Xs), np.array(Xu), np.array(Yr)

    def _mirror_dynamics_state(self, states):
        # states shape [B,T,7] = [vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr]
        mirrored = states.clone()
        mirrored[..., 1] = -states[..., 1]  # vy odd in mirror
        mirrored[..., 2] = -states[..., 2]  # yaw-rate odd
        mirrored[..., 3] = states[..., 4]   # swap FL/FR
        mirrored[..., 4] = states[..., 3]
        mirrored[..., 5] = states[..., 6]   # swap RL/RR
        mirrored[..., 6] = states[..., 5]
        return mirrored

    def _mirror_control(self, controls):
        mirrored = controls.clone()
        mirrored[..., 0] = -controls[..., 0]  # steering sign flips
        return mirrored

    def _mirror_residual(self, residual):
        mirrored = residual.clone()
        mirrored[..., 1] = -residual[..., 1]
        mirrored[..., 2] = -residual[..., 2]
        mirrored[..., 3] = residual[..., 4]
        mirrored[..., 4] = residual[..., 3]
        mirrored[..., 5] = residual[..., 6]
        mirrored[..., 6] = residual[..., 5]
        return mirrored

    def _traction_circle_regularizer(self, pred_residual):
        # Approximate residual accelerations from predicted residual velocity increments.
        # Uses configured model_dt to stay consistent with runtime discretization.
        # residual_v / dt approximates residual acceleration magnitude.
        dt = self.model_dt
        ax_res = pred_residual[..., 0] / dt
        ay_res = pred_residual[..., 1] / dt
        mag = torch.sqrt(ax_res * ax_res + ay_res * ay_res + 1e-8)
        excess = torch.relu(mag - self.max_residual_accel)
        return torch.mean(excess * excess)

    def _symmetry_regularizer(self, states, controls):
        mirrored_states = self._mirror_dynamics_state(states)
        mirrored_controls = self._mirror_control(controls)

        pred, _ = self.network(states, controls, hidden=None)
        pred_mirror, _ = self.network(mirrored_states, mirrored_controls, hidden=None)
        target_mirror = self._mirror_residual(pred)
        return torch.mean((pred_mirror - target_mirror) ** 2)

    def train_epoch(self, train_loader):
        self.network.train()
        total_loss = 0.0
        num_batches = 0

        for states, controls, residuals in train_loader:
            states = states.to(self.device)
            controls = controls.to(self.device)
            residuals = residuals.to(self.device)

            self.optimizer.zero_grad()
            predictions, _ = self.network(states, controls)

            data_loss = self.loss_fn(predictions, residuals)
            traction_loss = self._traction_circle_regularizer(predictions)
            symmetry_loss = self._symmetry_regularizer(states, controls)
            loss = data_loss + self.pinn_traction_weight * traction_loss + self.pinn_symmetry_weight * symmetry_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            raise ValueError(
                f"Empty training loader: no sequence batches were generated. "
                f"Check that training data has at least {self.sequence_length} samples."
            )
        return total_loss / num_batches

    def validate(self, val_loader):
        self.network.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for states, controls, residuals in val_loader:
                states = states.to(self.device)
                controls = controls.to(self.device)
                residuals = residuals.to(self.device)

                predictions, _ = self.network(states, controls)
                data_loss = self.loss_fn(predictions, residuals)
                traction_loss = self._traction_circle_regularizer(predictions)
                symmetry_loss = self._symmetry_regularizer(states, controls)
                loss = data_loss + self.pinn_traction_weight * traction_loss + self.pinn_symmetry_weight * symmetry_loss

                total_loss += loss.item()
                num_batches += 1

        if num_batches == 0:
            raise ValueError(
                f"Empty validation loader: no sequence batches were generated. "
                f"Check that validation data has at least {self.sequence_length} samples."
            )
        return total_loss / num_batches

    def fit(self, train_states, train_controls, train_residuals,
            val_states=None, val_controls=None, val_residuals=None,
            epochs=100, batch_size=32, verbose=True, sequence_length=None):
        seq_len = self.sequence_length if sequence_length is None else max(2, int(sequence_length))

        self.state_normalizer.fit(train_states)
        self.control_normalizer.fit(train_controls)
        self.residual_normalizer.fit(train_residuals)

        train_states_norm = self.state_normalizer.normalize(train_states)
        train_controls_norm = self.control_normalizer.normalize(train_controls)
        train_residuals_norm = self.residual_normalizer.normalize(train_residuals)

        tr_s, tr_u, tr_r = self._build_sequence_dataset(train_states_norm, train_controls_norm, train_residuals_norm, seq_len)

        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(tr_s),
                torch.FloatTensor(tr_u),
                torch.FloatTensor(tr_r),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = None
        if val_states is not None and len(val_states) >= seq_len:
            val_states_norm = self.state_normalizer.normalize(val_states)
            val_controls_norm = self.control_normalizer.normalize(val_controls)
            val_residuals_norm = self.residual_normalizer.normalize(val_residuals)
            va_s, va_u, va_r = self._build_sequence_dataset(val_states_norm, val_controls_norm, val_residuals_norm, seq_len)
            val_loader = DataLoader(
                TensorDataset(
                    torch.FloatTensor(va_s),
                    torch.FloatTensor(va_u),
                    torch.FloatTensor(va_r),
                ),
                batch_size=batch_size,
                shuffle=False,
            )

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)

            self.training_history['epochs'].append(epoch)
            self.training_history['train_loss'].append(train_loss)
            if val_loss is not None:
                self.training_history['val_loss'].append(val_loss)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                msg = f"Epoch {epoch}/{epochs}: train_loss={train_loss:.6f}"
                if val_loss is not None:
                    msg += f", val_loss={val_loss:.6f}"
                print(msg)

            if val_loss is not None:
                self.scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break

    def predict(self, states, controls, stateful=False):
        self.network.eval()

        single_sample = False
        if states.ndim == 1:
            single_sample = True
            states = states[np.newaxis, np.newaxis, :]   # [1,1,state_dim]
            controls = controls[np.newaxis, np.newaxis, :]  # [1,1,control_dim]
        elif states.ndim == 2:
            states = states[:, np.newaxis, :]  # [N,1,state_dim]
            controls = controls[:, np.newaxis, :]  # [N,1,control_dim]

        states_norm = self.state_normalizer.normalize(states)
        controls_norm = self.control_normalizer.normalize(controls)

        states_tensor = torch.FloatTensor(states_norm).to(self.device)
        controls_tensor = torch.FloatTensor(controls_norm).to(self.device)

        with torch.no_grad():
            hidden_in = self._stateful_hidden if stateful else None
            residuals_norm, hidden_out = self.network(states_tensor, controls_tensor, hidden=hidden_in)
            if stateful:
                self._stateful_hidden = hidden_out.detach()

        residuals_np = residuals_norm.cpu().numpy()
        residuals = self.residual_normalizer.denormalize(residuals_np)

        # pick the last timestep output
        residuals = residuals[:, -1, :]

        if single_sample:
            return residuals[0]
        return residuals

    def save(self, path):
        checkpoint = {
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'history': self.training_history,
            'state_normalizer': self.state_normalizer.save(),
            'control_normalizer': self.control_normalizer.save(),
            'residual_normalizer': self.residual_normalizer.save(),
            'sequence_length': self.sequence_length,
            'pinn_traction_weight': self.pinn_traction_weight,
            'pinn_symmetry_weight': self.pinn_symmetry_weight,
            'max_residual_accel': self.max_residual_accel,
            'model_dt': self.model_dt,
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_history = checkpoint.get('history', self.training_history)
        self.state_normalizer.load(checkpoint['state_normalizer'])
        self.control_normalizer.load(checkpoint['control_normalizer'])
        self.residual_normalizer.load(checkpoint['residual_normalizer'])
        self.sequence_length = checkpoint.get('sequence_length', self.sequence_length)
        self.pinn_traction_weight = checkpoint.get('pinn_traction_weight', self.pinn_traction_weight)
        self.pinn_symmetry_weight = checkpoint.get('pinn_symmetry_weight', self.pinn_symmetry_weight)
        self.max_residual_accel = checkpoint.get('max_residual_accel', self.max_residual_accel)
        self.model_dt = checkpoint.get('model_dt', self.model_dt)
        self.reset_stateful_inference()


def extract_dynamics_states(full_state):
    """
    Full state: [x, y, psi, vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr, rpm, gear, throttle]
    Dynamics states: [vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr]
    """
    if isinstance(full_state, np.ndarray):
        return full_state[[3, 4, 5, 6, 7, 8, 9]]
    return torch.stack([full_state[3], full_state[4], full_state[5],
                        full_state[6], full_state[7], full_state[8], full_state[9]])


def generate_training_data(baseline_model, real_model, n_samples=1000, params=None):
    states = []
    controls = []
    residuals = []

    state = np.array([0., 0., 0., 5., 0., 0., 5., 5., 5., 5., 3000., 1., 0.1], dtype=np.float64)
    dt = 0.05

    for i in range(n_samples):
        u = np.array([
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(0.0, 1.0),
            np.random.uniform(0.0, 0.3)
        ])

        try:
            if params is not None:
                baseline_next = baseline_model(state.copy(), u, dt, params)
                real_next = real_model(state.copy(), u, dt, params)
            else:
                baseline_next = baseline_model(state.copy(), u, dt)
                real_next = real_model(state.copy(), u, dt)

            state_dyn = extract_dynamics_states(state)
            baseline_dyn = extract_dynamics_states(baseline_next)
            real_dyn = extract_dynamics_states(real_next)

            residual = real_dyn - baseline_dyn

            states.append(state_dyn.copy())
            controls.append(u.copy())
            residuals.append(residual.copy())

            state = real_next.copy()
        except Exception as e:
            print(f"Warning at sample {i}: {e}")
            continue

    return np.array(states), np.array(controls), np.array(residuals)


if __name__ == "__main__":
    print("Recurrent Residual Dynamics Network Test")
    print("=" * 70)

    learner = ResidualDynamicsLearner(state_dim=7, control_dim=3, l2_reg=1e-5, sequence_length=8)

    np.random.seed(42)
    n_train = 200
    train_states = np.random.randn(n_train, 7) * 0.5
    train_states[:, 0] = 5.0 + np.random.randn(n_train) * 0.5
    train_controls = np.random.randn(n_train, 3) * 0.1
    train_residuals = np.random.randn(n_train, 7) * 0.01

    learner.fit(train_states, train_controls, train_residuals, epochs=5, batch_size=16, verbose=True)

    learner.reset_stateful_inference()
    test_state = np.random.randn(7) * 0.5
    test_state[0] = 5.0
    test_control = np.random.randn(3) * 0.1

    pred1 = learner.predict(test_state, test_control, stateful=True)
    pred2 = learner.predict(test_state, test_control, stateful=True)
    learner.reset_stateful_inference()
    pred3 = learner.predict(test_state, test_control, stateful=True)

    print(f"Pred shape: {pred1.shape}")
    print(f"Stateful delta norm: {np.linalg.norm(pred2 - pred1):.6f}")
    print(f"Reset delta norm: {np.linalg.norm(pred3 - pred1):.6f}")
