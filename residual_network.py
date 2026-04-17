"""
Neural Network for residual dynamics learning.
Learns to correct model predictions by training on the difference between
baseline model and real vehicle data.

Uses reduced state space (7 states) focusing on dynamics-relevant states:
[vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr]

Excludes: x, y, psi (analytically integrated), gear (discrete, check_upshift), 
rpm (computed from wheel speeds), throttle (part of control)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class StateNormalizer:
    """
    Normalizes input and output states to zero mean and unit variance.
    Improves neural network training by handling different state scales.
    """
    
    def __init__(self, state_dim):
        """
        Initialize normalizer.
        
        Args:
            state_dim: dimension of state vector
        """
        self.state_dim = state_dim
        self.mean = np.zeros(state_dim)
        self.std = np.ones(state_dim)
        self.fitted = False
    
    def fit(self, data):
        """
        Fit normalizer on data.
        
        Args:
            data: array of shape (N, state_dim)
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Prevent division by zero
        self.std[self.std < 1e-6] = 1.0
        self.fitted = True
    
    def normalize(self, data):
        """Normalize data to zero mean, unit variance."""
        if not self.fitted:
            return data
        return (data - self.mean) / self.std
    
    def denormalize(self, data):
        """Denormalize data back to original scale."""
        if not self.fitted:
            return data
        return data * self.std + self.mean
    
    def save(self):
        """Save normalizer state."""
        return {'mean': self.mean.copy(), 'std': self.std.copy()}
    
    def load(self, state):
        """Load normalizer state."""
        self.mean = state['mean'].copy()
        self.std = state['std'].copy()
        self.fitted = True


class ResidualDynamicsNetwork(nn.Module):
    """
    Neural network that learns residual dynamics: r = f_real(x, u) - f_model(x, u)
    
    Uses reduced state space (7 states) for dynamics-relevant learning:
    [vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr]
    
    This network takes reduced state and control as input and predicts the correction
    needed to adjust model predictions to match real vehicle behavior.
    """
    
    def __init__(self, state_dim=7, control_dim=3, hidden_dims=[128, 128, 64], 
                 dropout_rate=0.2, output_dim=7):
        """
        Initialize residual dynamics network.
        
        Args:
            state_dim: dimension of reduced state vector (7 for dynamics states)
            control_dim: dimension of control vector (3: steering, throttle, brake)
            hidden_dims: list of hidden layer dimensions
            dropout_rate: dropout probability for regularization
            output_dim: dimension of output (residuals for each state)
        """
        super(ResidualDynamicsNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim
        
        # Build network architecture
        input_dim = state_dim + control_dim
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers with batch normalization and dropout
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (no activation for residual prediction)
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state, control):
        """
        Forward pass: predict residual dynamics.
        
        Args:
            state: reduced state vector(s) [batch_size, 7]
            control: control vector(s) [batch_size, 3]
            
        Returns:
            residual dynamics [batch_size, 7]
        """
        # Concatenate state and control
        x = torch.cat([state, control], dim=-1)
        
        # Forward through network
        residual = self.network(x)
        
        return residual
    
    def predict_corrected_dynamics(self, state, control, baseline_dynamics):
        """
        Predict corrected state dynamics.
        
        Args:
            state: reduced state vector [batch_size, 7]
            control: control vector [batch_size, 3]
            baseline_dynamics: baseline model predictions [batch_size, 7]
            
        Returns:
            corrected dynamics = baseline + residual
        """
        with torch.no_grad():
            residual = self.forward(state, control)
            corrected = baseline_dynamics + residual
        
        return corrected


class ResidualDynamicsLearner:
    """
    Trainer for residual dynamics learning with online adaptation capability.
    Includes L2 regularization and input/output normalization.
    """
    
    def __init__(self, state_dim=7, control_dim=3, learning_rate=1e-3, 
                 l2_reg=1e-5, device='cpu'):
        """
        Initialize learner.
        
        Args:
            state_dim: dimension of reduced state vector (7)
            control_dim: dimension of control vector (3)
            learning_rate: optimizer learning rate
            l2_reg: L2 regularization coefficient (weight decay)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = state_dim
        
        # Build network
        self.network = ResidualDynamicsNetwork(state_dim, control_dim, 
                                               output_dim=self.output_dim).to(device)
        
        # Optimizer with L2 regularization (weight_decay)
        self.optimizer = optim.Adam(self.network.parameters(), 
                                   lr=learning_rate, 
                                   weight_decay=l2_reg)
        self.loss_fn = nn.MSELoss()
        
        # State normalizers for input and output
        self.state_normalizer = StateNormalizer(state_dim)
        self.control_normalizer = StateNormalizer(control_dim)
        self.residual_normalizer = StateNormalizer(state_dim)
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader with (state, control, residual) tuples
            
        Returns:
            average training loss
        """
        self.network.train()
        total_loss = 0.0
        num_batches = 0
        
        for states, controls, residuals in train_loader:
            # Move to device
            states = states.to(self.device)
            controls = controls.to(self.device)
            residuals = residuals.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.network(states, controls)
            
            # Compute loss
            loss = self.loss_fn(predictions, residuals)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader):
        """
        Validate on validation set.
        
        Args:
            val_loader: DataLoader with (state, control, residual) tuples
            
        Returns:
            average validation loss
        """
        self.network.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for states, controls, residuals in val_loader:
                states = states.to(self.device)
                controls = controls.to(self.device)
                residuals = residuals.to(self.device)
                
                predictions = self.network(states, controls)
                loss = self.loss_fn(predictions, residuals)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def fit(self, train_states, train_controls, train_residuals,
            val_states=None, val_controls=None, val_residuals=None,
            epochs=100, batch_size=32, verbose=True):
        """
        Train the network with input/output normalization.
        
        Args:
            train_states: training state data [N, 7]
            train_controls: training control data [N, 3]
            train_residuals: training residual data [N, 7]
            val_states: validation state data
            val_controls: validation control data
            val_residuals: validation residual data
            epochs: number of training epochs
            batch_size: batch size
            verbose: whether to print progress
        """
        # Fit normalizers on training data
        print("Fitting state and control normalizers...")
        self.state_normalizer.fit(train_states)
        self.control_normalizer.fit(train_controls)
        self.residual_normalizer.fit(train_residuals)
        
        # Normalize training data
        train_states_norm = self.state_normalizer.normalize(train_states)
        train_controls_norm = self.control_normalizer.normalize(train_controls)
        train_residuals_norm = self.residual_normalizer.normalize(train_residuals)
        
        # Convert to tensors
        train_states_tensor = torch.FloatTensor(train_states_norm).to(self.device)
        train_controls_tensor = torch.FloatTensor(train_controls_norm).to(self.device)
        train_residuals_tensor = torch.FloatTensor(train_residuals_norm).to(self.device)
        
        # Create data loader
        train_dataset = TensorDataset(train_states_tensor, train_controls_tensor, 
                                      train_residuals_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation loader (if provided)
        val_loader = None
        if val_states is not None:
            val_states_norm = self.state_normalizer.normalize(val_states)
            val_controls_norm = self.control_normalizer.normalize(val_controls)
            val_residuals_norm = self.residual_normalizer.normalize(val_residuals)
            
            val_states_tensor = torch.FloatTensor(val_states_norm).to(self.device)
            val_controls_tensor = torch.FloatTensor(val_controls_norm).to(self.device)
            val_residuals_tensor = torch.FloatTensor(val_residuals_norm).to(self.device)
            val_dataset = TensorDataset(val_states_tensor, val_controls_tensor, 
                                        val_residuals_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
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
            
            # Early stopping
            if val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break
    
    def predict(self, states, controls):
        """
        Predict residual dynamics with input/output denormalization.
        
        Args:
            states: state data [N, 7] or [7]
            controls: control data [N, 3] or [3]
            
        Returns:
            residuals [N, 7] or [7] (denormalized back to original scale)
        """
        # Handle single sample
        single_sample = False
        if states.ndim == 1:
            single_sample = True
            states = states[np.newaxis, :]
            controls = controls[np.newaxis, :]
        
        # Normalize inputs
        states_norm = self.state_normalizer.normalize(states)
        controls_norm = self.control_normalizer.normalize(controls)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states_norm).to(self.device)
        controls_tensor = torch.FloatTensor(controls_norm).to(self.device)
        
        # Predict
        with torch.no_grad():
            residuals_norm = self.network(states_tensor, controls_tensor)
        
        # Convert back to numpy and denormalize
        residuals_np = residuals_norm.cpu().numpy()
        residuals = self.residual_normalizer.denormalize(residuals_np)
        
        if single_sample:
            residuals = residuals[0]
        
        return residuals
    
    def save(self, path):
        """Save trained network and normalizers."""
        checkpoint = {
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'history': self.training_history,
            'state_normalizer': self.state_normalizer.save(),
            'control_normalizer': self.control_normalizer.save(),
            'residual_normalizer': self.residual_normalizer.save()
        }
        torch.save(checkpoint, path)
    
    def load(self, path):
        """Load trained network and normalizers."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_history = checkpoint['history']
        self.state_normalizer.load(checkpoint['state_normalizer'])
        self.control_normalizer.load(checkpoint['control_normalizer'])
        self.residual_normalizer.load(checkpoint['residual_normalizer'])


def extract_dynamics_states(full_state):
    """
    Extract dynamics-relevant states from full 13-state vector.
    
    Full state: [x, y, psi, vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr, rpm, gear, throttle]
    Dynamics states (7): [vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr]
    
    Excludes: x, y, psi (analytically integrated), rpm (computed from wheels),
              gear (discrete, found by check_upshift)
    
    Args:
        full_state: full 13-element state vector
        
    Returns:
        7-element dynamics state vector
    """
    if isinstance(full_state, np.ndarray):
        return full_state[[3, 4, 5, 6, 7, 8, 9]]  # indices for dynamics states
    else:
        return torch.stack([full_state[3], full_state[4], full_state[5], 
                           full_state[6], full_state[7], full_state[8], full_state[9]])


def generate_training_data(baseline_model, real_model, n_samples=1000, params=None):
    """
    Generate training data by running both models and computing residuals.
    Uses reduced 7-state space for dynamics-relevant learning.
    
    Args:
        baseline_model: function(state, control, dt) -> next_state (13-state)
        real_model: function(state, control, dt) -> next_state (13-state)
        n_samples: number of samples to generate
        params: vehicle parameters
        
    Returns:
        (states, controls, residuals) tuples for training
        states: [N, 7] - reduced dynamics states
        controls: [N, 3] - control inputs
        residuals: [N, 7] - residual dynamics in reduced state space
    """
    states = []
    controls = []
    residuals = []
    
    # Initialize state (13-element full state)
    state = np.array([0., 0., 0., 5., 0., 0., 5., 5., 5., 5., 3000., 1., 0.1], dtype=np.float64)
    
    dt = 0.05
    
    for i in range(n_samples):
        # Random control
        u = np.array([
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(0.0, 1.0),
            np.random.uniform(0.0, 0.3)
        ])
        
        try:
            # Get baseline and real predictions (full 13-state)
            if params is not None:
                baseline_next = baseline_model(state.copy(), u, dt, params)
                real_next = real_model(state.copy(), u, dt, params)
            else:
                baseline_next = baseline_model(state.copy(), u, dt)
                real_next = real_model(state.copy(), u, dt)
            
            # Extract reduced 7-state space
            state_dynamics = extract_dynamics_states(state)
            baseline_next_dynamics = extract_dynamics_states(baseline_next)
            real_next_dynamics = extract_dynamics_states(real_next)
            
            # Compute residual in reduced state space
            residual = real_next_dynamics - baseline_next_dynamics
            
            states.append(state_dynamics.copy())
            controls.append(u.copy())
            residuals.append(residual.copy())
            
            # Update full state for next iteration
            state = real_next.copy()
            
        except Exception as e:
            print(f"Warning at sample {i}: {e}")
            continue
    
    return np.array(states), np.array(controls), np.array(residuals)


if __name__ == "__main__":
    print("Residual Dynamics Network Test")
    print("="*70)
    
    # Create network with 7-state reduced space
    network = ResidualDynamicsNetwork(state_dim=7, control_dim=3, output_dim=7)
    print(f"Network created with {sum(p.numel() for p in network.parameters())} parameters")
    
    # Create learner with normalizer and L2 regularization
    learner = ResidualDynamicsLearner(state_dim=7, control_dim=3, l2_reg=1e-5)
    print("Learner created with input/output normalization and L2 regularization")
    
    # Generate dummy training data (7-state space)
    print("\nGenerating dummy training data (7-state space)...")
    np.random.seed(42)
    
    n_train = 100
    # 7 dynamics states: [vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr]
    train_states = np.random.randn(n_train, 7) * 0.5
    train_states[:, 0] = 5.0 + np.random.randn(n_train) * 0.5  # vx centered at 5
    
    train_controls = np.random.randn(n_train, 3) * 0.1
    train_residuals = np.random.randn(n_train, 7) * 0.01
    
    print(f"Train data: states {train_states.shape}, controls {train_controls.shape}, residuals {train_residuals.shape}")
    
    # Train
    print("\nTraining network with normalization and L2 regularization...")
    learner.fit(train_states, train_controls, train_residuals,
                epochs=50, batch_size=16, verbose=True)
    
    # Test prediction
    print("\nTesting prediction...")
    test_state = np.random.randn(7) * 0.5
    test_state[0] = 5.0
    test_control = np.random.randn(3) * 0.1
    
    prediction = learner.predict(test_state, test_control)
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction sample: {prediction}")
    
    # Show normalizer statistics
    print("\nNormalizer statistics:")
    print(f"  State mean: {learner.state_normalizer.mean}")
    print(f"  State std: {learner.state_normalizer.std}")
    print(f"  Control mean: {learner.control_normalizer.mean}")
    print(f"  Control std: {learner.control_normalizer.std}")
