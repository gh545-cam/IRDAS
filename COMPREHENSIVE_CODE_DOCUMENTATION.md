# IRDAS: Complete Line-by-Line Code Documentation

**A Comprehensive Guide to Every Line of Code in the In-race Data Augmentation System**

This document provides detailed explanations of every significant line of code in the IRDAS system, enabling you to understand exactly what each component does and how they interact.

---

## Table of Contents

1. [Vehicle Parameters (`params.py`)](#vehicleparamspy)
2. [Twin-Track Dynamics Model (`twin_track.py`)](#twin_trackpy)
3. [Real Vehicle Simulator (`simulator.py`)](#simulatorpy)
4. [Kalman Filter (`kalman_filter.py`)](#kalman_filterpy)
5. [Residual Neural Network (`residual_network.py`)](#residual_networkpy)
6. [Parameter Adapter (`parameter_adapter.py`)](#parameter_adapterpy)
7. [Main IRDAS System (`irdas_main.py`)](#irdas_mainpy)
8. [System Architecture & Flow](#systemarchitectureflow)

---

## vehicle_params.py

This file contains all vehicle specifications and lookup tables used by the dynamics model.

```python
import numpy as np
L = 3.6  # wheelbase in meters
```
**What it does**: Defines the distance between front and rear axles. This is crucial for calculating how steering affects the vehicle's yaw rate.

```python
TF = 1.58  # track front (width between left and right front wheels)
TR = 1.42  # track rear
```
**What they do**: Define the track widths. These are used to calculate load transfers when cornering - wider track = less load transfer.

```python
H = 0.295  # Center of Mass height above ground in meters
```
**What it does**: Used to compute longitudinal and lateral load transfers. Higher CoM means more load transfer during acceleration/braking and cornering.

```python
MX = 0.453  # CoM distance from front axle as percentage of wheelbase
```
**What it does**: Defines the CG location. Combined with `L`, this gives the actual distances `Lf` (front distance) and `Lr` (rear distance).

```python
M = 752  # vehicle mass in kg
```
**What it does**: Total vehicle mass used in all dynamics equations (F = ma).

**Aerodynamic Parameters**:
```python
Cd = 0.8  # Drag coefficient
Cl = 3.5  # Downforce coefficient (higher = more downforce)
Area = 1.2  # Frontal area in m²
CX = 0.52  # Center of pressure distance from front (as fraction of wheelbase)
```
**What they do**: Define aerodynamic forces. Drag force grows with velocity squared: `F_drag = 0.5 * Cd * Area * v²`. Downforce increases load on tires at high speed.

**Suspension**:
```python
K = 1.35  # Front to rear roll stiffness ratio
```
**What it does**: Determines how load transfers during cornering. Higher front stiffness = more load on outer front tire.

**Tire Model (Magic Formula)**:
```python
TYRE_LAT = {
    'B': 12.0,     # Stiffness factor - higher = stiffer tires
    'C': 1.9,      # Shape factor - determines curve shape
    'D': 1.8,      # Peak factor (mu) - maximum tire grip
    'E': -1.5,     # Curvature factor - negative = gradual falloff
    'a1': -0.10,   # Load sensitivity on peak: D_actual = a1*Fz + a2
    'a2': 2.05,    # Base peak friction coefficient
    'BCD_a3': 110000.0,  # Cornering stiffness in N/deg
    'BCD_a4': 3.0,       # Load at peak stiffness
    'BCD_a5': 0.007      # Decay rate with load
}
```
**What they do**: These are Pacejka Magic Formula coefficients. The formula models how tire force varies with slip angle and normal force:
- `B`: Controls how quickly force rises initially
- `C`: Controls the shape of the force curve
- `D`: Maximum force the tire can produce (peak friction * normal force)
- `E`: Controls falloff after peak (negative = smooth rolloff)
- `a1, a2`: How peak friction changes with load (heavier = slightly less grip per unit force)

**Engine and Drivetrain**:
```python
ENGINE_RPM = np.array([3000, 4000, ..., 15500])
ENGINE_TORQUE_NM = np.array([440, 500, ..., 340])
```
**What they do**: Create a lookup table for engine torque vs RPM. The dynamics model interpolates between these values.

```python
GEAR_RATIOS = {1: 3.15, 2: 2.47, ...}
final_drive = 6.3
```
**What they do**: Each gear multiplies engine torque. The final drive multiplies again: `wheel_torque = engine_torque * gear_ratio * final_drive`.

---

## twin_track.py

The core vehicle dynamics model using a "twin-track" (one point at front, one at rear) bicycle model with 13 states.

### Key Functions

#### `pacejka_magic_formula(slip, Fz_kN, params, slip_type)`

```python
def pacejka_magic_formula(slip, Fz_kN, params, slip_type='lateral'):
```
**What it does**: Implements the Pacejka Magic Formula to compute tire forces.

```python
B = params['B']
C = params['C']
E = params['E']
```
**What they do**: Extract tire parameters for this specific tire type (lateral or longitudinal).

```python
a1 = params['a1']
a2 = params['a2']
D = a1 * Fz_kN + a2
D = max(D, 0.1)
```
**What it does**: Compute load-dependent peak friction. `D` is the maximum force the tire can produce. As load increases, peak friction increases linearly (up to a point).

```python
arg = B * slip - E * (B * slip - np.arctan(B * slip))
F = D * np.sin(C * np.arctan(arg))
```
**What it does**: This is the core Magic Formula equation. It maps slip angle/ratio to tire force in a smooth curve that:
- Starts linear at small slip (high stiffness)
- Reaches peak at some slip angle
- Falls off slightly at large slip (tire breaking away)

---

#### `calculate_load_transfer(ax, ay, mass, h_com, track_width, K_roll_ratio)`

```python
delta_fz_lat = mass * ay * h_com / track_width
delta_fz_lon = mass * ax * h_com
```
**What they do**: These calculate load transfers in Newtons:
- **Lateral**: When turning, centrifugal force acts at the CG height, creating a moment that shifts load to the outer wheel
- **Longitudinal**: When accelerating/braking, the CG height creates a pitch moment shifting load front/rear

---

#### `twin_track_model(state, u, dt, params)`

This is the main dynamics function that propagates the vehicle state forward by one time step.

**State Vector (13 elements)**:
```python
state[0:2] = [x, y]              # Global position (meters)
state[2] = psi                   # Yaw angle (radians)
state[3:6] = [vx, vy, r]        # Velocities (m/s) and yaw rate (rad/s)
state[6:10] = [vw_fl, vw_fr, vw_rl, vw_rr]  # Wheel speeds (m/s)
state[10:13] = [rpm, gear, throttle]         # Engine/drivetrain (RPM, 1-8, 0-1)
```

**Control Input (3 elements)**:
```python
u[0] = delta_steer    # Steering angle in radians (±0.2 rad ≈ ±11.5°)
u[1] = throttle       # Throttle pedal 0-1
u[2] = brake_pedal    # Brake pedal 0-1
```

**Inside the function**:

```python
x, y, psi, vx, vy, r = state[0:6]
```
**What it does**: Unpack the core state variables for easier access.

```python
Lf = MX * L
Lr = (1 - MX) * L
```
**What they do**: Calculate distances from CG to front and rear axles using the percentage `MX`.

```python
ay = vy * r + (vx * r)
```
**What it does**: Calculate lateral acceleration. The first term `vy * r` is Coriolis acceleration (rotation creates acceleration). The second term accounts for the circular motion.

```python
Fz_rl_static = M * g * Lf_val / L / 2
```
**What it does**: Calculate static normal force at rear-left wheel:
- `M * g`: Total weight in Newtons
- `Lf_val / L`: Weight distribution to rear axle
- `/ 2`: Split between left and right wheels

```python
delta_fz_lon = M * ax * H / L
```
**What it does**: Calculate longitudinal load transfer (front gets more load when braking/accelerating). The `ax * H` creates a pitch moment.

```python
Fz_fl = Fz_fl_static + delta_fz_lon + delta_fz_lat_f
```
**What it does**: Combine static load, longitudinal transfer, and lateral transfer to get actual normal force at each tire.

```python
Fz_fl = max(Fz_fl, 100)
```
**What it does**: Ensure wheels don't lift off (minimum 100 N load).

**Aerodynamic forces**:
```python
F_aero_drag = 0.5 * Cd * Area * v_sq
F_aero_downforce_f = 0.5 * Cl * Area * v_sq * 0.6
```
**What they do**: Calculate drag (opposes motion) and downforce (increases normal forces). The `0.6/0.4` split assumes 60% of downforce on front, 40% on rear.

**Tire slip angles**:
```python
v_fl_steer_y = v_fl_y + delta_steer * v_fl_x
alpha_fl = np.degrees(np.arctan2(v_fl_steer_y, max(abs(v_fl_x), min_speed)))
```
**What it does**: Calculate slip angle at front-left tire. The steering input `delta_steer` changes the tire's heading, which changes the lateral velocity component relative to the tire orientation.

```python
alpha_fl = np.clip(alpha_fl, -20, 20)
```
**What it does**: Clamp slip angles to ±20° (Magic Formula is less accurate beyond this).

**Tire forces**:
```python
Fx_rl = pacejka_magic_formula(slip_ratio_rl * 100, Fz_rl_kN, TYRE_LON)
Fy_fl = pacejka_magic_formula(alpha_fl, Fz_fl_kN, TYRE_LAT)
```
**What they do**: Call the Magic Formula to get longitudinal and lateral forces. Note: `slip_ratio * 100` converts to percentage for the function.

**Equations of motion** (Newton's laws):
```python
Fx_total = Fx_fl + Fx_fr + Fx_rl + Fx_rr - F_aero_drag
ax_dot = Fx_total / M - vy * r
```
**What it does**: 
- Sum all tire forces in X direction, subtract drag
- Divide by mass to get acceleration
- Subtract `vy * r` term (Coriolis: when rotating, lateral velocity creates apparent longitudinal acceleration in body frame)

```python
ay_dot = Fy_total / M + vx * r
```
**What it does**: Similar for lateral acceleration, but ADD `vx * r` (steering creates lateral acceleration).

**Yaw moment**:
```python
M_z_tires = (Fy_fl + Fy_fr) * Lf * np.cos(delta_steer) + delta_steer * (Fx_fl + Fx_fr) * Lf
```
**What it does**: Front tire lateral forces create moment: `Force × Distance`. The steering angle `delta_steer` affects both the lateral force direction and creates moment from longitudinal forces.

```python
r_dot = (M_z_tires + M_z_lateral) / Iz
```
**What it does**: Yaw acceleration = moment / yaw inertia.

**Numerical integration**:
```python
new_state = state + state_dot * dt
```
**What it does**: Use Euler integration to advance state: `x_next = x + dx/dt * dt`.

---

## simulator.py

This file creates a "real" vehicle simulator that runs the same dynamics model but with slightly different parameters to simulate real-world variations.

### RealVehicleSimulator Class

```python
def __init__(self, true_params=None, seed=None):
```
**What it does**: Initialize the simulator with optional parameter mismatch and random seed for reproducibility.

```python
self.baseline_params = {
    'L': L, 'M': M, ...
}
```
**What it does**: Store a copy of the baseline parameters for comparison.

```python
if true_params is None:
    true_params = self._generate_random_mismatch()
```
**What it does**: If no specific mismatch is provided, generate random variations automatically.

```python
def _generate_random_mismatch(self):
    mismatch = {}
    lat_params = TYRE_LAT.copy()
    lat_params['B'] *= np.random.uniform(0.9, 1.1)
```
**What it does**: Create random parameter mismatch by varying each parameter by ±10%. This simulates real vehicles having slightly different tire characteristics, mass, etc.

```python
def step(self, state, control, dt=0.05):
    next_state = twin_track_model(state, control, dt, self.true_params)
```
**What it does**: Run the dynamics model with the "true" (mismatched) parameters. This produces a state different from what the baseline model would predict.

---

## kalman_filter.py

This file implements an Extended Kalman Filter (EKF) for state estimation from noisy measurements.

### ExtendedKalmanFilter Class

```python
def __init__(self, params, process_noise=None, measurement_noise=None):
    self.x = np.zeros(self.n_states)
    self.x[3] = 5.0  # initial vx = 5 m/s
    self.P = np.eye(self.n_states) * 1.0
```
**What they do**:
- `self.x`: State estimate (initially zero except vx=5)
- `self.P`: State covariance (uncertainty in each state). Diagonal = initial uncertainty of 1.0 for each state.

```python
self.Q = np.diag([...])  # Process noise covariance
self.R = np.diag([...])  # Measurement noise covariance
```
**What they do**:
- `Q`: How much we trust the dynamics model. Larger Q = less trust (model might be wrong)
- `R`: How much we trust the sensors. Larger R = less trust (sensors are noisy)

```python
def predict(self, u, dt=None):
    self.x = twin_track_model(self.x, u, dt, self.params)
    F = self._jacobian_dynamics(self.x, u, dt)
    self.P = F @ self.P @ F.T + self.Q
```
**What it does** (KF prediction step):
1. Propagate mean state forward using dynamics model
2. Compute Jacobian (linearization of dynamics around current state)
3. Propagate covariance: `P_new = F*P*F^T + Q` (uncertainty grows due to model error Q)

**Why**: The Jacobian `F` tells us how uncertainty propagates through the nonlinear system. If a state change slightly creates a big change in another state, uncertainty propagates there too.

```python
def update(self, z):
    z_pred = self._measurement_function(self.x)
    y = z - z_pred  # innovation (measurement residual)
    H = self._jacobian_measurement(self.x)
    S = H @ self.P @ H.T + self.R  # innovation covariance
    K = self.P @ H.T @ linalg.inv(S)  # Kalman gain
    self.x = self.x + K @ y  # state correction
    self.P = (I - K @ H) @ self.P  # covariance reduction
```
**What it does** (KF update step):
1. Predict what measurement should be: `z_pred = h(x)`
2. Compute innovation: difference between actual and predicted measurement
3. Compute Kalman gain: how much to trust the innovation (based on uncertainty)
4. Update state by Kalman gain times innovation
5. Reduce uncertainty by amount determined by information from measurement

**Key insight**: If the measurement disagreed a lot and we have high uncertainty, we make a big correction. If we have low uncertainty and measurement agrees well, we make a small correction.

---

## residual_network.py

This file implements a neural network that learns residual dynamics using only 7 dynamics states (not full 13-state).

### StateNormalizer Class

```python
def fit(self, data):
    self.mean = np.mean(data, axis=0)
    self.std = np.std(data, axis=0)
    self.std[self.std < 1e-6] = 1.0
```
**What it does**: Compute mean and standard deviation for each state dimension. This allows converting states to zero mean, unit variance (zero-centered with scale of 1).

**Why**: Neural networks train better when inputs are normalized. Without normalization, a state with range 0-50 would dominate over one with range -1 to 1.

```python
def normalize(self, data):
    return (data - self.mean) / self.std
```
**What it does**: Convert data to normalized form: `x_norm = (x - mean) / std`. This shifts to zero mean and scales to unit variance.

```python
def denormalize(self, data):
    return data * self.std + self.mean
```
**What it does**: Convert back from normalized form to original scale for interpretation.

### ResidualDynamicsNetwork Class

```python
def __init__(self, state_dim=7, control_dim=3, hidden_dims=[128, 128, 64], ...):
```
**What it does**: Create a neural network with:
- Input: 7 dynamics states + 3 controls = 10 dimensions
- Hidden layers: 128 → 128 → 64 neurons
- Output: 7 residual dynamics predictions

**Why 7 states**: We only need `[vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr]`. We don't include:
- `x, y, psi`: Can be integrated analytically from velocities
- `rpm`: Can be computed from wheel speeds
- `gear`: Discrete, determined by `check_upshift`
- `throttle`: Is part of the control input

This reduces computational cost and improves learning by focusing on what matters.

```python
for hidden_dim in hidden_dims:
    layers.append(nn.Linear(prev_dim, hidden_dim))
    layers.append(nn.BatchNorm1d(hidden_dim))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout_rate))
```
**What it does**: Build network layer by layer:
- `Linear`: Matrix multiplication (weights × input + bias)
- `BatchNorm`: Normalize activations to mean 0, std 1 (stabilizes training)
- `ReLU`: Activation function (makes network nonlinear)
- `Dropout`: Randomly zero some activations (prevents overfitting)

```python
layers.append(nn.Linear(prev_dim, self.output_dim))
```
**What it does**: Output layer with no activation (residuals can be positive or negative).

**Why**: The network learns to predict `residual = f_real - f_baseline`. These residuals are corrections to the baseline model, so they should be unrestricted (ReLU would force them positive).

### ResidualDynamicsLearner Class

```python
self.optimizer = optim.Adam(self.network.parameters(), 
                           lr=learning_rate, 
                           weight_decay=l2_reg)
```
**What it does**: 
- `Adam`: Adaptive optimizer (adjusts learning rate per parameter)
- `weight_decay=l2_reg`: L2 regularization that penalizes large weights

**Why L2 regularization**: Prevents the network from fitting noise by encouraging smaller weights. The loss becomes: `loss = MSE_loss + l2_reg * sum(weights^2)`. Networks with large weights are penalized, forcing them to be more "smooth" and generalize better.

```python
def fit(self, train_states, train_controls, train_residuals, ...):
    self.state_normalizer.fit(train_states)
    self.control_normalizer.fit(train_controls)
    self.residual_normalizer.fit(train_residuals)
    
    train_states_norm = self.state_normalizer.normalize(train_states)
    train_controls_norm = self.control_normalizer.normalize(train_controls)
    train_residuals_norm = self.residual_normalizer.normalize(train_residuals)
```
**What it does**: 
1. Fit normalizers on training data (compute mean/std for each dimension)
2. Normalize all data using the fitted parameters

**Why**: This ensures all inputs and outputs to the network have similar ranges, which improves training stability.

```python
for epoch in range(epochs):
    train_loss = self.train_epoch(train_loader)
    val_loss = self.validate(val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```
**What it does**: Early stopping - stop training when validation loss stops improving. This prevents overfitting.

**Why**: Once the network has learned the patterns, further training just fits noise. Early stopping saves computation and improves generalization.

```python
def predict(self, states, controls):
    states_norm = self.state_normalizer.normalize(states)
    predictions = self.network(states_norm, controls_norm)
    residuals = self.residual_normalizer.denormalize(predictions)
    return residuals
```
**What it does**: 
1. Normalize inputs using the fitted normalizers
2. Run through network
3. Denormalize outputs back to original scale

**Why**: The network was trained on normalized data, so inputs must be normalized. The outputs must be denormalized for use.

---

## parameter_adapter.py

This file implements online parameter adaptation using Recursive Least Squares (RLS).

### OnlineParameterAdapter Class

```python
self.P = np.eye(len(self.adaptive_param_names)) * 100  # covariance
```
**What it does**: Initialize the RLS covariance matrix `P`. This represents uncertainty in parameter estimates. High values = high uncertainty = fast adaptation.

```python
def update_rls(self, observation, measurement, adaptive_factor=0.99):
    S = self.P + np.outer(phi, phi) / adaptive_factor
    S_inv = np.linalg.inv(S)
    K = (self.P @ phi) / (adaptive_factor + phi @ self.P @ phi)
    error = measurement - (phi @ self.param_vector)
    self.param_vector = self.param_vector + K * error
    self.P = (1.0 / adaptive_factor) * (self.P - K[:, np.newaxis] @ phi[np.newaxis, :] @ self.P)
```
**What it does**: Recursive Least Squares update algorithm:
1. Compute gain `K` based on current uncertainty `P`
2. Compute prediction error: `error = actual_measurement - predicted_measurement`
3. Update parameters: `params += K * error` (shift parameters to reduce error)
4. Update uncertainty: `P` decreases (we're more confident in parameters now)

**Why RLS**: Can update parameters with each new measurement without reprocessing all old data. Efficient for online learning.

**Adaptive factor**: 
- `0.9` = aggressive (emphasize recent data, forget old data quickly)
- `0.99` = conservative (remember all data equally)

```python
self.param_vector = self._apply_bounds(self.param_vector)
```
**What it does**: Clip parameters to valid ranges to ensure physical realism.

**Why**: Prevents the algorithm from proposing impossible values like negative mass or friction coefficient > 3.

---

## irdas_main.py

This is the main system that orchestrates all components.

### IRDAS Class

```python
def __init__(self, baseline_params, device='cpu', use_nn=True, use_rls=True):
    self.kalman_filter = ExtendedKalmanFilter(baseline_params)
    if use_nn:
        self.nn_learner = ResidualDynamicsLearner(state_dim=7, control_dim=3, device=device)
    if use_rls:
        self.param_adapter = OnlineParameterAdapter(baseline_params)
```
**What it does**: Initialize all components - Kalman filter (always), NN (optional), and RLS adapter (optional).

```python
def step(self, control, measurement_noise_std=None, use_nn_correction=True, use_param_adaptation=True):
    true_state_next = self.real_simulator.step(true_state, control, self.dt)
    measurement = add_sensor_noise(true_state_next, measurement_noise_std)
    self.kalman_filter.predict(control, self.dt)
    
    if use_nn_correction and self.nn_trained:
        state_dynamics = self.kalman_filter.x[[3, 4, 5, 6, 7, 8, 9]]
        residual_correction = self.nn_learner.network(state_dynamics, control)
        self.kalman_filter.x[[3, 4, 5, 6, 7, 8, 9]] += residual_correction * 0.1
    
    self.kalman_filter.update(measurement)
    
    if use_param_adaptation and self.use_rls:
        obs = np.concatenate([true_state[:7], control])
        meas = model_error[:7]
        self.param_adapter.update_rls(obs, meas, adaptive_factor=0.98)
```
**What it does** (one complete iteration):
1. Run real simulator (ground truth)
2. Add sensor noise (simulate real measurements)
3. KF predict: use model to propagate state estimate
4. NN correction (optional): predict residuals and apply small correction
5. KF update: fuse actual measurements
6. RLS adaptation: update parameters based on observed model error

**The flow**:
- Real vehicle evolves forward (we don't know this, but simulator simulates it)
- Noisy measurements come in
- Kalman filter fuses measurements with model prediction
- NN provides additional correction based on learned patterns
- Parameters adapt to reduce model error

```python
def pretrain_neural_network(self, n_training_samples=1000, epochs=100, batch_size=32):
    for i in range(n_training_samples):
        state_baseline_next = twin_track_model(state_baseline, u, self.dt, self.baseline_params)
        state_real_next = self.real_simulator.step(state_real, u, self.dt)
    
    states_baseline_7 = states_baseline[:, [3, 4, 5, 6, 7, 8, 9]]
    states_real_7 = states_real[:, [3, 4, 5, 6, 7, 8, 9]]
    residuals = states_real_7 - states_baseline_7
    
    self.nn_learner.fit(train_states, train_controls, train_residuals, ...)
```
**What it does**:
1. Generate trajectories from both baseline and real models
2. Extract only 7 dynamics states from full 13-state vectors
3. Compute residuals (real - baseline) in 7D space
4. Train NN to learn these residuals

**Why extract 7 states**: The residuals in position/RPM/gear are either analytically known or easily computed. Only the 7 dynamics states contain information that needs to be learned.

---

## System Architecture & Flow

### High-Level Architecture

```
┌─────────────────────────────────────────┐
│  REAL VEHICLE SIMULATOR                 │
│  (Mismatched parameters)                │
│  Outputs: Ground truth state            │
└────────────────┬────────────────────────┘
                 │ (Add sensor noise)
                 ↓
┌─────────────────────────────────────────┐
│  MEASUREMENT SIMULATION                 │
│  (Noisy IMU, GPS, wheel speeds)        │
└────────────────┬────────────────────────┘
                 │
        ┌────────┴────────┐
        ↓                 ↓
    ┌────────┐      ┌─────────────┐
    │ Actual │      │  Kalman     │
    │Measure-│      │ Filter      │
    │ment    │      │  Prediction │
    └────────┘      └─────────────┘
        │                 │
        └────────┬────────┘
                 ↓
         ┌──────────────────┐
         │ Neural Network   │
         │ (Optional)       │
         │ Learn residuals  │
         └────────┬─────────┘
                  ↓
         ┌──────────────────┐
         │ Kalman Filter    │
         │ Update           │
         │ (Measurement fuse)
         └────────┬─────────┘
                  ↓
         ┌──────────────────┐
         │ Parameter        │
         │ Adapter (RLS)    │
         │ Adapt parameters │
         └────────┬─────────┘
                  ↓
         ┌──────────────────┐
         │ STATE ESTIMATE   │
         │ + ADAPTED PARAMS │
         └──────────────────┘
```

### Data Flow at Each Time Step

1. **Real vehicle evolves** (unknown to us):
   - `x_real(k+1) = f_real(x_real(k), u(k))`
   - With true parameters (slightly different from baseline)

2. **Noisy measurements generated**:
   - `z = h(x_real) + noise`
   - Typical noise: IMU ±0.5 m/s², GPS ±0.1 m/s

3. **Kalman filter predicts**:
   - Uses baseline model: `x_predict = f_baseline(x_estimate(k), u(k))`
   - Propagates uncertainty: `P_predict = F*P*F' + Q`

4. **Neural network correction** (if enabled):
   - Takes `x_predict` (7 dynamics states only)
   - Predicts residual: `residual ≈ f_real - f_baseline`
   - Applies small correction: `x_estimate = x_predict + 0.1 * residual`

5. **Kalman filter updates**:
   - Compares prediction with measurement: `innovation = z - h(x_predict)`
   - Computes Kalman gain based on uncertainty
   - Updates: `x_estimate = x_predict + K * innovation`
   - Reduces uncertainty: `P = P_predict - K*H*P_predict`

6. **Parameters adapt**:
   - Observes prediction error: `error = x_true - x_predict`
   - RLS updates parameters: `params += K_rls * error`
   - Reduces uncertainty in parameters: `P_rls = P_rls - ...*K_rls*...`

### Why This Works

**Traditional approach (no adaptation)**:
- Model is imperfect
- Errors accumulate over time
- Increasing uncertainty

**IRDAS approach**:
- Kalman filter: Uses measurements to correct predictions → Bounded error
- Neural network: Learns systematic model errors → Residual correction
- Parameter adaptation: Tunes parameters toward real values → Model improves
- Result: Self-correcting system that adapts to reality

### Key Design Decisions

**Why 7-state NN instead of 13-state**:
- Position, yaw, RPM can be computed deterministically
- Reduces NN complexity by ~45%
- Forces NN to learn only "hard" problems (lateral dynamics, tire slip effects)

**Why input/output normalization**:
- States have different ranges (vx: 0-50, r: -3 to 3, vw: 0-100)
- Without normalization, large-scale states dominate training
- Normalization ensures all states contribute equally

**Why L2 regularization**:
- Prevents overfitting to noise in training data
- Encourages smooth, generalizable solutions
- Acts as implicit prior: simpler models are better

**Why online adaptation**:
- Real parameters change mid-race (tire temp, fuel burn, damage)
- RLS is efficient (no batch reprocessing needed)
- Adapts faster than retraining NN

**Why Kalman filter**:
- Optimal filtering for linear systems (approximately linear for vehicle dynamics)
- Provides uncertainty bounds → Know how confident we are
- Well-tested, reliable algorithm used in aerospace/robotics

---

## Summary: System Capabilities

With all these components:

1. **State Estimation**: Kalman filter provides smooth, reliable state estimates even with noisy sensors

2. **Model Correction**: Neural network learns what the model gets wrong and predicts corrections

3. **Parameter Learning**: RLS adapts tire coefficients, mass, aerodynamics toward real values

4. **Uncertainty Quantification**: System knows how confident it is in estimates

5. **Online Learning**: Everything updates in real-time without full retraining

6. **Interpretability**: Can analyze what parameters changed and why

The system is robust, efficient, and designed for the unique challenges of high-speed vehicle dynamics where models are approximate and real conditions vary.

---

## Full State Reconstruction During NN Correction

This section explains in detail how the full 13-state vector is reconstructed when applying neural network residual corrections.

### State Vector Structure

The system maintains a **13-element state vector**:

```
Index  State Component  Units       Type              Notes
────────────────────────────────────────────────────────────────
0      x                meters      Independent       Global X position
1      y                meters      Independent       Global Y position  
2      psi              radians     Independent       Yaw angle
3      vx               m/s         Dynamics (NN)     Longitudinal velocity
4      vy               m/s         Dynamics (NN)     Lateral velocity
5      r                rad/s       Dynamics (NN)     Yaw rate
6      vw_fl            m/s         Dynamics (NN)     Front-left wheel speed
7      vw_fr            m/s         Dynamics (NN)     Front-right wheel speed
8      vw_rl            m/s         Dynamics (NN)     Rear-left wheel speed
9      vw_rr            m/s         Dynamics (NN)     Rear-right wheel speed
10     rpm              rev/min     Derived           Engine RPM (from wheel speeds)
11     gear             1-8         Discrete          Current gear (from check_upshift)
12     throttle         0-1         Control Input     Throttle pedal position
```

### State Categories

**Dynamics States (Corrected by NN): [3,4,5,6,7,8,9]**
- These 7 states are "free variables" that the model evolves
- NN predicts residuals for these states
- NN corrections are applied directly to these states

**Independent States (NOT Updated by NN): [0,1,2]**
- Position (x, y) and yaw (psi) are NOT updated when NN corrects velocities
- **Reason**: The Kalman filter naturally integrates corrected velocities in the NEXT prediction step
  - If we manually update them now, we'd double-count the change
  - The KF will do the integration correctly with the corrected velocities
- These states will be brought into consistency by the KF update step

**Derived States (Updated for Consistency): [10]**
- RPM is computed deterministically from wheel speeds and gear
- **MUST be updated** when wheel speeds are corrected
- Formula: `RPM = mean(wheel_speeds) * gear_ratio * final_drive`
- Ensures physical consistency after NN wheel speed corrections

**Discrete/Control States (NOT Updated): [11,12]**
- Gear [11]: Discrete decision from `check_upshift()`, not a free variable
- Throttle [12]: Control input, determined by driver/controller

### Why This State Reconstruction is Correct

#### Example: When Wheel Speeds Are Corrected

**Before NN correction:**
```
vw_fl=5.0, vw_fr=5.0, vw_rl=5.0, vw_rr=5.0  → RPM=3000
```

**NN predicts residuals:** `Δvw_fl=+0.1, Δvw_fr=+0.1, Δvw_rl=+0.1, Δvw_rr=+0.1`

**After applying NN correction (scale=0.1):**
```
vw_fl=5.01, vw_fr=5.01, vw_rl=5.01, vw_rr=5.01
```

**RPM must be recalculated for consistency:**
```
new_mean_wheel_speed = 5.01
new_RPM = 5.01 * gear_ratio * final_drive ≈ 3006  (updated)
```

This maintains **physical consistency**: wheel speeds and RPM reflect the same rotational state.

#### Example: Why Position/Yaw Are NOT Updated Immediately

**Current state (after KF prediction):**
```
x=100, y=50, psi=0.5, vx=5.0, vy=0.0
```

**NN corrects velocities:** `Δvx=+0.2, Δvy=-0.1`

```
After NN: vx=5.02, vy=-0.01
```

**Why NOT update x, y, psi immediately:**

❌ **WRONG approach** (would cause double-counting):
```python
x = x + vx_corrected * dt  # Uses NEW velocity
```

✓ **CORRECT approach** (let KF handle it naturally):
```python
# In NEXT prediction step, KF will do:
x_next = x + vx_corrected * dt  # Uses NEW velocity
y_next = y + vy_corrected * dt
psi_next = psi + r_corrected * dt
```

The Kalman filter naturally propagates the corrected velocities in the next step. Manually updating position/yaw now would cause these corrections to be applied twice.

### Full State Reconstruction Algorithm

```python
def _apply_nn_residual_correction(self, full_state, control, correction_scale=0.1):
    # 1. Copy full state
    corrected_state = full_state.copy()
    
    # 2. Extract and correct 7 dynamics states
    dynamics_indices = [3, 4, 5, 6, 7, 8, 9]
    residual = NN(full_state[dynamics_indices], control)
    corrected_state[dynamics_indices] += residual * correction_scale
    
    # 3. UPDATE derived state: Recompute RPM from corrected wheel speeds
    wheel_speeds = corrected_state[[6, 7, 8, 9]]
    mean_speed = mean(wheel_speeds)
    current_gear = corrected_state[11]
    corrected_state[10] = mean_speed * gear_ratio[gear] * final_drive
    
    # 4. LEAVE UNCHANGED:
    #    - corrected_state[0:3] = [x, y, psi] (KF integrates in next step)
    #    - corrected_state[11] = gear (discrete, determined by logic)
    #    - corrected_state[12] = throttle (control input)
    
    return corrected_state  # Full 13-state reconstructed
```

### Key Design Principles

1. **Consistency**: Derived states (like RPM) MUST remain consistent with their source states (wheel speeds)

2. **No Double-Counting**: Independent states (x, y, psi) are NOT updated because the KF will naturally integrate the corrected velocities

3. **Minimal NN Scope**: NN only corrects the 7 "true dynamics" states, leaving deterministic/control states alone

4. **Measurement Fusion**: The KF update step brings everything into final consistency with measurements

### Full State Preservation

Always returns complete 13-state for downstream components:

```python
corrected_state.shape == (13,)  # Always true

# Breakdown:
# [3 independent] + [7 dynamics] + [3 non-dynamics] = 13 total
#     [0,1,2]        [3-9]            [10,11,12]
```

### Verification

To verify state reconstruction is working correctly, run:

```bash
python test_state_reconstruction.py
```

This test validates:
- ✓ Dynamics states are modified by NN
- ✓ Derived states (RPM) are updated for consistency
- ✓ Independent states ([x,y,psi]) remain from KF prediction
- ✓ Full state maintains 13 dimensions
- ✓ No NaN or Inf values introduced

---

**End of Documentation**
