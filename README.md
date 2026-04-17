# IRDAS: In-race Data Augmentation System

A comprehensive system for online vehicle model correction and state estimation using Neural Networks, Kalman Filtering, and parameter adaptation. Designed to adapt a baseline twin-track vehicle dynamics model to real-world conditions during racing/driving scenarios.

## System Architecture

IRDAS integrates four key components:

### 1. **Real Vehicle Simulator** (`simulator.py`)
- Generates synthetic "real" vehicle data with parameter mismatches
- Simulates tire coefficient variations, mass changes, aerodynamic differences
- Creates the ground truth for training and validation

### 2. **Extended Kalman Filter** (`kalman_filter.py`)
- State estimation from noisy sensor measurements (IMU, GPS, wheel speed sensors)
- Handles combined longitudinal and lateral vehicle dynamics
- Reduces measurement noise and provides smooth state estimates

### 3. **Residual Dynamics Neural Network** (`residual_network.py`)
- Learns the difference between baseline model and real vehicle
- Trained on simulated data with various driving conditions
- Provides correction terms during runtime for improved prediction

### 4. **Online Parameter Adapter** (`parameter_adapter.py`)
- Recursive Least Squares (RLS) adaptation for tire parameters
- Real-time tuning of vehicle mass, aerodynamic coefficients
- Bounds-constrained optimization to prevent unrealistic values

### 5. **Integrated IRDAS System** (`irdas_main.py`)
- Orchestrates all components
- Manages data flow and timing
- Provides metrics and performance tracking

## System Model

**Vehicle State Vector (13 elements):**
```
[x, y, ψ, vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr, rpm, gear, throttle]
```

Where:
- `x, y`: Global position
- `ψ`: Yaw angle
- `vx, vy`: Longitudinal and lateral velocity (body frame)
- `r`: Yaw rate
- `vw_*`: Wheel speeds (front-left, front-right, rear-left, rear-right)
- `rpm`: Engine RPM
- `gear`: Current gear (1-8)
- `throttle`: Throttle input (0-1)

**Control Input (3 elements):**
```
[δ_steer, throttle_pedal, brake_pedal]
```

**Measurements (8 elements):**
```
[ax, ay, r, vx_gps, vy_gps, rpm, wheel_speed_avg]
```

## Installation

```bash
# Clone or navigate to the project directory
cd c:\Users\Gabriel Ho\Desktop\IRDAS

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Test (No Neural Network)

```python
from irdas_main import IRDAS
from params import *

# Create baseline parameters
baseline_params = {
    'L': L, 'TF': TF, 'TR': TR, 'H': H, 'MX': MX, 'M': M,
    'Cd': Cd, 'Cl': Cl, 'Area': Area, 'CX': CX,
    'K': K, 'final_drive': final_drive, 'tyre_radius': tyre_radius,
    'TYRE_LAT': TYRE_LAT.copy(), 'TYRE_LON': TYRE_LON.copy(),
    'GEAR_RATIOS': GEAR_RATIOS.copy(), 'UPSHIFT_SPEED_KPH': UPSHIFT_SPEED_KPH.copy(),
    'ENGINE_RPM': ENGINE_RPM.copy(), 'ENGINE_TORQUE_NM': ENGINE_TORQUE_NM.copy()
}

# Initialize IRDAS (Kalman Filter + Parameter Adaptation only)
irdas = IRDAS(baseline_params, device='cpu', use_nn=False, use_rls=True)

# Initialize real vehicle simulator with mismatched parameters
irdas.initialize_real_vehicle(seed=42)

# Run simulation
irdas.simulate(n_steps=1000, control_strategy='random')

# Get metrics
metrics = irdas.get_metrics()
print(f"Average model error: {metrics['avg_model_error']:.6f}")
print(f"Average estimation error: {metrics['avg_estimation_error']:.6f}")
```

### 2. Full Training and Testing

```bash
# Run comprehensive training and evaluation
python train_test.py --mode full --device cpu

# Or quick test (2 short scenarios)
python train_test.py --mode quick

# Or train neural network only
python train_test.py --mode train-only --device cpu
```

## Usage Examples

### Example 1: Online Parameter Adaptation

```python
from irdas_main import IRDAS
from params import *

# Setup
baseline_params = {...}  # as above
irdas = IRDAS(baseline_params, use_rls=True)
irdas.initialize_real_vehicle(true_params={'M': 800, 'Cd': 0.85})

# Simulate
for i in range(1000):
    control = np.array([steering_angle, throttle, brake])
    state = irdas.step(control, use_param_adaptation=True)

# Check adapted parameters
changes = irdas.param_adapter.get_parameter_changes()
for param_name, info in changes.items():
    print(f"{param_name}: {info['change_pct']:+.2f}%")
```

### Example 2: State Estimation with Kalman Filter

```python
from kalman_filter import ExtendedKalmanFilter, add_sensor_noise
from params import *

# Create baseline parameters
baseline_params = {...}

# Initialize Kalman Filter
ekf = ExtendedKalmanFilter(baseline_params)

# Simulate with measurement noise
for i in range(500):
    # Generate noisy measurement
    noise_std = {
        'ax': 0.5, 'ay': 0.5, 'r': 0.01,
        'vx': 0.1, 'vy': 0.1, 'rpm': 50, 'wheel_speed': 0.2
    }
    measurement = add_sensor_noise(true_state, noise_std)
    
    # Predict
    ekf.predict(control, dt=0.05)
    
    # Update
    ekf.update(measurement)
    
    # Get estimated state and uncertainty
    state_estimate = ekf.get_state()
    uncertainty = ekf.get_uncertainty()
```

### Example 3: Residual Dynamics Learning

```python
from residual_network import ResidualDynamicsLearner, generate_training_data
from simulator import RealVehicleSimulator
from twin_track import twin_track_model

# Create learner
learner = ResidualDynamicsLearner(device='cpu')

# Generate training data
real_sim = RealVehicleSimulator()
states, controls, residuals = generate_training_data(
    baseline_model=twin_track_model,
    real_model=real_sim.step,
    n_samples=2000
)

# Train
learner.fit(states, controls, residuals, epochs=100, batch_size=32)

# Predict residuals
residual_prediction = learner.predict(test_state, test_control)

# Get corrected dynamics
corrected_dynamics = baseline_dynamics + residual_prediction
```

## Performance Tuning

### Kalman Filter Tuning

```python
# Adjust process noise (Q) for model confidence
process_noise = np.diag([
    0.01, 0.01,                    # position (x, y)
    0.01,                          # yaw angle
    0.5, 0.5, 0.01,               # velocities
    1.0, 1.0, 1.0, 1.0,           # wheel speeds (higher = trust measurements more)
    50.0, 0.0, 0.01               # engine_rpm, gear, throttle
])

# Adjust measurement noise (R) for sensor reliability
measurement_noise = np.diag([
    1.0, 1.0,      # IMU accelerations (higher = less trust in IMU)
    0.05,          # yaw rate
    0.5, 0.5,      # GPS velocity
    100.0,         # RPM sensor
    0.5            # wheel speed
])

ekf = ExtendedKalmanFilter(baseline_params, 
                           process_noise=process_noise,
                           measurement_noise=measurement_noise)
```

### Neural Network Architecture

```python
from residual_network import ResidualDynamicsNetwork

# Custom network with different hidden dimensions
network = ResidualDynamicsNetwork(
    state_dim=13,
    control_dim=3,
    hidden_dims=[256, 256, 128],  # Larger network
    dropout_rate=0.3               # More regularization
)
```

### Parameter Adaptation Tuning

```python
from parameter_adapter import OnlineParameterAdapter

adapter = OnlineParameterAdapter(
    baseline_params,
    learning_rate=0.05,        # Faster adaptation
    memory_horizon=200         # Consider more history
)

# Adjust RLS forgetting factor (higher = more recent data emphasized)
adapter.update_rls(observation, measurement, adaptive_factor=0.95)  # 95% = aggressive
```

## Key Features

### ✅ Online Learning
- Parameters adapt in real-time as new data arrives
- No need for full retraining during races

### ✅ State Estimation
- Kalman filter provides smooth estimates from noisy sensors
- Uncertainty quantification available

### ✅ Model Correction
- Neural network learns systematic model errors
- Residual dynamics approach is modular and interpretable

### ✅ Parameter Tuning
- Tire coefficients, mass, aero parameters adapt
- Bounds ensure physically realistic values

### ✅ Simulation Validation
- Integrated simulator creates realistic mismatches
- Multiple test scenarios included

## System Components

| File | Purpose |
|------|---------|
| `params.py` | Vehicle parameters and constants |
| `twin_track.py` | Baseline vehicle dynamics model |
| `simulator.py` | Real vehicle simulator with parameter mismatch |
| `kalman_filter.py` | Extended Kalman filter for state estimation |
| `residual_network.py` | Neural network for residual dynamics |
| `parameter_adapter.py` | Online parameter adaptation (RLS) |
| `irdas_main.py` | Main integrated IRDAS system |
| `train_test.py` | Training and testing scripts |

## Performance Metrics

The system tracks:
- **Model Error**: Difference between baseline model and real vehicle
- **Estimation Error**: Difference between true and estimated state
- **Parameter Changes**: Percentage deviation from baseline
- **State Trajectories**: Complete history for analysis

## Advanced Usage

### Save/Load Models

```python
# Save trained network
irdas.nn_learner.save('models/my_model.pt')

# Load pretrained network
irdas.load_pretrained_network('models/my_model.pt')
```

### Export Results

```python
# Save simulation results
irdas.save_results('results/simulation_run.pkl')

# Analyze offline
import pickle
with open('results/simulation_run.pkl', 'rb') as f:
    data = pickle.load(f)
    
true_states = data['history']['true_states']
estimated_states = data['history']['estimated_states']
model_errors = data['history']['model_errors']
```

### Custom Control Strategies

```python
# Implement custom control law
def my_control_law(state, t):
    # State-feedback control
    steering = -0.05 * state[4]  # proportional to vy
    throttle = 0.5 + 0.2 * (5.0 - state[3])  # speed control
    brake = max(0, -0.1 * state[4]) if state[3] > 1.0 else 0.0
    return np.array([steering, throttle, brake])

# Use in simulation
for t in np.arange(0, 30, irdas.dt):
    control = my_control_law(irdas.kalman_filter.get_state(), t)
    irdas.step(control)
```

## Troubleshooting

**Neural network training is slow:**
- Reduce `n_training_samples` or `n_epochs`
- Use `device='cuda'` if GPU available
- Reduce network size via `hidden_dims` parameter

**Kalman filter diverging:**
- Increase process noise (Q) - trust sensors more
- Reduce measurement noise (R) - trust model more
- Check measurement units match expected ranges

**Parameters not adapting:**
- Increase learning rate in `OnlineParameterAdapter`
- Decrease forgetting factor (adaptive_factor closer to 0.9)
- Check that residuals are non-zero

## Citation

If you use IRDAS in your research, please cite:

```
@software{irdas2024,
  title={IRDAS: In-race Data Augmentation System},
  author={Gabriel Ho},
  year={2024},
  url={https://github.com/...}
}
```

## Future Enhancements

- [ ] Recurrent neural networks for temporal dynamics
- [ ] Multiple hypothesis testing for sudden parameter changes
- [ ] Uncertainty propagation through entire pipeline
- [ ] Real sensor integration (CAN bus, IMU hardware)
- [ ] Distributed/federated learning for multiple vehicles
- [ ] GPU optimization for embedded systems

## License

MIT License - See LICENSE file for details

## Support

For issues, questions, or contributions, please contact or open an issue on the project repository.

---

**Version:** 1.0  
**Last Updated:** April 2026  
**Status:** Stable for research/testing
