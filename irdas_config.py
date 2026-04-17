"""
IRDAS Configuration Guide
Customize system behavior for your specific use case.
"""

# ============================================================================
# KALMAN FILTER CONFIGURATION
# ============================================================================

# Process Noise (Q matrix) - How much we trust the dynamics model
# Higher values = trust model less, rely more on measurements
KALMAN_PROCESS_NOISE = {
    'position': 0.01,           # Position measurement (x, y)
    'yaw_angle': 0.01,          # Yaw angle
    'velocity': 0.1,            # Longitudinal and lateral velocity
    'yaw_rate': 0.01,           # Yaw rate
    'wheel_speeds': 0.5,        # Wheel speeds (0.5 = medium trust)
    'engine_rpm': 10.0,         # Engine RPM
    'throttle': 0.01            # Throttle
}

# Measurement Noise (R matrix) - Sensor reliability
# Higher values = trust sensor less, rely more on model
KALMAN_MEASUREMENT_NOISE = {
    'imu_accel_x': 0.5,         # IMU longitudinal acceleration
    'imu_accel_y': 0.5,         # IMU lateral acceleration
    'imu_yaw_rate': 0.01,       # IMU yaw rate (very reliable)
    'gps_vx': 0.1,              # GPS longitudinal velocity
    'gps_vy': 0.1,              # GPS lateral velocity
    'rpm_sensor': 50.0,         # Engine RPM sensor (noisy)
    'wheel_speed': 0.2          # Wheel speed sensor
}

# ============================================================================
# NEURAL NETWORK CONFIGURATION
# ============================================================================

NEURAL_NETWORK_CONFIG = {
    'state_dim': 13,                    # Vehicle state dimension
    'control_dim': 3,                   # Control input dimension
    'hidden_layers': [128, 128, 64],    # Hidden layer sizes
    'dropout_rate': 0.2,                # Regularization
    'learning_rate': 1e-3,              # Training learning rate
    'batch_size': 32,                   # Training batch size
    'epochs': 100,                      # Training epochs
    'early_stopping_patience': 20,      # Epochs before stopping
    'use_gpu': False                    # Use GPU (if available)
}

# ============================================================================
# PARAMETER ADAPTATION CONFIGURATION
# ============================================================================

PARAMETER_ADAPTER_CONFIG = {
    'learning_rate': 0.01,              # Gradient descent step size
    'memory_horizon': 100,              # Recent samples to consider
    'adaptive_factor': 0.98,            # RLS forgetting factor (0.9-0.99)
    'update_frequency': 1,              # Update every N steps
    
    # Which parameters to adapt
    'adapt_tire_lat_b': True,           # Tire lateral stiffness
    'adapt_tire_lat_peak': True,        # Tire lateral peak friction
    'adapt_tire_lon_b': True,           # Tire longitudinal stiffness
    'adapt_tire_lon_peak': True,        # Tire longitudinal peak friction
    'adapt_mass': True,                 # Vehicle mass
    'adapt_cd': True,                   # Drag coefficient
    'adapt_cl': True,                   # Downforce coefficient
}

# Parameter bounds (prevent unrealistic values)
PARAMETER_BOUNDS = {
    'tire_lat_b': (8.0, 16.0),          # Stiffness range
    'tire_lat_peak': (1.5, 2.5),        # Peak friction range
    'tire_lon_b': (8.0, 16.0),
    'tire_lon_peak': (1.5, 2.5),
    'mass_ratio': (0.85, 1.15),         # Mass as fraction of baseline
    'cd_ratio': (0.85, 1.15),           # Cd variation
    'cl_ratio': (0.85, 1.15),           # Cl variation
}

# ============================================================================
# REAL VEHICLE SIMULATOR CONFIGURATION
# ============================================================================

# Parameter mismatch ranges for simulation
SIMULATED_MISMATCH = {
    'tire_b_variation': 0.1,            # ±10% tire stiffness
    'tire_peak_variation': 0.05,        # ±5% peak friction
    'mass_variation': 0.05,             # ±5% mass
    'cd_variation': 0.05,               # ±5% drag coefficient
    'cl_variation': 0.05,               # ±5% downforce coefficient
    'engine_torque_variation': 0.02,    # ±2% engine torque
}

# ============================================================================
# SYSTEM-WIDE CONFIGURATION
# ============================================================================

SYSTEM_CONFIG = {
    # Simulation
    'dt': 0.05,                         # Time step (seconds)
    'simulation_length': 30,            # Default simulation length (seconds)
    
    # Device
    'device': 'cpu',                    # 'cpu' or 'cuda'
    
    # Components
    'use_kalman_filter': True,          # Enable KF
    'use_neural_network': True,         # Enable NN
    'use_parameter_adaptation': True,   # Enable RLS
    
    # Sensor simulation
    'add_sensor_noise': True,           # Add noise to measurements
    'sensor_noise_level': 1.0,          # Multiplier for noise std devs
    
    # Logging
    'log_every_n_steps': 10,            # Log metrics every N steps
    'save_history': True,               # Save complete trajectory
    'verbose': True                     # Print progress messages
}

# ============================================================================
# CONTROL STRATEGY CONFIGURATIONS
# ============================================================================

CONTROL_STRATEGIES = {
    'highway': {
        'description': 'Smooth highway driving',
        'steering_frequency': 0.5,      # Hz
        'steering_amplitude': 0.05,     # radians
        'throttle_base': 0.7,           # baseline throttle
        'throttle_variation': 0.1,      # ±10% variation
    },
    
    'aggressive': {
        'description': 'Aggressive performance driving',
        'steering_frequency': 1.0,      # Hz
        'steering_amplitude': 0.2,      # radians
        'throttle_base': 0.8,
        'throttle_variation': 0.2,
    },
    
    'slalom': {
        'description': 'Slalom/cone weaving',
        'steering_frequency': 1.5,      # Hz
        'steering_amplitude': 0.25,     # radians
        'throttle_base': 0.6,
        'throttle_variation': 0.05,
    },
    
    'track': {
        'description': 'Full track driving with speed optimization',
        'steering_frequency': 0.3,      # Hz
        'steering_amplitude': 0.3,      # radians
        'throttle_base': 0.85,
        'throttle_variation': 0.15,
    }
}

# ============================================================================
# TESTING & VALIDATION CONFIGURATION
# ============================================================================

TESTING_CONFIG = {
    # Training data generation
    'n_training_samples': 2000,         # Samples for NN pretraining
    'n_validation_samples': 500,        # Validation samples
    'train_val_split': 0.8,             # 80/20 split
    
    # Test scenarios
    'scenarios': {
        'highway': {'duration': 30, 'control': 'highway'},
        'aggressive': {'duration': 20, 'control': 'aggressive'},
        'slalom': {'duration': 25, 'control': 'slalom'},
        'track': {'duration': 60, 'control': 'track'},
    },
    
    # Performance evaluation
    'eval_metrics': [
        'avg_model_error',
        'max_model_error',
        'avg_estimation_error',
        'parameter_convergence_rate',
        'state_estimation_lag'
    ]
}

# ============================================================================
# FUNCTION: Load and apply configuration
# ============================================================================

def load_config(config_name='default'):
    """
    Load a predefined configuration.
    
    Args:
        config_name: 'default', 'fast', 'accurate', 'realtime'
        
    Returns:
        Configuration dictionary
    """
    
    if config_name == 'fast':
        # Fast but less accurate - for testing
        cfg = {
            'nn_epochs': 50,
            'nn_hidden': [64, 32],
            'rls_factor': 0.95,  # Aggressive adaptation
            'dt': 0.1,           # Larger time step
            'training_samples': 500
        }
    
    elif config_name == 'accurate':
        # Slow but accurate - for final evaluation
        cfg = {
            'nn_epochs': 200,
            'nn_hidden': [256, 256, 128],
            'rls_factor': 0.98,  # Conservative adaptation
            'dt': 0.01,          # Smaller time step
            'training_samples': 5000
        }
    
    elif config_name == 'realtime':
        # Optimized for real-time performance
        cfg = {
            'nn_epochs': 100,
            'nn_hidden': [128],
            'rls_factor': 0.97,
            'dt': 0.05,
            'training_samples': 1000,
            'use_nn': True,      # Must use NN for speed
            'device': 'cuda'     # Should use GPU
        }
    
    else:  # 'default'
        cfg = SYSTEM_CONFIG.copy()
        cfg.update(NEURAL_NETWORK_CONFIG)
        cfg.update(PARAMETER_ADAPTER_CONFIG)
    
    return cfg


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
Example 1: Use fast configuration for testing

from irdas_config import load_config
config = load_config('fast')

irdas = IRDAS(baseline_params, 
              use_nn=config['use_neural_network'],
              use_rls=config['use_parameter_adaptation'])

irdas.pretrain_neural_network(
    n_training_samples=config['training_samples'],
    epochs=config['nn_epochs']
)

---

Example 2: Custom configuration

from irdas_config import KALMAN_PROCESS_NOISE, NEURAL_NETWORK_CONFIG

# Modify for specific conditions
KALMAN_PROCESS_NOISE['wheel_speeds'] = 1.0  # Trust wheel speed less
NEURAL_NETWORK_CONFIG['learning_rate'] = 5e-4  # Slower learning

# Use in system...

---

Example 3: Different scenarios

from irdas_config import CONTROL_STRATEGIES

for scenario_name, strategy in CONTROL_STRATEGIES.items():
    print(f"Testing {scenario_name}...")
    # Run simulation with strategy
"""

if __name__ == "__main__":
    print("IRDAS Configuration Guide")
    print("="*70)
    print("\nAvailable configurations:")
    print("  - 'default': Balanced performance")
    print("  - 'fast': Quick testing (50 NN epochs)")
    print("  - 'accurate': High accuracy (200 NN epochs)")
    print("  - 'realtime': Optimized for real-time (GPU)")
    
    # Show example
    config = load_config('fast')
    print(f"\nExample 'fast' config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
