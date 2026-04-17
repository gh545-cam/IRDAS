"""
Training and testing scripts for IRDAS system.
Includes scenarios for different driving conditions and performance evaluation.
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

from params import *
from irdas_main import IRDAS


def create_baseline_params():
    """Create baseline vehicle parameters."""
    return {
        'L': L, 'TF': TF, 'TR': TR, 'H': H, 'MX': MX, 'M': M,
        'Cd': Cd, 'Cl': Cl, 'Area': Area, 'CX': CX,
        'K': K, 'final_drive': final_drive, 'tyre_radius': tyre_radius,
        'TYRE_LAT': TYRE_LAT.copy(), 'TYRE_LON': TYRE_LON.copy(),
        'GEAR_RATIOS': GEAR_RATIOS.copy(), 'UPSHIFT_SPEED_KPH': UPSHIFT_SPEED_KPH.copy(),
        'ENGINE_RPM': ENGINE_RPM.copy(), 'ENGINE_TORQUE_NM': ENGINE_TORQUE_NM.copy()
    }


def train_irdas_system(n_pretraining_samples=2000, n_epochs=150, batch_size=64, device='cpu'):
    """
    Train IRDAS system with neural network pretraining.
    
    Args:
        n_pretraining_samples: number of training samples to generate
        n_epochs: neural network training epochs
        batch_size: training batch size
        device: 'cpu' or 'cuda'
        
    Returns:
        trained IRDAS instance
    """
    print("\n" + "="*70)
    print("IRDAS SYSTEM TRAINING")
    print("="*70)
    
    # Create baseline params
    baseline_params = create_baseline_params()
    
    # Initialize IRDAS
    irdas = IRDAS(baseline_params, device=device, use_nn=True, use_rls=True)
    print(f"IRDAS system created (device: {device})")
    
    # Initialize real vehicle simulator with random mismatch
    print("\nInitializing real vehicle simulator...")
    irdas.initialize_real_vehicle(seed=42)
    
    # Pretrain neural network
    print("\nPretraining neural network for residual dynamics...")
    irdas.pretrain_neural_network(
        n_training_samples=n_pretraining_samples,
        epochs=n_epochs,
        batch_size=batch_size
    )
    
    # Save pretrained model
    os.makedirs('models', exist_ok=True)
    model_path = f"models/irdas_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    irdas.nn_learner.save(model_path)
    print(f"\nPretrained model saved to {model_path}")
    
    return irdas


def test_scenario_1_highway_driving(irdas, duration_seconds=30):
    """
    Test Scenario 1: Highway driving with smooth controls.
    
    Args:
        irdas: IRDAS instance
        duration_seconds: simulation duration
    """
    print("\n" + "="*70)
    print("TEST SCENARIO 1: Highway Driving")
    print("="*70)
    
    n_steps = int(duration_seconds / irdas.dt)
    
    # Reset history
    irdas.history = {
        'true_states': [], 'estimated_states': [], 'measured_states': [],
        'controls': [], 'residuals': [], 'model_errors': [], 'param_changes': [],
        'timestamps': []
    }
    irdas.time_step = 0.0
    
    print(f"Running {duration_seconds}s of highway driving ({n_steps} steps)...")
    
    for step in range(n_steps):
        t = step * irdas.dt
        
        # Smooth highway controls
        steering = 0.05 * np.sin(0.5 * np.pi * t)  # gentle lane changes
        throttle = 0.7 + 0.1 * np.sin(0.1 * np.pi * t)  # maintain speed
        brake = 0.0
        
        u = np.array([steering, throttle, brake])
        
        try:
            irdas.step(u, use_nn_correction=True, use_param_adaptation=True)
        except Exception as e:
            print(f"Error at step {step}: {e}")
            break
        
        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1}/{n_steps}")
    
    metrics = irdas.get_metrics()
    print("\nHighway Driving Metrics:")
    _print_metrics(metrics)
    irdas.reset()
    return metrics


def test_scenario_2_aggressive_handling(irdas, duration_seconds=20):
    """
    Test Scenario 2: Aggressive handling with dynamic maneuvers.
    
    Args:
        irdas: IRDAS instance
        duration_seconds: simulation duration
    """
    print("\n" + "="*70)
    print("TEST SCENARIO 2: Aggressive Handling")
    print("="*70)
    
    n_steps = int(duration_seconds / irdas.dt)
    
    # Reset history
    irdas.history = {
        'true_states': [], 'estimated_states': [], 'measured_states': [],
        'controls': [], 'residuals': [], 'model_errors': [], 'param_changes': [],
        'timestamps': []
    }
    irdas.time_step = 0.0
    
    print(f"Running {duration_seconds}s of aggressive handling ({n_steps} steps)...")
    
    for step in range(n_steps):
        t = step * irdas.dt
        
        # Aggressive maneuvers
        # Phase 1 (0-5s): acceleration
        if t < 5:
            steering = 0.0
            throttle = 1.0
            brake = 0.0
        # Phase 2 (5-10s): hard cornering
        elif t < 10:
            steering = 0.15 * np.sin(2 * np.pi * 0.5 * (t - 5))
            throttle = 0.5
            brake = 0.0
        # Phase 3 (10-15s): braking
        elif t < 15:
            steering = 0.0
            throttle = 0.0
            brake = 0.5
        # Phase 4 (15+s): recovery
        else:
            steering = 0.0
            throttle = 0.3
            brake = 0.0
        
        u = np.array([steering, throttle, brake])
        
        try:
            irdas.step(u, use_nn_correction=True, use_param_adaptation=True)
        except Exception as e:
            print(f"Error at step {step}: {e}")
            break
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{n_steps}")
    
    metrics = irdas.get_metrics()
    print("\nAggressive Handling Metrics:")
    _print_metrics(metrics)
    irdas.reset()
    return metrics


def test_scenario_3_slalom(irdas, duration_seconds=25):
    """
    Test Scenario 3: Slalom maneuver for combined lateral and longitudinal control.
    
    Args:
        irdas: IRDAS instance
        duration_seconds: simulation duration
    """
    print("\n" + "="*70)
    print("TEST SCENARIO 3: Slalom Maneuver")
    print("="*70)
    
    n_steps = int(duration_seconds / irdas.dt)
    
    # Reset history
    irdas.history = {
        'true_states': [], 'estimated_states': [], 'measured_states': [],
        'controls': [], 'residuals': [], 'model_errors': [], 'param_changes': [],
        'timestamps': []
    }
    irdas.time_step = 0.0
    
    print(f"Running {duration_seconds}s of slalom driving ({n_steps} steps)...")
    
    for step in range(n_steps):
        t = step * irdas.dt
        
        # Slalom: high frequency steering
        steering = 0.2 * np.sin(2 * np.pi * 1.0 * t)  # 1 Hz steering oscillation
        throttle = 0.6  # constant moderate speed
        brake = 0.0
        
        u = np.array([steering, throttle, brake])
        
        try:
            irdas.step(u, use_nn_correction=True, use_param_adaptation=True)
        except Exception as e:
            print(f"Error at step {step}: {e}")
            break
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{n_steps}")
    
    metrics = irdas.get_metrics()
    print("\nSlalom Metrics:")
    _print_metrics(metrics)
    irdas.reset()
    return metrics


def _print_metrics(metrics):
    """Print metrics in formatted way."""
    print(f"  Average model error: {metrics['avg_model_error']:.6f}")
    print(f"  Max model error: {metrics['max_model_error']:.6f}")
    print(f"  Std model error: {metrics['std_model_error']:.6f}")
    print(f"  Average estimation error: {metrics['avg_estimation_error']:.6f}")
    print(f"  Max estimation error: {metrics['max_estimation_error']:.6f}")
    print(f"  Simulation steps: {metrics['n_steps']}")
    print(f"  Total time: {metrics['total_time']:.1f}s")
    
    if 'final_param_changes' in metrics:
        print("  Final parameter changes:")
        for param, change in metrics['final_param_changes'].items():
            print(f"    {param}: {change:+.2f}%")


def run_full_evaluation(device='cpu'):
    """
    Run full evaluation of IRDAS system across all test scenarios.
    
    Args:
        device: 'cpu' or 'cuda'
    """
    print("\n" + "="*70)
    print("FULL IRDAS SYSTEM EVALUATION")
    print(f"Device: {device}")
    print("="*70)
    
    # Train system
    irdas = train_irdas_system(n_pretraining_samples=1000, n_epochs=100, 
                              batch_size=32, device=device)
    
    # Run test scenarios
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'scenarios': {}
    }
    
    # Scenario 1: Highway
    results['scenarios']['highway'] = test_scenario_1_highway_driving(irdas, duration_seconds=20)
    
    # Scenario 2: Aggressive
    results['scenarios']['aggressive'] = test_scenario_2_aggressive_handling(irdas, duration_seconds=15)
    
    # Scenario 3: Slalom
    results['scenarios']['slalom'] = test_scenario_3_slalom(irdas, duration_seconds=20)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_file = f"results/irdas_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.ndarray, np.integer)):
            return float(obj)
        elif isinstance(obj, float):
            return round(obj, 8)
        else:
            return obj
    
    results_serializable = convert_to_json_serializable(results)
    
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"\nResults saved to {results_file}")
    
    print("\nAverage Performance Across Scenarios:")
    avg_model_error = np.mean([results['scenarios'][s]['avg_model_error'] 
                               for s in results['scenarios']])
    avg_estimation_error = np.mean([results['scenarios'][s]['avg_estimation_error'] 
                                    for s in results['scenarios']])
    
    print(f"  Average model error: {avg_model_error:.6f}")
    print(f"  Average estimation error: {avg_estimation_error:.6f}")
    
    print("\nPer-Scenario Breakdown:")
    for scenario_name, scenario_results in results['scenarios'].items():
        print(f"\n  {scenario_name.upper()}:")
        print(f"    Model error: {scenario_results['avg_model_error']:.6f}")
        print(f"    Estimation error: {scenario_results['avg_estimation_error']:.6f}")


def quick_test():
    """Quick test for debugging."""
    print("\nRunning quick test (2 scenarios, short duration)...")
    
    baseline_params = create_baseline_params()
    irdas = IRDAS(baseline_params, device='cpu', use_nn=False, use_rls=True)  # no NN for speed
    irdas.initialize_real_vehicle(seed=42)
    
    print("\nScenario 1: Quick Highway Test (5s)")
    test_scenario_1_highway_driving(irdas, duration_seconds=5)
    
    print("\nScenario 2: Quick Aggressive Test (5s)")
    test_scenario_2_aggressive_handling(irdas, duration_seconds=5)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='IRDAS Training and Testing')
    parser.add_argument('--mode', choices=['quick', 'full', 'train-only'], 
                       default='quick', help='Execution mode')
    parser.add_argument('--device', choices=['cpu', 'cuda'], 
                       default='cpu', help='Device to use')
    parser.add_argument('--no-nn', action='store_true', 
                       help='Disable neural network (faster for testing)')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        quick_test()
    elif args.mode == 'full':
        run_full_evaluation(device=args.device)
    elif args.mode == 'train-only':
        irdas = train_irdas_system(device=args.device)
        print("\nTraining complete!")
        print("Model saved in 'models/' directory")
    
    print("\n" + "="*70)
    print("Execution complete!")
    print("="*70)
