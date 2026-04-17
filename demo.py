"""
Simple demo script showing IRDAS in action.
Run this to quickly see the system working end-to-end.
"""
import numpy as np
import sys

from params import *
from irdas_main import IRDAS


def main():
    """Run simple IRDAS demo."""
    
    print("\n" + "="*70)
    print("IRDAS DEMO - In-race Data Augmentation System")
    print("="*70)
    
    # Create baseline parameters
    baseline_params = {
        'L': L, 'TF': TF, 'TR': TR, 'H': H, 'MX': MX, 'M': M,
        'Cd': Cd, 'Cl': Cl, 'Area': Area, 'CX': CX,
        'K': K, 'final_drive': final_drive, 'tyre_radius': tyre_radius,
        'TYRE_LAT': TYRE_LAT.copy(), 'TYRE_LON': TYRE_LON.copy(),
        'GEAR_RATIOS': GEAR_RATIOS.copy(), 'UPSHIFT_SPEED_KPH': UPSHIFT_SPEED_KPH.copy(),
        'ENGINE_RPM': ENGINE_RPM.copy(), 'ENGINE_TORQUE_NM': ENGINE_TORQUE_NM.copy()
    }
    
    print("\n1. INITIALIZING IRDAS SYSTEM...")
    print("-" * 70)
    
    # Create IRDAS instance (no NN for speed, just Kalman + parameter adaptation)
    irdas = IRDAS(baseline_params, device='cpu', use_nn=False, use_rls=True)
    print("✓ IRDAS core system initialized")
    print(f"  - State dimension: 13")
    print(f"  - Control dimension: 3")
    print(f"  - Kalman filter: Enabled")
    print(f"  - Parameter adaptation: Enabled")
    
    print("\n2. INITIALIZING REAL VEHICLE SIMULATOR...")
    print("-" * 70)
    
    # Create real vehicle with parameter mismatch
    print("Creating real vehicle with random parameter mismatch...")
    irdas.initialize_real_vehicle(seed=42)
    
    # Display parameter differences
    param_diff = irdas.real_simulator.get_parameter_difference()
    print("\nTrue vehicle parameter differences from baseline:")
    for param_name, diff in param_diff.items():
        pct_change = (diff / irdas.baseline_params.get(param_name.split('_')[0], 1.0)) * 100 if param_name[0].isupper() else 0
        print(f"  - {param_name}: {diff:+.4f}")
    print("✓ Real vehicle simulator ready (simulating real-world conditions)")
    
    print("\n3. RUNNING SHORT DEMONSTRATION...")
    print("-" * 70)
    
    print("Running 30 seconds of highway driving...")
    print("(This will show the system adapting to real vehicle conditions)")
    print()
    
    # Run simulation with periodic reporting
    n_steps = int(30 / irdas.dt)
    
    for step in range(n_steps):
        # Simple control: maintain speed with smooth steering
        t = step * irdas.dt
        steering = 0.1 * np.sin(0.5 * np.pi * t)
        throttle = 0.7
        brake = 0.0
        
        u = np.array([steering, throttle, brake])
        
        try:
            state = irdas.step(u, use_nn_correction=False, use_param_adaptation=True)
        except Exception as e:
            print(f"Error at step {step}: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # Print progress every 10 seconds
        if (step + 1) % int(10 / irdas.dt) == 0:
            elapsed_time = (step + 1) * irdas.dt
            print(f"\n  Time: {elapsed_time:.1f}s")
            print(f"    - Velocity: {state[3]:.2f} m/s ({state[3]*3.6:.1f} km/h)")
            print(f"    - Yaw rate: {state[5]:.3f} rad/s")
            print(f"    - Model error: {irdas.history['model_errors'][-1]:.6f}")
            
            if len(irdas.history['param_changes']) > 0:
                latest_changes = irdas.history['param_changes'][-1]
                print(f"    - Parameter adaptation (last 10s):")
                for param_name, change_info in latest_changes.items():
                    if abs(change_info['change_pct']) > 0.1:  # only show significant changes
                        print(f"      • {param_name}: {change_info['change_pct']:+.2f}%")
    
    print("\n✓ Simulation complete!")
    
    print("\n4. RESULTS & METRICS")
    print("-" * 70)
    
    metrics = irdas.get_metrics()
    
    print(f"\nPerformance Metrics:")
    print(f"  - Total simulation time: {metrics['total_time']:.1f}s")
    print(f"  - Total steps: {metrics['n_steps']}")
    print(f"  - Average model error: {metrics['avg_model_error']:.6f}")
    print(f"  - Max model error: {metrics['max_model_error']:.6f}")
    print(f"  - Model error std dev: {metrics['std_model_error']:.6f}")
    print(f"  - Average state estimation error: {metrics['avg_estimation_error']:.6f}")
    print(f"  - Max state estimation error: {metrics['max_estimation_error']:.6f}")
    
    print(f"\nParameter Adaptation:")
    if 'final_param_changes' in metrics:
        param_changes = metrics['final_param_changes']
        
        print(f"  Final parameter changes from baseline:")
        for param_name, change_pct in param_changes.items():
            if abs(change_pct) > 0.1:
                status = "✓" if abs(change_pct) > 0.5 else "•"
                print(f"    {status} {param_name}: {change_pct:+.2f}%")
        print(f"\n  Note: The system adapted {len([x for x in param_changes.values() if abs(x) > 0.1])} parameters")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    
    print("\nNext Steps:")
    print("  1. Check README.md for detailed documentation")
    print("  2. Run full training: python train_test.py --mode full")
    print("  3. Explore individual components in separate Python scripts")
    print("  4. Customize for your specific vehicle parameters")
    
    print("\nKey Files:")
    print("  - params.py: Vehicle parameters")
    print("  - twin_track.py: Baseline dynamics model")
    print("  - kalman_filter.py: State estimation")
    print("  - residual_network.py: Neural network (with NN disabled in this demo)")
    print("  - parameter_adapter.py: Parameter tuning")
    print("  - irdas_main.py: Main system integration")
    print()


if __name__ == "__main__":
    main()
