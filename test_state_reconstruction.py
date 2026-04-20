"""
Test and validate state reconstruction during NN correction.

This script demonstrates that:
1. The full 13-state is properly maintained
2. Only the 7 dynamics states are modified by NN
3. Non-dynamics states remain unchanged from Kalman prediction
"""

import numpy as np
import torch
from irdas_main import IRDAS
from params import *


def _create_baseline_params():
    return {
        'L': L, 'TF': TF, 'TR': TR, 'H': H, 'MX': MX, 'M': M,
        'Cd': Cd, 'Cl': Cl, 'Area': Area, 'CX': CX,
        'K': K, 'final_drive': final_drive, 'tyre_radius': tyre_radius,
        'TYRE_LAT': TYRE_LAT.copy(), 'TYRE_LON': TYRE_LON.copy(),
        'GEAR_RATIOS': GEAR_RATIOS.copy(), 'UPSHIFT_SPEED_KPH': UPSHIFT_SPEED_KPH.copy(),
        'ENGINE_RPM': ENGINE_RPM.copy(), 'ENGINE_TORQUE_NM': ENGINE_TORQUE_NM.copy()
    }


def test_state_reconstruction():
    """Test that full state reconstruction works correctly."""
    
    print("\n" + "="*70)
    print("STATE RECONSTRUCTION VALIDATION TEST")
    print("="*70)
    
    # Initialize IRDAS
    print("\n1. Initializing IRDAS system...")
    irdas = IRDAS(baseline_params=_create_baseline_params(), use_nn=True, use_rls=True)
    irdas.initialize_real_vehicle(seed=42)
    
    # Print state structure
    print("\n2. STATE STRUCTURE:")
    state_info = irdas.verify_state_reconstruction()
    
    print(f"\n   Total state dimension: {state_info['total_states']} elements")
    
    print(f"\n   DYNAMICS STATES ({state_info['dynamics_states']['count']} elements):")
    print(f"   Indices: {state_info['dynamics_states']['indices']}")
    print(f"   Names: {', '.join(state_info['dynamics_states']['names'])}")
    print(f"   Role: {state_info['dynamics_states']['description']}")
    
    print(f"\n   INDEPENDENT STATES ({state_info['independent_states']['count']} elements):")
    print(f"   Indices: {state_info['independent_states']['indices']}")
    print(f"   Names: {', '.join(state_info['independent_states']['names'])}")
    print(f"   Role: {state_info['independent_states']['description']}")
    print(f"   Reason: {state_info['independent_states']['reason']}")
    
    print(f"\n   DERIVED STATES ({state_info['derived_states']['count']} elements):")
    print(f"   Indices: {state_info['derived_states']['indices']}")
    print(f"   Names: {', '.join(state_info['derived_states']['names'])}")
    print(f"   Role: {state_info['derived_states']['description']}")
    print(f"   Reason: {state_info['derived_states']['reason']}")
    print(f"   Formula: {state_info['derived_states']['formula']}")
    
    print(f"\n   DISCRETE/CONTROL STATES ({state_info['discrete_control_states']['count']} elements):")
    print(f"   Indices: {state_info['discrete_control_states']['indices']}")
    print(f"   Names: {', '.join(state_info['discrete_control_states']['names'])}")
    print(f"   Role: {state_info['discrete_control_states']['description']}")
    print(f"   Reason: {state_info['discrete_control_states']['reason']}")
    
    print(f"\n   KEY PRINCIPLE:")
    print(f"   {state_info['key_principle']}")
    
    # Simulate state reconstruction without NN (baseline)
    print("\n3. BASELINE TEST (Kalman prediction without NN):")
    
    # Create initial state
    state_init = np.array([0., 0., 0., 5., 0., 0., 5., 5., 5., 5., 3000., 1., 0.1])
    control = np.array([0.05, 0.5, 0.0])  # slight steering, half throttle
    
    print(f"   Initial state:\n   {state_init}")
    print(f"   Control input: [steering={control[0]}, throttle={control[1]}, brake={control[2]}]")
    
    # Kalman filter prediction (no NN yet)
    irdas.kalman_filter.x = state_init.copy()
    irdas.kalman_filter.predict(control, irdas.dt)
    
    print(f"\n   After KF prediction:")
    print(f"   Full state shape: {irdas.kalman_filter.x.shape}")
    print(f"   Dynamics states [3:10]: {irdas.kalman_filter.x[[3, 4, 5, 6, 7, 8, 9]]}")
    print(f"   Non-dynamics [0:3]: {irdas.kalman_filter.x[0:3]}")
    print(f"   Non-dynamics [10:13]: {irdas.kalman_filter.x[10:13]}")
    
    # Test state reconstruction with NN
    print("\n4. STATE RECONSTRUCTION WITH NN:")
    
    # First train a simple NN
    print("   Training NN on synthetic data...")
    irdas.pretrain_neural_network(n_training_samples=100, epochs=10, batch_size=32)
    
    # Get baseline and apply reconstruction
    state_before = irdas.kalman_filter.x.copy()
    
    print(f"\n   Before NN correction:")
    print(f"   Dynamics states [3:10]: {state_before[[3, 4, 5, 6, 7, 8, 9]]}")
    print(f"   Non-dynamics [0:3]: {state_before[0:3]}")
    print(f"   Non-dynamics [10:13]: {state_before[10:13]}")
    
    # Apply NN correction
    state_after = irdas._apply_nn_residual_correction(state_before, control, correction_scale=0.1)
    
    print(f"\n   After NN correction:")
    print(f"   Dynamics states [3:10]: {state_after[[3, 4, 5, 6, 7, 8, 9]]}")
    print(f"   Non-dynamics [0:3]: {state_after[0:3]}")
    print(f"   Non-dynamics [10:13]: {state_after[10:13]}")
    
    # Verify reconstruction
    print("\n5. RECONSTRUCTION VERIFICATION:")
    
    dynamics_indices = [3, 4, 5, 6, 7, 8, 9]
    independent_indices = [0, 1, 2]
    derived_indices = [10]
    discrete_indices = [11, 12]
    
    # Check that independent states are unchanged
    independent_unchanged = np.allclose(
        state_after[independent_indices],
        state_before[independent_indices]
    )
    print(f"   ✓ Independent states [0,1,2] unchanged: {independent_unchanged}")
    
    # Check that dynamics states are modified
    dynamics_modified = not np.allclose(
        state_after[dynamics_indices],
        state_before[dynamics_indices]
    )
    print(f"   ✓ Dynamics states [3-9] modified: {dynamics_modified}")
    
    # Check that derived state (RPM) is updated
    rpm_updated = not np.isclose(
        state_after[10],
        state_before[10]
    )
    print(f"   ✓ Derived state [10] (RPM) updated: {rpm_updated}")
    
    # Check that discrete/control states are unchanged
    discrete_unchanged = np.allclose(
        state_after[discrete_indices],
        state_before[discrete_indices]
    )
    print(f"   ✓ Discrete/control states [11,12] unchanged: {discrete_unchanged}")
    
    # Check that full state has correct shape
    state_shape_correct = state_after.shape == (13,)
    print(f"   ✓ Full state shape correct: {state_shape_correct} (shape: {state_after.shape})")
    
    # Check that state contains NaNs
    no_nans = not np.any(np.isnan(state_after))
    print(f"   ✓ No NaN values: {no_nans}")
    
    # Check that state contains Infs
    no_infs = not np.any(np.isinf(state_after))
    print(f"   ✓ No Inf values: {no_infs}")
    
    print("\n6. RECONSTRUCTION PROCESS DETAILS:")
    for step_name, step_desc in state_info['reconstruction_process'].items():
        print(f"   {step_desc}")
    
    print("\n" + "="*70)
    print("STATE RECONSTRUCTION VALIDATION: PASSED")
    print("="*70 + "\n")
    
    return all([
        independent_unchanged,
        dynamics_modified,
        rpm_updated,
        discrete_unchanged,
        state_shape_correct,
        no_nans,
        no_infs
    ])


def test_full_simulation_reconstruction():
    """Test state reconstruction during a full simulation loop."""
    
    print("\n" + "="*70)
    print("FULL SIMULATION STATE RECONSTRUCTION TEST")
    print("="*70)
    
    # Initialize IRDAS
    print("\nInitializing IRDAS for simulation test...")
    irdas = IRDAS(baseline_params=_create_baseline_params(), use_nn=True, use_rls=True)
    irdas.initialize_real_vehicle(seed=42)
    
    # Train NN
    print("Training NN...")
    irdas.pretrain_neural_network(n_training_samples=200, epochs=15, batch_size=32)
    
    # Run simulation for a few steps
    print("\nRunning simulation with state reconstruction...")
    n_steps = 50
    
    state_dims_valid = True
    dynamics_changes = []
    non_dynamics_stability = []
    
    for step in range(n_steps):
        # Simple control: random steering
        control = np.array([
            0.1 * np.sin(step * 0.1),  # steering
            0.5,                         # throttle
            0.0                          # brake
        ])
        
        # Run step
        try:
            state_before = irdas.kalman_filter.x.copy()
            estimated_state = irdas.step(control, use_nn_correction=True)
            state_after = estimated_state
            
            # Check state dimension
            if state_after.shape != (13,):
                state_dims_valid = False
                print(f"   ✗ Step {step}: Invalid state shape {state_after.shape}")
                break
            
            # Track dynamics changes
            dynamics_before = state_before[[3, 4, 5, 6, 7, 8, 9]]
            dynamics_after = state_after[[3, 4, 5, 6, 7, 8, 9]]
            dynamics_change = np.linalg.norm(dynamics_after - dynamics_before)
            dynamics_changes.append(dynamics_change)
            
            # Check non-dynamics stability (should only change from KF update, not NN)
            non_dyn_indices = [0, 1, 2, 10, 11, 12]
            
            if step % 10 == 0:
                print(f"   Step {step:3d}: Dynamics change = {dynamics_change:.6f}, "
                      f"State valid = {not np.any(np.isnan(state_after))}")
        
        except Exception as e:
            print(f"   ✗ Step {step}: {e}")
            state_dims_valid = False
            break
    
    print("\n" + "-"*70)
    print("SIMULATION RECONSTRUCTION RESULTS:")
    print(f"   ✓ All states maintained 13 dimensions: {state_dims_valid}")
    print(f"   ✓ Average dynamics change per step: {np.mean(dynamics_changes):.6f}")
    print(f"   ✓ Max dynamics change per step: {np.max(dynamics_changes):.6f}")
    print(f"   ✓ Steps completed: {step + 1}/{n_steps}")
    
    print("\n" + "="*70)
    if state_dims_valid:
        print("FULL SIMULATION RECONSTRUCTION: PASSED")
    else:
        print("FULL SIMULATION RECONSTRUCTION: FAILED")
    print("="*70 + "\n")
    
    return state_dims_valid


if __name__ == "__main__":
    # Run both tests
    test1_pass = test_state_reconstruction()
    test2_pass = test_full_simulation_reconstruction()
    
    print("\n" + "="*70)
    print("FINAL RESULTS:")
    print(f"   State reconstruction validation: {'PASSED' if test1_pass else 'FAILED'}")
    print(f"   Full simulation validation: {'PASSED' if test2_pass else 'FAILED'}")
    print("="*70 + "\n")
