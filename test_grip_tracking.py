#!/usr/bin/env python3
"""Fast grip tracking test - diagnose oscillation behavior."""
import numpy as np
from pathlib import Path
from params import *
from irdas_main import IRDAS

N_LAPS = 30
STEPS_PER_LAP = 200  # 10 seconds per lap
DT = 0.05
INITIAL_STATE = np.array([0., 0., 0., 25., 0., 0., 25., 25., 25., 25., 8000., 4., 0.5])

def theoretical_grip(lap: int, n_laps: int) -> float:
    """Tire degradation model."""
    wear_rate = 0.012
    thermal_optimal = 90.0
    thermal_penalty = 0.003
    
    temp = thermal_optimal * min(lap / 3.0, 1.0)
    wear_factor = max(1.0 - wear_rate * lap, 0.5)
    temp_factor = 1.0 - thermal_penalty * abs(temp - thermal_optimal)
    temp_factor = max(temp_factor, 0.7)
    
    return wear_factor * temp_factor

# Setup
baseline_params = {
    'L': L, 'TF': TF, 'TR': TR, 'H': H, 'MX': MX, 'M': M,
    'Cd': Cd, 'Cl': Cl, 'Area': Area, 'CX': CX,
    'K': K, 'final_drive': final_drive, 'tyre_radius': tyre_radius,
    'TYRE_LAT': TYRE_LAT.copy(), 'TYRE_LON': TYRE_LON.copy(),
    'GEAR_RATIOS': GEAR_RATIOS.copy(),
    'UPSHIFT_SPEED_KPH': UPSHIFT_SPEED_KPH.copy(),
    'ENGINE_RPM': ENGINE_RPM.copy(),
    'ENGINE_TORQUE_NM': ENGINE_TORQUE_NM.copy()
}

print("Initializing IRDAS (no NN pretraining)...")
irdas = IRDAS(baseline_params, device='cpu', use_nn=False, use_rls=True)
irdas.initialize_real_vehicle(seed=42)

print(f"\nTracking grip over {N_LAPS} laps\n")
print(f"{'Lap':>3} {'Grip':>6} {'RLS a2':>8} {'P[1,1]':>8} {'Deviation':>10} {'Error':>8}")
print("="*60)

rls_a2_history = []
theoretical_grips = []
covariance_history = []

for lap in range(N_LAPS):
    # Reset state for lap
    reset_state = INITIAL_STATE.copy()
    irdas.true_state = reset_state.copy()
    irdas.kalman_filter.reset(reset_state)
    irdas.real_simulator.reset_history()
    
    # Degrade tyre
    grip = theoretical_grip(lap, N_LAPS)
    degraded_lat_a2 = irdas.baseline_params['TYRE_LAT']['a2'] * grip
    irdas.real_simulator.true_params['TYRE_LAT'] = irdas.real_simulator.true_params['TYRE_LAT'].copy()
    irdas.real_simulator.true_params['TYRE_LAT']['a2'] = degraded_lat_a2
    
    # Run lap
    for step in range(STEPS_PER_LAP):
        u = np.array([0.1 * np.sin(0.5 * step), 0.8, 0.0])
        try:
            _ = irdas.step(u, use_nn_correction=False, use_param_adaptation=True)
        except Exception as e:
            print(f"Error lap {lap+1} step {step}: {e}")
            break
    
    # Record
    current = irdas.param_adapter.get_current_params()
    rls_a2 = current['TYRE_LAT']['a2']
    rls_a2_history.append(rls_a2)
    theoretical_grips.append(grip)
    covariance_history.append(irdas.param_adapter.P[1, 1])
    
    error = abs(rls_a2 - degraded_lat_a2) / degraded_lat_a2 * 100
    
    print(f"{lap+1:3d} {grip:6.3f} {rls_a2:8.4f} {irdas.param_adapter.P[1,1]:8.4f} "
          f"{irdas.param_adapter.param_vector[1]:+10.6f} {error:8.1f}%")

print("\n" + "="*60)

# Check for oscillations
rls_changes = np.diff(rls_a2_history)
print(f"\nGrip tracking analysis:")
print(f"  RLS range: {min(rls_a2_history):.4f} to {max(rls_a2_history):.4f}")
print(f"  RLS change range: {min(rls_changes):.6f} to {max(rls_changes):.6f}")
print(f"  Covariance range: {min(covariance_history):.6f} to {max(covariance_history):.6f}")

# Correlation
rls_changes_theory = np.array(rls_a2_history) - irdas.baseline_params['TYRE_LAT']['a2']
theory_changes = np.array(theoretical_grips) - 1.0
corr = np.corrcoef(rls_changes_theory, theory_changes)[0, 1]
print(f"  Correlation: {corr:.3f}")

if max(rls_changes) > 0.05:
    print(f"  WARNING: Large RLS oscillations detected (max change {max(abs(rls_changes)):.6f})")
