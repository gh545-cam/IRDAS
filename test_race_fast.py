#!/usr/bin/env python3
"""Fast race scenario test - no NN pretraining."""
import numpy as np
from pathlib import Path
from params import *
from irdas_main import IRDAS

# Quick config
N_LAPS = 10  # Just 10 laps for fast test
STEPS_PER_LAP = 200  # 10 seconds per lap
DT = 0.05
INITIAL_STATE = np.array([0., 0., 0., 25., 0., 0., 25., 25., 25., 25., 8000., 4., 0.5])

def theoretical_grip(lap: int, n_laps: int) -> float:
    """Simple tyre degradation model."""
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

print(f"\nRunning {N_LAPS}-lap test")
print("="*70)

rls_a2_history = []
theoretical_grips = []

for lap in range(N_LAPS):
    # Reset state for lap
    reset_state = INITIAL_STATE.copy()
    irdas.true_state = reset_state.copy()
    irdas.kalman_filter.reset(reset_state)
    irdas.real_simulator.reset_history()
    
    # Degrade tyre for this lap
    grip = theoretical_grip(lap, N_LAPS)
    degraded_lat_a2 = irdas.baseline_params['TYRE_LAT']['a2'] * grip
    irdas.real_simulator.true_params['TYRE_LAT'] = irdas.real_simulator.true_params['TYRE_LAT'].copy()
    irdas.real_simulator.true_params['TYRE_LAT']['a2'] = degraded_lat_a2
    
    # Run lap
    for step in range(STEPS_PER_LAP):
        u = np.array([0.1 * np.sin(0.5 * step), 0.8, 0.0])  # Simple controls
        
        try:
            _ = irdas.step(u, use_nn_correction=False, use_param_adaptation=True)
        except Exception as e:
            print(f"Error lap {lap+1} step {step}: {e}")
            break
    
    # Record adapted parameters
    current = irdas.param_adapter.get_current_params()
    rls_a2 = current['TYRE_LAT']['a2']
    deviation = irdas.param_adapter.param_vector[1]
    rls_a2_history.append(rls_a2)
    theoretical_grips.append(grip)
    
    if lap < 3:  # Debug first few laps
        print(f"Lap {lap+1:>2}/{N_LAPS} | "
              f"grip={grip:.3f} | "
              f"RLS a2={rls_a2:.4f} | "
              f"deviation={deviation:+.6f} | "
              f"Change: {(rls_a2 - irdas.baseline_params['TYRE_LAT']['a2'])*100/irdas.baseline_params['TYRE_LAT']['a2']:+.1f}%")
    else:
        print(f"Lap {lap+1:>2}/{N_LAPS} | "
              f"Theoretical grip: {grip:.3f} | "
              f"RLS a2: {rls_a2:.4f} | "
              f"Change: {(rls_a2 - irdas.baseline_params['TYRE_LAT']['a2'])*100/irdas.baseline_params['TYRE_LAT']['a2']:+.1f}%")

print("\n" + "="*70)
print("RESULTS:")
print(f"Baseline TYRE_LAT_a2:  {irdas.baseline_params['TYRE_LAT']['a2']:.4f}")
print(f"Initial RLS a2:        {rls_a2_history[0]:.4f}")
print(f"Final RLS a2:          {rls_a2_history[-1]:.4f}")
print(f"Final theoretical:     {theoretical_grips[-1]:.4f}")

# Check if RLS tracked degradation
rls_changes = np.array(rls_a2_history) - irdas.baseline_params['TYRE_LAT']['a2']
theoretical_changes = np.array(theoretical_grips) - 1.0

corr = np.corrcoef(rls_changes, theoretical_changes)[0, 1]
print(f"\nCorrelation (RLS vs theory): {corr:.3f}")
if corr > 0.5:
    print("✓ RLS successfully tracked tyre degradation!")
elif corr > 0.2:
    print("~ RLS partially tracked degradation")
else:
    print("✗ RLS did not track degradation well")
