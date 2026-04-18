#!/usr/bin/env python3
"""Test M, Cd, Cl parameter adaptation in IRDAS."""
import numpy as np
from params import *
from irdas_main import IRDAS

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

print(f"\nReal vehicle parameter differences:")
diffs = irdas.real_simulator.get_parameter_difference()
for name in ['M', 'Cd', 'Cl']:
    if name in diffs:
        pct = diffs[name] / baseline_params[name] * 100 if baseline_params[name] != 0 else 0
        print(f"  {name}: {diffs[name]:+.4f} ({pct:+.1f}%)")

INITIAL_STATE = np.array([0., 0., 0., 25., 0., 0., 25., 25., 25., 25., 8000., 4., 0.5])

print(f"\n{'='*70}")
print(f"Baseline parameters:")
print(f"  M:  {baseline_params['M']:.1f} kg")
print(f"  Cd: {baseline_params['Cd']:.3f}")
print(f"  Cl: {baseline_params['Cl']:.3f}")

# Run a few laps and monitor M, Cd, Cl changes
for lap in range(3):
    irdas.true_state = INITIAL_STATE.copy()
    irdas.kalman_filter.reset(INITIAL_STATE)
    irdas.real_simulator.reset_history()
    
    for step in range(300):  # 300 steps per lap
        u = np.array([0.1 * np.sin(0.5 * step), 0.8, 0.0])
        irdas.step(u, use_nn_correction=False, use_param_adaptation=True)
    
    # Record adapted parameters
    current = irdas.param_adapter.get_current_params()
    
    M_est = current['M']
    Cd_est = current['Cd']
    Cl_est = current['Cl']
    
    M_change = (M_est - baseline_params['M']) / baseline_params['M'] * 100
    Cd_change = (Cd_est - baseline_params['Cd']) / baseline_params['Cd'] * 100
    Cl_change = (Cl_est - baseline_params['Cl']) / baseline_params['Cl'] * 100
    
    print(f"\nLap {lap+1}:")
    print(f"  M:  {M_est:7.1f} kg ({M_change:+6.1f}%)")
    print(f"  Cd: {Cd_est:7.4f} ({Cd_change:+6.1f}%)")
    print(f"  Cl: {Cl_est:7.4f} ({Cl_change:+6.1f}%)")

print(f"\n{'='*70}")
print("M, Cd, Cl adaptation test complete.")
