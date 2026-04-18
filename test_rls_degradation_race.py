#!/usr/bin/env python3
"""Test RLS tracking of KNOWN tire degradation in race-like scenario."""
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

# Initialize IRDAS
irdas = IRDAS(baseline_params, device='cpu', use_nn=False, use_rls=True)
irdas.initialize_real_vehicle(seed=42)

INITIAL_STATE = np.array([0., 0., 0., 25., 0., 0., 25., 25., 25., 25., 8000., 4., 0.5])

print("Testing RLS with KNOWN tire degradation")
print("="*70)
print(f"Baseline a2: {baseline_params['TYRE_LAT']['a2']:.4f}")

# Run 3 laps with progressive degradation
degradations = [1.0, 0.95, 0.90]  # Relative multipliers

for lap, deg_factor in enumerate(degradations):
    # APPLY DEGRADATION to the real simulator
    true_lat_a2 = baseline_params['TYRE_LAT']['a2'] * deg_factor
    irdas.real_simulator.true_params['TYRE_LAT'] = irdas.real_simulator.true_params['TYRE_LAT'].copy()
    irdas.real_simulator.true_params['TYRE_LAT']['a2'] = true_lat_a2
    
    # RESET RLS STATE to allow re-adaptation (simulate robust re-initialization)
    # This is a common strategy in adaptive control
    irdas.param_adapter.P = np.eye(len(irdas.param_adapter.adaptive_param_names)) * 5.0
    irdas.param_adapter.param_vector = np.zeros(len(irdas.param_adapter.adaptive_param_names))
    irdas.param_adapter.current_params = irdas.param_adapter.baseline_params.copy()
    
    # Reset for lap
    irdas.true_state = INITIAL_STATE.copy()
    irdas.kalman_filter.reset(INITIAL_STATE)
    irdas.real_simulator.reset_history()
    
    print(f"\nLap {lap+1}: Degradation factor = {deg_factor:.2f}, True a2 = {true_lat_a2:.4f}")
    if lap == 0:
        print(f"  Initial param_vector[1] = {irdas.param_adapter.param_vector[1]:.6f}")
        current_before = irdas.param_adapter.get_current_params()
        print(f"  Initial current a2 = {current_before['TYRE_LAT']['a2']:.4f}")
    
    # Run lap with RLS updates
    for step in range(200):  # 200 steps per lap
        u = np.array([0.1 * np.sin(0.5 * step), 0.8, 0.0])
        irdas.step(u, use_nn_correction=False, use_param_adaptation=True)
    
    # Get RLS estimate
    current = irdas.param_adapter.get_current_params()
    rls_a2 = current['TYRE_LAT']['a2']
    deviation = irdas.param_adapter.param_vector[1]
    error = abs(rls_a2 - true_lat_a2)
    
    print(f"  RLS a2: {rls_a2:.4f} (deviation: {deviation:+.6f})")
    print(f"  Error: {error:.4f} ({error/true_lat_a2*100:.1f}%)")
