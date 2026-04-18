#!/usr/bin/env python3
"""Minimal RLS update test with detailed tracing."""
import numpy as np
from params import TYRE_LAT, TYRE_LON, M, Cd, Cl
from parameter_adapter import OnlineParameterAdapter

# Simple setup
baseline = {
    'M': M, 'Cd': Cd, 'Cl': Cl,
    'TYRE_LAT': TYRE_LAT.copy(), 
    'TYRE_LON': TYRE_LON.copy(),
}

print("Baseline:")
print(f"  TYRE_LAT_a2 = {baseline['TYRE_LAT']['a2']}")

adapter = OnlineParameterAdapter(baseline)
print(f"After init: param_vector[1] = {adapter.param_vector[1]}")
print(f"After init: current_params TYRE_LAT_a2 = {adapter.current_params['TYRE_LAT']['a2']}")

# Simulate 3 steps
for step in range(3):
    print(f"\n--- STEP {step+1} ---")
    
    # Small random errors to trigger updates
    slip_angles = np.array([0.05, 0.05, 0.02, 0.02])  # Front L, R, Rear L, R
    slip_ratios = np.array([0.01, 0.01])  # Rear L, R
    Fz_wheels = np.array([2.0, 2.0, 2.0, 2.0])  # kN
    lat_error = 50.0  # Small lateral force error (N)
    lon_error = 25.0  # Small longitudinal error (N)
    
    print(f"Before update:")
    print(f"  param_vector[1] = {adapter.param_vector[1]:.6f}")
    print(f"  P[1,1] = {adapter.P[1, 1]:.6f}")
    
    try:
        adapter.update_rls(slip_angles, slip_ratios, Fz_wheels, lat_error, lon_error, debug=True)
        print(f"After update:")
        print(f"  param_vector[1] = {adapter.param_vector[1]:.6f}")
        print(f"  P[1,1] = {adapter.P[1, 1]:.6f}")
        
        params = adapter.get_current_params()
        print(f"  get_current_params TYRE_LAT_a2 = {params['TYRE_LAT']['a2']:.6f}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        break
