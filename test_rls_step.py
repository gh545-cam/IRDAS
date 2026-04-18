#!/usr/bin/env python3
"""Debug RLS parameter updates step by step."""
import numpy as np
from params import *
from parameter_adapter import OnlineParameterAdapter

baseline_params = {
    'L': L, 'TF': TF, 'TR': TR, 'H': H, 'MX': MX, 'M': M,
    'Cd': Cd, 'Cl': Cl, 'Area': Area, 'CX': CX,
    'K': K, 'final_drive': final_drive, 'tyre_radius': tyre_radius,
    'TYRE_LAT': TYRE_LAT.copy(), 'TYRE_LON': TYRE_LON.copy(),
}

adapter = OnlineParameterAdapter(baseline_params)

print("Initial state:")
print(f"  param_vector = {adapter.param_vector}")
params = adapter.get_current_params()
print(f"  TYRE_LAT_a2 = {params['TYRE_LAT']['a2']}")

# Simulate a single RLS update with small errors
print("\nSimulating RLS update with small model errors...")
slip_angles = np.array([0.01, 0.02, 0.01, 0.02])  # Front L, Front R, Rear L, Rear R
slip_ratios = np.array([0.01, 0.01])  # rear wheels only
Fz_kN_wheels = np.array([2.0, 2.0, 2.0, 2.0])  # Reasonable tire loads
lateral_error = np.array([0.1, 0.1, 0.1, 0.1])  # Small force errors
longitudinal_error = 0.1  # scalar  # Small errors
accel_error = 0.05
speed_error = 0.1

print(f"Before update: param_vector[1] (TYRE_LAT_a2) = {adapter.param_vector[1]}")
print(f"Before update: P[1,1] = {adapter.P[1, 1]}")

adapter.update_rls(
    slip_angles=slip_angles,
    slip_ratios=slip_ratios,
    Fz_kN_wheels=Fz_kN_wheels,
    lateral_force_error=lateral_error,
    longitudinal_force_error=longitudinal_error,
    acceleration_error=accel_error,
    speed_error=speed_error
)

print(f"After update: param_vector[1] (TYRE_LAT_a2) = {adapter.param_vector[1]}")
print(f"After update: P[1,1] = {adapter.P[1, 1]}")

params = adapter.get_current_params()
print(f"get_current_params() TYRE_LAT_a2 = {params['TYRE_LAT']['a2']}")

# Check if it's within bounds
a2_bounds = adapter.param_bounds['TYRE_LAT_a2']
print(f"\nBounds for TYRE_LAT_a2: {a2_bounds}")
print(f"param_vector[1] within bounds? {a2_bounds[0] <= adapter.param_vector[1] <= a2_bounds[1]}")
