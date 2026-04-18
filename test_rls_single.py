#!/usr/bin/env python3
"""Single degradation level test with detailed RLS debugging."""
import numpy as np
from params import TYRE_LAT, TYRE_LON
from parameter_adapter import OnlineParameterAdapter
from twin_track import tire_lateral_force

baseline = {
    'M': 752, 'Cd': 0.8, 'Cl': 3.5,
    'TYRE_LAT': TYRE_LAT.copy(), 
    'TYRE_LON': TYRE_LON.copy(),
}

adapter = OnlineParameterAdapter(baseline)

# Test with 5% degradation
deg_factor = 0.95
true_lat_a2 = baseline['TYRE_LAT']['a2'] * deg_factor

print(f"Baseline a2: {baseline['TYRE_LAT']['a2']:.4f}")
print(f"True a2 (95%): {true_lat_a2:.4f}")
print()

slip_angle = np.array([5.0, 5.0, 3.0, 3.0])
slip_ratio = np.array([0.02, 0.02])
Fz_wheels = np.array([2.0, 2.0, 2.0, 2.0])

true_params = baseline['TYRE_LAT'].copy()
true_params['a2'] = true_lat_a2

# First RLS update with debug
F_true_lat = sum(tire_lateral_force(s, Fz_wheels[i], true_params) 
                for i, s in enumerate(slip_angle)) / 4.0
F_pred_lat = sum(tire_lateral_force(s, Fz_wheels[i], baseline['TYRE_LAT']) 
                for i, s in enumerate(slip_angle)) / 4.0
force_error = F_true_lat - F_pred_lat

print(f"F_true: {F_true_lat:.4f}, F_pred: {F_pred_lat:.4f}")
print(f"Force error: {force_error:.4f}")
print()

params_before = adapter.get_current_params()
print(f"Before RLS: a2 = {params_before['TYRE_LAT']['a2']:.4f}")

adapter.update_rls(slip_angle, slip_ratio, Fz_wheels, force_error, 0.0, debug=True)

params_after = adapter.get_current_params()
print(f"\nAfter RLS: a2 = {params_after['TYRE_LAT']['a2']:.4f}")
print(f"Deviation: {adapter.param_vector[1]:.6f}")
