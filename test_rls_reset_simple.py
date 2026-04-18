#!/usr/bin/env python3
"""Simple test to verify RLS reset behavior."""
import numpy as np
from params import TYRE_LAT
from parameter_adapter import OnlineParameterAdapter

baseline = {
    'M': 752, 'Cd': 0.8, 'Cl': 3.5,
    'TYRE_LAT': TYRE_LAT.copy(), 
    'TYRE_LON': {'B': 12.0, 'a2': 2.1, 'C': 1.65, 'E': -1.5, 'a1': -0.10},
}

adapter = OnlineParameterAdapter(baseline)

print(f"Initial: param_vector[1] = {adapter.param_vector[1]:.6f}")
print(f"Initial: current_params a2 = {adapter.current_params['TYRE_LAT']['a2']:.6f}")

# Simulate some RLS updates
for step in range(5):
    slip_angle = np.array([5.0, 5.0, 3.0, 3.0])
    slip_ratio = np.array([0.02, 0.02])
    Fz = np.array([2.0, 2.0, 2.0, 2.0])
    adapter.update_rls(slip_angle, slip_ratio, Fz, 50.0, 0.0)

print(f"\nAfter 5 steps:")
print(f"param_vector[1] = {adapter.param_vector[1]:.6f}")
print(f"current_params a2 = {adapter.current_params['TYRE_LAT']['a2']:.6f}")

# NOW RESET
print("\n--- RESET ---")
adapter.param_vector = np.zeros(len(adapter.adaptive_param_names))
adapter.current_params = adapter.baseline_params.copy()

print(f"After reset:")
print(f"param_vector[1] = {adapter.param_vector[1]:.6f}")
print(f"current_params a2 = {adapter.current_params['TYRE_LAT']['a2']:.6f}")
print(f"get_current_params a2 = {adapter.get_current_params()['TYRE_LAT']['a2']:.6f}")

# RLS update after reset
print("\nRLS update after reset:")
adapter.update_rls(slip_angle, slip_ratio, Fz, 50.0, 0.0)

print(f"After 1 step:")
print(f"param_vector[1] = {adapter.param_vector[1]:.6f}")
print(f"current_params a2 = {adapter.current_params['TYRE_LAT']['a2']:.6f}")
