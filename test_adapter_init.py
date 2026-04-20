#!/usr/bin/env python3
"""Debug parameter adapter initialization."""
import numpy as np
from params import TYRE_LAT
from parameter_adapter import OnlineParameterAdapter

baseline_params = {
    'M': 752, 'Cd': 0.8, 'Cl': 3.5,
    'TYRE_LAT': TYRE_LAT.copy(),
    'TYRE_LON': {'B': 12.0, 'a2': 2.1, 'C': 1.65, 'E': -1.5, 'a1': -0.10},
}

print("Baseline TYRE_LAT_a2:", baseline_params['TYRE_LAT']['a2'])
print("Baseline TYRE_LAT_B:", baseline_params['TYRE_LAT']['B'])

adapter = OnlineParameterAdapter(baseline_params)

print("\nAfter initialization:")
params = adapter.get_current_params()
print("Param vector:", adapter.param_vector)
print("Current TYRE_LAT_a2:", params['TYRE_LAT']['a2'])
print("Current TYRE_LAT_B:", params['TYRE_LAT']['B'])
print("Bounds for TYRE_LAT_a2:", adapter.param_bounds['TYRE_LAT_a2'])
print("TYRE_LAT_B is derived from a2 in get_current_params() and has no direct bound entry")

# Check if values are within bounds
print("\nBounds check:")
for i, name in enumerate(adapter.adaptive_param_names):
    # param_vector stores deviations from baseline, while bounds apply to absolute params
    val = adapter._extract_vector(adapter.get_current_params())[i]
    lo, hi = adapter.param_bounds[name]
    in_bounds = lo <= val <= hi
    print(f"  {name}: {val:.4f} in [{lo:.4f}, {hi:.4f}]? {in_bounds}")
