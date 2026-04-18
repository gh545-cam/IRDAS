#!/usr/bin/env python3
"""Debug M, Cd, Cl RLS adaptation."""
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

irdas = IRDAS(baseline_params, device='cpu', use_nn=False, use_rls=True)
irdas.initialize_real_vehicle(seed=42)

INITIAL_STATE = np.array([0., 0., 0., 25., 0., 0., 25., 25., 25., 25., 8000., 4., 0.5])
irdas.true_state = INITIAL_STATE.copy()
irdas.kalman_filter.reset(INITIAL_STATE)
irdas.real_simulator.reset_history()

# Manually call RLS with known parameters to see what happens
print("Testing RLS update with M, Cd, Cl signals...\n")

slip_angles = np.array([5.0, 5.0, 3.0, 3.0])
slip_ratios = np.array([0.02, 0.02])
Fz_wheels = np.array([2.0, 2.0, 2.0, 2.0])

# Simulate signals
lat_force_error = 50.0  # N
lon_force_error = -100.0  # N (negative because real is slower)
accel_error = lon_force_error / 25.0  # -4 m/s^2
speed_error = -1.0  # Real vehicle slower by 1 m/s

print(f"Input signals:")
print(f"  lat_force_error: {lat_force_error:.1f} N")
print(f"  lon_force_error: {lon_force_error:.1f} N")
print(f"  accel_error: {accel_error:.4f} m/s^2")
print(f"  speed_error: {speed_error:.4f} m/s")

# Track param_vector before and after
print(f"\nBefore RLS update:")
print(f"  param_vector: {irdas.param_adapter.param_vector}")

# Manually update with debug=True (but our debug needs to print for all i)
# Let me just run one step and see what parameters change
irdas.param_adapter.update_rls(slip_angles, slip_ratios, Fz_wheels, 
                               lat_force_error, lon_force_error,
                               acceleration_error=accel_error,
                               speed_error=speed_error,
                               debug=True)

print(f"\nAfter RLS update:")
print(f"  param_vector: {irdas.param_adapter.param_vector}")

print(f"\nParameter changes:")
for i, name in enumerate(irdas.param_adapter.adaptive_param_names):
    print(f"  {name}: {irdas.param_adapter.param_vector[i]:+.6f}")
