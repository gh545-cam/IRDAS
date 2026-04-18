#!/usr/bin/env python3
"""Debug RLS adaptation with detailed logging."""
import numpy as np
from params import *
from irdas_main import IRDAS

# Create baseline params
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

print("Creating IRDAS...")
irdas = IRDAS(baseline_params, device='cpu', use_nn=False, use_rls=True)
irdas.initialize_real_vehicle(seed=42)

print("\nBaseline parameters:")
print(f"  TYRE_LAT_a2: {irdas.baseline_params['TYRE_LAT']['a2']:.4f}")
print(f"  TYRE_LAT_B:  {irdas.baseline_params['TYRE_LAT']['B']:.4f}")

print("\nInitial adapter parameters:")
params = irdas.param_adapter.get_current_params()
print(f"  TYRE_LAT_a2: {params['TYRE_LAT']['a2']:.4f}")
print(f"  TYRE_LAT_B:  {params['TYRE_LAT']['B']:.4f}")
print(f"  Param vector: {irdas.param_adapter.param_vector}")
print(f"  Covariance diag: {np.diag(irdas.param_adapter.P)}")

print("\n" + "="*60)
print("Testing RLS update")
print("="*60)

# Simulate slip angles and forces
slip_angles = np.array([5.0, 5.0, 3.0, 3.0])  # degrees
slip_ratios = np.array([0.05, 0.05])           # dimensionless
Fz_kN_wheels = np.array([2.0, 2.0, 2.2, 2.2]) # kN

# Simulate errors (tyres degraded)
lat_force_error = 500.0   # N (tyres producing less lateral force)
lon_force_error = 300.0   # N

print(f"\nInput to update_rls:")
print(f"  slip_angles: {slip_angles}")
print(f"  slip_ratios: {slip_ratios}")
print(f"  Fz_kN_wheels: {Fz_kN_wheels}")
print(f"  lat_force_error: {lat_force_error}")
print(f"  lon_force_error: {lon_force_error}")

# Compute phi directly
phi = irdas.param_adapter._compute_pacejka_regressor(slip_angles, slip_ratios, Fz_kN_wheels)
print(f"\nRegressor phi: {phi}")
print(f"  phi[0] (lat_B):   {phi[0]:.6f}")
print(f"  phi[1] (lat_a2):  {phi[1]:.6f}")
print(f"  phi[2] (lon_B):   {phi[2]:.6f}")
print(f"  phi[3] (lon_a2):  {phi[3]:.6f}")

# Do update
irdas.param_adapter.update_rls(
    slip_angles, slip_ratios, Fz_kN_wheels,
    lat_force_error, lon_force_error,
    acceleration_error=0.5,
    speed_error=0.3,
    adaptive_factor=0.98
)

print("\nAfter 1 update:")
params = irdas.param_adapter.get_current_params()
print(f"  TYRE_LAT_a2: {params['TYRE_LAT']['a2']:.4f} (was {baseline_params['TYRE_LAT']['a2']:.4f})")
print(f"  TYRE_LAT_B:  {params['TYRE_LAT']['B']:.4f} (was {baseline_params['TYRE_LAT']['B']:.4f})")
print(f"  Param vector: {irdas.param_adapter.param_vector}")
print(f"  Covariance diag: {np.diag(irdas.param_adapter.P)}")

print("\nDoing 10 more updates...")
for i in range(10):
    irdas.param_adapter.update_rls(
        slip_angles, slip_ratios, Fz_kN_wheels,
        lat_force_error, lon_force_error,
        acceleration_error=0.5,
        speed_error=0.3,
        adaptive_factor=0.98
    )

params = irdas.param_adapter.get_current_params()
print(f"\nAfter 11 total updates:")
print(f"  TYRE_LAT_a2: {params['TYRE_LAT']['a2']:.4f}")
print(f"  TYRE_LAT_B:  {params['TYRE_LAT']['B']:.4f}")
print(f"  Param vector: {irdas.param_adapter.param_vector}")
print(f"  Covariance diag: {np.diag(irdas.param_adapter.P)}")

# Check if they've changed
if abs(params['TYRE_LAT']['a2'] - baseline_params['TYRE_LAT']['a2']) > 0.001:
    print("\n✓ RLS is working - parameters have adapted!")
else:
    print("\n✗ RLS not working - parameters haven't changed!")
