#!/usr/bin/env python3
"""Debug parameter adapter state during lap reset."""
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

INITIAL_STATE = np.array([0., 0., 0., 25., 0., 0., 25., 25., 25., 25., 8000., 4., 0.5])

print("\nAfter IRDAS init:")
print(f"  param_vector[1] = {irdas.param_adapter.param_vector[1]:.6f}")
print(f"  baseline a2 = {irdas.param_adapter.baseline_params['TYRE_LAT']['a2']:.6f}")
print(f"  current_params a2 = {irdas.param_adapter.current_params['TYRE_LAT']['a2']:.6f}")

# Run lap 1
print("\nRunning Lap 1...")
irdas.true_state = INITIAL_STATE.copy()
irdas.kalman_filter.reset(INITIAL_STATE)
irdas.real_simulator.reset_history()

for step in range(10):  # Just 10 steps
    u = np.array([0.1 * np.sin(0.5 * step), 0.8, 0.0])
    output = irdas.step(u, use_nn_correction=False, use_param_adaptation=True)
    if step < 3:
        print(f"  Step {step}: residual norm = {output}")

print(f"\nAfter Lap 1 (10 steps):")
print(f"  param_vector[1] = {irdas.param_adapter.param_vector[1]:.6f}")
current = irdas.param_adapter.get_current_params()
print(f"  current_params a2 = {current['TYRE_LAT']['a2']:.6f}")

# Lap reset (like in race scenario)
print("\nLap Reset (calling reset_history only)...")
irdas.true_state = INITIAL_STATE.copy()
irdas.kalman_filter.reset(INITIAL_STATE)
irdas.real_simulator.reset_history()

print(f"After lap reset:")
print(f"  param_vector[1] = {irdas.param_adapter.param_vector[1]:.6f}")
current = irdas.param_adapter.get_current_params()
print(f"  current_params a2 = {current['TYRE_LAT']['a2']:.6f}")

# Lap 2
print("\nRunning Lap 2...")
for step in range(10):
    u = np.array([0.1 * np.sin(0.5 * step), 0.8, 0.0])
    irdas.step(u, use_nn_correction=False, use_param_adaptation=True)

print(f"\nAfter Lap 2 (10 steps):")
print(f"  param_vector[1] = {irdas.param_adapter.param_vector[1]:.6f}")
current = irdas.param_adapter.get_current_params()
print(f"  current_params a2 = {current['TYRE_LAT']['a2']:.6f}")
