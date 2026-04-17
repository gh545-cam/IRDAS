#!/usr/bin/env python3
"""Debug script to identify the broadcast error."""
import numpy as np
import traceback
from params import *
from irdas_main import IRDAS

# Create baseline parameters
baseline_params = {
    'L': L, 'TF': TF, 'TR': TR, 'H': H, 'MX': MX, 'M': M,
    'Cd': Cd, 'Cl': Cl, 'Area': Area, 'CX': CX,
    'K': K, 'final_drive': final_drive, 'tyre_radius': tyre_radius,
    'TYRE_LAT': TYRE_LAT.copy(), 'TYRE_LON': TYRE_LON.copy(),
    'GEAR_RATIOS': GEAR_RATIOS.copy(), 'UPSHIFT_SPEED_KPH': UPSHIFT_SPEED_KPH.copy(),
    'ENGINE_RPM': ENGINE_RPM.copy(), 'ENGINE_TORQUE_NM': ENGINE_TORQUE_NM.copy()
}

print("Creating IRDAS system...")
irdas = IRDAS(baseline_params, device='cpu', use_nn=False, use_rls=True)

print("Initializing real vehicle...")
irdas.initialize_real_vehicle(seed=42)

print("Running first step...")
try:
    u = np.array([0.1, 0.7, 0.0])
    state = irdas.step(u, use_nn_correction=False, use_param_adaptation=True)
    print(f"Success! State: {state}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
