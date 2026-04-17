import numpy as np
from twin_track import twin_track_model
from params import *

params = {
    'L': L, 'TF': TF, 'TR': TR, 'H': H, 'MX': MX, 'M': M,
    'Cd': Cd, 'Cl': Cl, 'Area': Area, 'CX': CX, 'K': K,
    'final_drive': final_drive, 'tyre_radius': tyre_radius,
    'TYRE_LAT': TYRE_LAT, 'TYRE_LON': TYRE_LON,
    'GEAR_RATIOS': GEAR_RATIOS, 'UPSHIFT_SPEED_KPH': UPSHIFT_SPEED_KPH,
    'ENGINE_RPM': ENGINE_RPM, 'ENGINE_TORQUE_NM': ENGINE_TORQUE_NM
}

state = np.array([0., 0., 0., 30., 0., 0., 30., 30., 30., 30., 8000., 4., 0.5])
u = np.array([0.0, 0.6, 0.0])  # straight line, gentle throttle, zero brake

print(f"{'Step':<6} {'vx':>8} {'vy':>8} {'r':>8} {'ax_net':>10}")
for i in range(20):
    state_next = twin_track_model(state, u, 0.05, params)
    ax = (state_next[3] - state[3]) / 0.05
    print(f"{i:<6} {state_next[3]:>8.3f} {state_next[4]:>8.3f} {state_next[5]:>8.3f} {ax:>10.3f}")
    state = state_next