from irdas_main import IRDAS
from twin_track import twin_track_model
from params import *
from irdas_main import IRDAS
from params import *

# Create baseline parameters
baseline_params = {
    'L': L, 'TF': TF, 'TR': TR, 'H': H, 'MX': MX, 'M': M,
    'Cd': Cd, 'Cl': Cl, 'Area': Area, 'CX': CX,
    'K': K, 'final_drive': final_drive, 'tyre_radius': tyre_radius,
    'TYRE_LAT': TYRE_LAT.copy(), 'TYRE_LON': TYRE_LON.copy(),
    'GEAR_RATIOS': GEAR_RATIOS.copy(), 'UPSHIFT_SPEED_KPH': UPSHIFT_SPEED_KPH.copy(),
    'ENGINE_RPM': ENGINE_RPM.copy(), 'ENGINE_TORQUE_NM': ENGINE_TORQUE_NM.copy()
}

from irdas_main import IRDAS
from params import *

# Setup

irdas = IRDAS(baseline_params, use_rls=True)
irdas.initialize_real_vehicle(true_params={'TYRE_LAT'['B']: 12.0, 'M': M * 1.10}, seed=42)
ou_controls  = np.zeros((10000, 3))
steer, throttle = 0.0, 0.85
rng = np.random.default_rng(42)
for i in range(10000):
    # Sinusoidal steering simulates corners — 3 corners per lap
    lap_phase    = (i % 300) / 300
    corner_steer = 0.12 * np.sin(2 * np.pi * lap_phase * 3)
    steer        = np.clip(corner_steer + rng.normal(0, 0.01), -0.20, 0.20)
    throttle     = np.clip(throttle + rng.normal(0, 0.02), 0.75, 1.0)
    ou_controls[i] = [steer, throttle, 0.0]
# Simulate
for i in range(1000):
    control = np.array([steer, throttle, 0])
    state = irdas.step(control, use_param_adaptation=True)

# Check adapted parameters
changes = irdas.param_adapter.get_parameter_changes()
for param_name, info in changes.items():
    print(f"{param_name}: {info['change_pct']:+.2f}%")