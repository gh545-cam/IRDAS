#!/usr/bin/env python3
"""Debug RLS guard condition and model errors."""
import numpy as np
from params import *
from irdas_main import IRDAS
from twin_track import twin_track_model

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

print("Monitoring model errors and RLS updates...\n")

model_error_history = []
rls_triggered = []

for step in range(50):
    u = np.array([0.1 * np.sin(0.5 * step), 0.8, 0.0])
    
    # Manually run the step to capture model errors
    true_state_before = irdas.true_state.copy()
    true_state_next = irdas.real_simulator.step(irdas.true_state, u, irdas.dt)
    irdas.true_state = true_state_next.copy()
    
    measurement = irdas.sensor_sim.measure(true_state_next)
    irdas.kalman_filter.predict(u, irdas.dt)
    irdas.kalman_filter.update(measurement)
    
    from twin_track import twin_track_model
    baseline_next = twin_track_model(true_state_before, u, irdas.dt, irdas.baseline_params)
    model_error = true_state_next - baseline_next
    model_error_norm = np.linalg.norm(model_error[:7])
    
    model_error_history.append(model_error_norm)
    
    triggered = model_error_norm > 1e-12
    rls_triggered.append(triggered)
    
    if step < 5 or step % 10 == 0:
        print(f"Step {step:2d}: model_error_norm = {model_error_norm:.4e}, RLS triggered: {triggered}")

print(f"\nRLS triggered in {sum(rls_triggered)}/{len(rls_triggered)} steps")
print(f"Mean model error: {np.mean(model_error_history):.4e}")
print(f"Max model error: {np.max(model_error_history):.4e}")
print(f"Min model error: {np.min(model_error_history):.4e}")
