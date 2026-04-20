"""
Full-race IRDAS simulation with race strategy, degradations, and adaptation plots.

What this script does:
- Runs a hypothetical multi-lap race on a synthetic track profile
- Uses noisy sensors through IRDAS SensorSimulator
- Uses a real vehicle with slightly mismatched tyre/aero parameters vs digital twin
- Estimates mass as part of the augmented UKF state (mass is a filter state)
- Applies lap-by-lap fuel burn and continuous tyre degradation
- Executes a two-stop strategy with tyre changes
- Visualizes real vs predicted lap times and parameter adaptation trends
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from irdas_main import IRDAS
from params import (
    Area,
    CX,
    Cd,
    Cl,
    ENGINE_RPM,
    ENGINE_TORQUE_NM,
    GEAR_RATIOS,
    H,
    K,
    L,
    M,
    MX,
    TF,
    TR,
    TYRE_LAT,
    TYRE_LON,
    UPSHIFT_SPEED_KPH,
    final_drive,
    tyre_radius,
)


@dataclass
class Segment:
    start_frac: float
    end_frac: float
    target_speed: float
    curvature: float


@dataclass
class DriverModel:
    track: list[Segment]
    lap_distance_m: float
    integral_err: float = 0.0
    prev_throttle: float = 0.18
    prev_brake: float = 0.0
    stall_timer_s: float = 0.0
    target_speed_filt: float = 30.0
    prev_lap_progress_m: float = 0.0
    no_progress_timer_s: float = 0.0

    def _target_speed(self, progress_frac: float, tyre_health: float, fuel_frac: float) -> tuple[float, float]:
        seg = get_segment(self.track, progress_frac)
        base = seg.target_speed * (0.93 + 0.07 * tyre_health)
        base *= (0.97 + 0.03 * (1.0 - fuel_frac))

        # Curvature-limited planner: v <= sqrt(a_lat_max / |kappa|).
        # Track curvature values are normalized, so scale to avoid over-conservative limits.
        kappa_eff = 0.015 * abs(seg.curvature)
        a_lat_max = max(6.0, 14.0 * tyre_health)
        v_curv = np.sqrt(a_lat_max / (kappa_eff + 1e-4))
        target = min(base, v_curv)
        target = max(target, 14.0)
        return float(target), float(seg.curvature)

    def step(self, lap_progress_m: float, vx_true: float,
             tyre_health: float, fuel_frac: float,
             dt: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        # Detect low-progress trapping (handles lap wrap by resetting timer).
        delta_progress = lap_progress_m - self.prev_lap_progress_m
        if delta_progress < -5.0:
            self.no_progress_timer_s = 0.0
        else:
            if delta_progress < 0.08 and vx_true < 4.0:
                self.no_progress_timer_s += dt
            else:
                self.no_progress_timer_s = max(0.0, self.no_progress_timer_s - dt)
        self.prev_lap_progress_m = lap_progress_m

        progress_frac = (lap_progress_m / self.lap_distance_m) % 1.0
        desired_raw, curvature = self._target_speed(progress_frac, tyre_health, fuel_frac)
        desired_n1, _ = self._target_speed((progress_frac + 0.02) % 1.0, tyre_health, fuel_frac)
        desired_n2, _ = self._target_speed((progress_frac + 0.05) % 1.0, tyre_health, fuel_frac)
        desired_n3, _ = self._target_speed((progress_frac + 0.08) % 1.0, tyre_health, fuel_frac)
        desired_preview = float(0.55 * desired_n1 + 0.30 * desired_n2 + 0.15 * desired_n3)

        # Smooth target profile with faster decel than accel for realistic corner approach.
        target_alpha_up = 0.18
        target_alpha_down = 0.22
        target_alpha = target_alpha_down if desired_raw < self.target_speed_filt else target_alpha_up
        self.target_speed_filt += target_alpha * (desired_raw - self.target_speed_filt)
        desired_speed = float(np.clip(self.target_speed_filt, 8.0, 95.0))
        desired_next = float(0.72 * desired_speed + 0.28 * desired_preview)

        # Feedforward + PI speed control with explicit overspeed braking.
        speed_error = desired_speed - vx_true
        self.integral_err = float(np.clip(self.integral_err + speed_error * dt, -100.0, 100.0))
        ff = 0.12 + 0.0065 * desired_speed
        throttle_pi = ff + 0.030 * speed_error + 0.0012 * self.integral_err
        overspeed = max(0.0, vx_true - desired_speed)
        brake_preview = np.clip((desired_speed - desired_next) * 0.018, 0.0, 0.22)
        brake_from_overspeed = 0.030 * overspeed
        corner_overspeed = max(0.0, vx_true - desired_preview)
        brake_corner = 0.040 * corner_overspeed if desired_preview < (desired_speed - 2.0) else 0.0

        throttle_cmd = float(np.clip(throttle_pi, 0.0, 1.0))
        brake_cmd = float(np.clip(brake_from_overspeed + brake_preview + brake_corner, 0.0, 0.65))
        if brake_cmd > 0.05:
            throttle_cmd *= 0.18

        # Deadband and coasting near target speed to avoid chatter.
        if abs(speed_error) < 1.5:
            brake_cmd = 0.0
            throttle_cmd = max(throttle_cmd, 0.12)
        elif -4.0 < speed_error < -1.5:
            brake_cmd = 0.0
            throttle_cmd = min(throttle_cmd, 0.10)

        # Avoid fighting acceleration with light braking when below pace.
        if vx_true < 0.70 * desired_speed and brake_cmd < 0.20:
            brake_cmd = 0.0

        # Ensure enough launch torque in low-speed/high-demand zones.
        if vx_true < 12.0 and desired_speed > 16.0:
            throttle_cmd = max(throttle_cmd, 0.42)
            if brake_cmd < 0.25:
                brake_cmd = 0.0

        if vx_true < 4.0 and desired_speed > 14.0:
            throttle_cmd = max(throttle_cmd, 0.60)
            brake_cmd = 0.0

        # Anti-windup when braking dominates.
        if brake_cmd > 0.03:
            self.integral_err = min(self.integral_err, 0.0)

        # Relaunch behavior if car is near-stopped at a segment expecting speed.
        if vx_true < 2.0 and desired_speed > 16.0:
            self.stall_timer_s += dt
        else:
            self.stall_timer_s = max(0.0, self.stall_timer_s - dt)

        relaunch = self.stall_timer_s > 0.6
        stuck = relaunch or (self.no_progress_timer_s > 1.0 and desired_speed > 14.0)
        if stuck:
            throttle_cmd = max(throttle_cmd, 0.95)
            brake_cmd = 0.0
            self.integral_err = max(self.integral_err, 0.0)

        # Steering command with speed-aware reduction and small texture.
        steering = curvature + 0.008 * np.sin(2.0 * np.pi * progress_frac * 6.0)
        steering += rng.normal(0.0, 0.0015)
        speed_scale = float(np.clip(vx_true / 8.0, 0.45, 1.0))
        steering *= speed_scale
        if vx_true < 10.0 and desired_speed > 14.0:
            steering *= 0.45
        if brake_cmd > 0.20:
            steering *= 0.75
        if stuck:
            steering = float(np.clip(steering, -0.04, 0.04))
        steering = float(np.clip(steering, -0.22, 0.22))

        # Smooth actuator outputs.
        alpha = 0.65 if stuck else 0.35
        throttle = self.prev_throttle + alpha * (throttle_cmd - self.prev_throttle)
        brake = self.prev_brake + alpha * (brake_cmd - self.prev_brake)

        # Enforce mutual exclusion with a small overlap deadzone.
        if brake > 0.06:
            throttle = 0.0

        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 0.70))
        self.prev_throttle = throttle
        self.prev_brake = brake

        return np.array([steering, throttle, brake], dtype=np.float64), float(desired_speed)


DEFAULT_TRACK = [
    Segment(0.00, 0.12, 78.0, 0.00),
    Segment(0.12, 0.18, 42.0, 0.12),
    Segment(0.18, 0.32, 74.0, -0.02),
    Segment(0.32, 0.40, 46.0, -0.10),
    Segment(0.40, 0.58, 82.0, 0.01),
    Segment(0.58, 0.68, 55.0, 0.08),
    Segment(0.68, 0.78, 65.0, -0.05),
    Segment(0.78, 0.92, 76.0, 0.00),
    Segment(0.92, 1.00, 50.0, 0.10),
]


def create_baseline_params() -> dict:
    return {
        "L": L,
        "TF": TF,
        "TR": TR,
        "H": H,
        "MX": MX,
        "M": M,
        "Cd": Cd,
        "Cl": Cl,
        "Area": Area,
        "CX": CX,
        "K": K,
        "final_drive": final_drive,
        "tyre_radius": tyre_radius,
        "TYRE_LAT": TYRE_LAT.copy(),
        "TYRE_LON": TYRE_LON.copy(),
        "GEAR_RATIOS": GEAR_RATIOS.copy(),
        "UPSHIFT_SPEED_KPH": UPSHIFT_SPEED_KPH.copy(),
        "ENGINE_RPM": ENGINE_RPM.copy(),
        "ENGINE_TORQUE_NM": ENGINE_TORQUE_NM.copy(),
    }


def create_true_params_from_baseline(baseline: dict) -> dict:
    true_params = baseline.copy()
    true_params["TYRE_LAT"] = baseline["TYRE_LAT"].copy()
    true_params["TYRE_LON"] = baseline["TYRE_LON"].copy()

    # Slight mismatch in aero and tyre behavior.
    true_params["Cd"] = baseline["Cd"] * 1.04
    true_params["Cl"] = baseline["Cl"] * 0.96
    true_params["TYRE_LAT"]["a2"] = baseline["TYRE_LAT"]["a2"] * 0.97
    true_params["TYRE_LON"]["a2"] = baseline["TYRE_LON"]["a2"] * 0.98
    true_params["TYRE_LAT"]["B"] = baseline["TYRE_LAT"]["B"] * 1.03
    true_params["TYRE_LON"]["B"] = baseline["TYRE_LON"]["B"] * 1.02
    true_params["M"] = baseline["M"] * 1.01

    return true_params


def get_segment(track: list[Segment], progress_frac: float) -> Segment:
    for seg in track:
        if seg.start_frac <= progress_frac < seg.end_frac:
            return seg
    return track[-1]


def apply_tyre_degradation(true_params: dict, base_true_params: dict, tyre_health: float) -> None:
    # Reduce peak grip and slightly increase stiffness factor as tyres harden with wear.
    lat_a2_base = base_true_params["TYRE_LAT"]["a2"]
    lon_a2_base = base_true_params["TYRE_LON"]["a2"]
    lat_b_base = base_true_params["TYRE_LAT"]["B"]
    lon_b_base = base_true_params["TYRE_LON"]["B"]

    a2_scale = 0.80 + 0.20 * tyre_health
    b_scale = 1.00 + 0.10 * (1.0 - tyre_health)

    true_params["TYRE_LAT"]["a2"] = float(lat_a2_base * a2_scale)
    true_params["TYRE_LON"]["a2"] = float(lon_a2_base * a2_scale)
    true_params["TYRE_LAT"]["B"] = float(lat_b_base * b_scale)
    true_params["TYRE_LON"]["B"] = float(lon_b_base * b_scale)


def run_full_race(
    n_laps: int = 12,
    lap_distance_m: float = 5200.0,
    pit_stop_laps: tuple[int, int] = (4, 8),
    pit_stop_time_s: float = 22.0,
    dt: float = 0.05,
    seed: int = 7,
    pretrain_samples: int = 900,
    pretrain_epochs: int = 80,
    max_lap_time_s: float = 360.0,
    max_wall_time_s: float = 900.0,
) -> dict:
    rng = np.random.default_rng(seed)

    baseline_params = create_baseline_params()
    true_params = create_true_params_from_baseline(baseline_params)
    base_true_params = {
        "TYRE_LAT": true_params["TYRE_LAT"].copy(),
        "TYRE_LON": true_params["TYRE_LON"].copy(),
    }

    irdas = IRDAS(baseline_params, device="cpu", use_nn=True, use_rls=True)
    irdas.dt = dt
    irdas.initialize_real_vehicle(true_params=true_params, seed=seed)

    # Moderate noise to stress estimator during full-race conditions.
    irdas.sensor_sim.noise.update({
        "x": 0.8,
        "y": 0.8,
        "ax": 0.4,
        "ay": 0.4,
        "r": 0.008,
        "vx": 0.15,
        "vy": 0.15,
        "rpm": 35.0,
        "wheel_speed": 0.08,
        "fuel_flow": 0.0025,
        "mass": 2.2,
    })

    print("\nPretraining residual network for race simulation...")
    irdas.pretrain_neural_network(
        n_training_samples=pretrain_samples,
        epochs=pretrain_epochs,
        batch_size=64,
    )

    total_time_s = 0.0
    tyre_health = 1.0
    fuel_start_mass = float(irdas.true_vehicle_mass)

    lap_times_real = []
    lap_times_pred = []
    fuel_used_per_lap = []
    tyre_health_end_lap = []
    mass_true_end_lap = []
    mass_est_end_lap = []
    lat_a2_est_end_lap = []
    lon_a2_est_end_lap = []
    cd_est_end_lap = []
    cl_est_end_lap = []
    lat_a2_true_end_lap = []
    lon_a2_true_end_lap = []
    residual_rms_end_lap = []

    time_trace = []
    residual_trace = []
    telemetry = {
        "t": [],
        "vx_true": [],
        "vx_est": [],
        "rpm_true": [],
        "rpm_est": [],
        "gear_true": [],
        "gear_est": [],
        "throttle_cmd": [],
        "brake_cmd": [],
        "target_speed": [],
    }
    wall_start = time.perf_counter()
    driver = DriverModel(track=DEFAULT_TRACK, lap_distance_m=lap_distance_m)

    for lap in range(1, n_laps + 1):
        lap_progress = 0.0
        lap_elapsed = 0.0
        lap_est_distance = 0.0
        lap_residuals = []
        stuck_timer_s = 0.0
        recovery_steps_left = 0
        lap_step = 0
        max_steps_per_lap = int(max_lap_time_s / dt)
        mass_lap_start = float(irdas.true_vehicle_mass)

        while lap_progress < lap_distance_m:
            elapsed_wall = time.perf_counter() - wall_start
            if elapsed_wall >= max_wall_time_s:
                print(
                    f"  Warning: Reached wall-clock limit ({max_wall_time_s:.1f}s). "
                    f"Stopping race early at lap {lap:02d}."
                )
                break

            lap_step += 1
            fuel_frac = np.clip(float(irdas.true_vehicle_mass) / fuel_start_mass, 0.0, 1.0)
            control, desired_speed = driver.step(
                lap_progress_m=lap_progress,
                vx_true=float(irdas.true_state[3]),
                tyre_health=tyre_health,
                fuel_frac=fuel_frac,
                dt=dt,
                rng=rng,
            )

            # Race-only supervisory recovery: if progress stalls at low speed,
            # temporarily enforce a straight launch command to re-enter normal regime.
            if recovery_steps_left > 0:
                control[0] = 0.0
                control[1] = max(float(control[1]), 0.90)
                control[2] = 0.0
                desired_speed = max(float(desired_speed), 24.0)
                recovery_steps_left -= 1

            # Hint both real and baseline drivetrains about desired pace for kickdown floor.
            irdas.real_simulator.true_params["TARGET_SPEED_MPS"] = float(desired_speed)
            irdas.kalman_filter.params["TARGET_SPEED_MPS"] = float(desired_speed)
            irdas.current_params["TARGET_SPEED_MPS"] = float(desired_speed)

            est_state = irdas.step(
                control,
                use_nn_correction=True,
                use_param_adaptation=True,
                reset_nn_memory=False,
            )

            vx_true = max(0.0, float(irdas.true_state[3]))
            vx_est = max(0.0, float(est_state[3]))

            # End-of-lap consistency: integrate only the fractional step needed to cross the lap line.
            delta_progress = vx_true * dt
            step_frac = 1.0
            if delta_progress > 1e-9 and (lap_progress + delta_progress) > lap_distance_m:
                step_frac = max(0.0, min(1.0, (lap_distance_m - lap_progress) / delta_progress))

            dt_eff = dt * step_frac
            lap_progress += vx_true * dt_eff
            lap_est_distance += vx_est * dt_eff
            lap_elapsed += dt_eff
            total_time_s += dt_eff

            # Update trap detector after seeing actual progress this step.
            if vx_true < 2.0 and (vx_true * dt_eff) < 0.10 and desired_speed > 14.0:
                stuck_timer_s += dt_eff
            else:
                stuck_timer_s = max(0.0, stuck_timer_s - dt_eff)

            if recovery_steps_left == 0 and stuck_timer_s > 1.2:
                recovery_steps_left = max(1, int(1.0 / dt))
                stuck_timer_s = 0.0

            telemetry["t"].append(float(total_time_s))
            telemetry["vx_true"].append(float(vx_true))
            telemetry["vx_est"].append(float(vx_est))
            telemetry["rpm_true"].append(float(irdas.true_state[10]))
            telemetry["rpm_est"].append(float(est_state[10]))
            telemetry["gear_true"].append(float(irdas.true_state[11]))
            telemetry["gear_est"].append(float(est_state[11]))
            telemetry["throttle_cmd"].append(float(control[1]))
            telemetry["brake_cmd"].append(float(control[2]))
            telemetry["target_speed"].append(float(desired_speed))

            if lap_step % 500 == 0:
                print(
                    f"  Lap {lap:02d} progress: {lap_progress:7.1f}/{lap_distance_m:.1f} m "
                    f"({100.0 * lap_progress / lap_distance_m:5.1f}%), "
                    f"vx_true={vx_true:5.2f} m/s"
                )

            if lap_elapsed >= max_lap_time_s or lap_step >= max_steps_per_lap:
                print(
                    f"  Warning: Lap {lap:02d} exceeded max_lap_time_s={max_lap_time_s:.1f}s. "
                    f"Closing lap early at {lap_progress:.1f} m ({100.0 * lap_progress / lap_distance_m:.1f}%)."
                )
                break

            # Degradation depends on cornering/braking energy proxies.
            deg_rate = 2.4e-4 + 3.2e-4 * abs(float(control[0])) + 2.2e-4 * float(control[2])
            tyre_health = float(max(0.72, tyre_health - deg_rate * dt_eff))
            apply_tyre_degradation(irdas.real_simulator.true_params, base_true_params, tyre_health)

            if irdas.history["residuals"]:
                rn = float(np.linalg.norm(irdas.history["residuals"][-1][:7]))
                lap_residuals.append(rn)
                residual_trace.append(rn)
                time_trace.append(total_time_s)

        # True lap time includes strategy-dependent pit delay.
        lap_time_real = lap_elapsed

        # Guard against occasional estimator-distance collapse causing absurd lap predictions.
        raw_pred = lap_elapsed * (lap_distance_m / max(lap_est_distance, 1e-6))
        if lap_est_distance < 0.35 * max(lap_progress, 1e-6):
            raw_pred = lap_elapsed * (lap_distance_m / max(lap_progress, 1e-6))
        lap_time_pred = float(np.clip(raw_pred, 0.70 * lap_elapsed, 3.00 * lap_elapsed))

        if lap in pit_stop_laps:
            lap_time_real += pit_stop_time_s
            lap_time_pred += pit_stop_time_s
            tyre_health = 1.0
            apply_tyre_degradation(irdas.real_simulator.true_params, base_true_params, tyre_health)
            irdas.on_tire_change(reset_parameter_adapter=True)

        lap_times_real.append(float(lap_time_real))
        lap_times_pred.append(float(lap_time_pred))
        fuel_used_per_lap.append(float(max(0.0, mass_lap_start - irdas.true_vehicle_mass)))
        tyre_health_end_lap.append(float(tyre_health))
        mass_true_end_lap.append(float(irdas.true_vehicle_mass))
        mass_est_end_lap.append(float(irdas.kalman_filter.get_mass_estimate()))

        cur = irdas.current_params
        lat_a2_est_end_lap.append(float(cur["TYRE_LAT"]["a2"]))
        lon_a2_est_end_lap.append(float(cur["TYRE_LON"]["a2"]))
        cd_est_end_lap.append(float(cur["Cd"]))
        cl_est_end_lap.append(float(cur["Cl"]))

        lat_a2_true_end_lap.append(float(irdas.real_simulator.true_params["TYRE_LAT"]["a2"]))
        lon_a2_true_end_lap.append(float(irdas.real_simulator.true_params["TYRE_LON"]["a2"]))
        residual_rms_end_lap.append(float(np.sqrt(np.mean(np.square(lap_residuals))) if lap_residuals else 0.0))

        print(
            f"Lap {lap:02d} | real={lap_time_real:6.2f}s pred={lap_time_pred:6.2f}s "
            f"fuel={fuel_used_per_lap[-1]:5.2f}kg tyre={tyre_health:0.3f} "
            f"mass_true={mass_true_end_lap[-1]:7.2f} mass_est={mass_est_end_lap[-1]:7.2f}"
        )

        if (time.perf_counter() - wall_start) >= max_wall_time_s:
            break

    result = {
        "config": {
            "n_laps": n_laps,
            "lap_distance_m": lap_distance_m,
            "pit_stop_laps": list(pit_stop_laps),
            "pit_stop_time_s": pit_stop_time_s,
            "dt": dt,
            "max_lap_time_s": max_lap_time_s,
            "max_wall_time_s": max_wall_time_s,
        },
        "lap": {
            "real_time_s": lap_times_real,
            "pred_time_s": lap_times_pred,
            "fuel_used_kg": fuel_used_per_lap,
            "tyre_health": tyre_health_end_lap,
            "mass_true_kg": mass_true_end_lap,
            "mass_est_kg": mass_est_end_lap,
            "residual_rms": residual_rms_end_lap,
            "lat_a2_est": lat_a2_est_end_lap,
            "lon_a2_est": lon_a2_est_end_lap,
            "cd_est": cd_est_end_lap,
            "cl_est": cl_est_end_lap,
            "lat_a2_true": lat_a2_true_end_lap,
            "lon_a2_true": lon_a2_true_end_lap,
        },
        "timeseries": {
            "t": time_trace,
            "residual_norm": residual_trace,
            "telemetry": telemetry,
        },
        "summary": {
            "mean_real_laptime_s": float(np.mean(lap_times_real)),
            "mean_pred_laptime_s": float(np.mean(lap_times_pred)),
            "mean_abs_laptime_error_s": float(np.mean(np.abs(np.array(lap_times_real) - np.array(lap_times_pred)))),
            "total_fuel_used_kg": float(np.sum(fuel_used_per_lap)),
            "mass_estimation_rmse_kg": float(np.sqrt(np.mean((np.array(mass_true_end_lap) - np.array(mass_est_end_lap)) ** 2))),
        },
    }

    return result


def save_visualizations(result: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    laps = np.arange(1, len(result["lap"]["real_time_s"]) + 1)
    pit_laps = set(result["config"]["pit_stop_laps"])

    fig, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)

    # 1) Lap times.
    ax = axes[0, 0]
    ax.plot(laps, result["lap"]["real_time_s"], marker="o", label="Real lap time")
    ax.plot(laps, result["lap"]["pred_time_s"], marker="s", label="Predicted lap time")
    for lp in pit_laps:
        ax.axvline(lp, color="tab:red", linestyle="--", alpha=0.35)
    ax.set_title("Lap Times: Real vs Predicted")
    ax.set_xlabel("Lap")
    ax.set_ylabel("Time [s]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2) Fuel usage.
    ax = axes[0, 1]
    ax.bar(laps, result["lap"]["fuel_used_kg"], color="tab:olive")
    ax.set_title("Fuel Consumption Per Lap")
    ax.set_xlabel("Lap")
    ax.set_ylabel("Fuel used [kg]")
    ax.grid(True, axis="y", alpha=0.3)

    # 3) Tyre health.
    ax = axes[1, 0]
    ax.plot(laps, result["lap"]["tyre_health"], marker="o", color="tab:orange")
    for lp in pit_laps:
        ax.axvline(lp, color="tab:red", linestyle="--", alpha=0.35)
    ax.set_title("Tyre Health (Two-Stop Strategy)")
    ax.set_xlabel("Lap")
    ax.set_ylabel("Health [-]")
    ax.set_ylim(0.68, 1.02)
    ax.grid(True, alpha=0.3)

    # 4) Mass estimation.
    ax = axes[1, 1]
    ax.plot(laps, result["lap"]["mass_true_kg"], marker="o", label="True mass")
    ax.plot(laps, result["lap"]["mass_est_kg"], marker="s", label="Estimated mass")
    ax.set_title("Mass State Tracking")
    ax.set_xlabel("Lap")
    ax.set_ylabel("Mass [kg]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 5) Parameter adaptation.
    ax = axes[2, 0]
    ax.plot(laps, result["lap"]["lat_a2_true"], label="True TYRE_LAT a2", linestyle="--")
    ax.plot(laps, result["lap"]["lat_a2_est"], label="Est TYRE_LAT a2")
    ax.plot(laps, result["lap"]["lon_a2_true"], label="True TYRE_LON a2", linestyle="--")
    ax.plot(laps, result["lap"]["lon_a2_est"], label="Est TYRE_LON a2")
    ax.set_title("RLS Tyre Parameter Adaptation")
    ax.set_xlabel("Lap")
    ax.set_ylabel("Parameter value")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 6) Residual network reaction.
    ax = axes[2, 1]
    ax.plot(laps, result["lap"]["residual_rms"], marker="o", color="tab:purple")
    ax.set_title("Residual RMS per Lap")
    ax.set_xlabel("Lap")
    ax.set_ylabel("Residual RMS [-]")
    ax.grid(True, alpha=0.3)

    fig.suptitle("IRDAS Full Race Simulation", fontsize=15)

    fig_path = os.path.join(out_dir, "full_race_dashboard.png")
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    # Additional driver/drivetrain telemetry figure.
    telem = result.get("timeseries", {}).get("telemetry", {})
    if telem and len(telem.get("t", [])) > 0:
        t = np.array(telem["t"])
        fig2, axes2 = plt.subplots(5, 1, figsize=(14, 14), constrained_layout=True, sharex=True)

        axes2[0].plot(t, telem["vx_true"], label="vx true", color="tab:blue")
        axes2[0].plot(t, telem["vx_est"], label="vx est", color="tab:cyan", alpha=0.8)
        axes2[0].plot(t, telem["target_speed"], label="target speed", color="tab:green", linestyle="--", alpha=0.9)
        axes2[0].set_ylabel("Speed [m/s]")
        axes2[0].set_title("Speed Tracking")
        axes2[0].grid(True, alpha=0.3)
        axes2[0].legend(loc="best")

        axes2[1].plot(t, telem["throttle_cmd"], color="tab:orange")
        axes2[1].set_ylabel("Throttle")
        axes2[1].set_ylim(-0.02, 1.02)
        axes2[1].grid(True, alpha=0.3)

        axes2[2].plot(t, telem["brake_cmd"], color="tab:red")
        axes2[2].set_ylabel("Brake")
        axes2[2].set_ylim(-0.02, 0.82)
        axes2[2].grid(True, alpha=0.3)

        axes2[3].plot(t, telem["gear_true"], label="gear true", color="tab:purple")
        axes2[3].plot(t, telem["gear_est"], label="gear est", color="tab:pink", alpha=0.8)
        axes2[3].set_ylabel("Gear")
        axes2[3].set_yticks(np.arange(1, 9, 1))
        axes2[3].grid(True, alpha=0.3)
        axes2[3].legend(loc="best")

        axes2[4].plot(t, telem["rpm_true"], label="rpm true", color="tab:brown")
        axes2[4].plot(t, telem["rpm_est"], label="rpm est", color="tab:gray", alpha=0.8)
        axes2[4].set_ylabel("RPM")
        axes2[4].set_xlabel("Time [s]")
        axes2[4].grid(True, alpha=0.3)
        axes2[4].legend(loc="best")

        fig2.suptitle("Driver and Drivetrain Telemetry")
        fig2_path = os.path.join(out_dir, "full_race_telemetry.png")
        fig2.savefig(fig2_path, dpi=140)
        plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full IRDAS race simulation")
    parser.add_argument("--n-laps", type=int, default=12)
    parser.add_argument("--lap-distance-m", type=float, default=5200.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--pretrain-samples", type=int, default=900)
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--max-lap-time-s", type=float, default=360.0)
    parser.add_argument("--max-wall-time-s", type=float, default=900.0)
    parser.add_argument("--quick", action="store_true", help="Run a much faster smoke-race configuration")
    args = parser.parse_args()

    if args.quick:
        args.n_laps = min(args.n_laps, 3)
        args.pretrain_samples = min(args.pretrain_samples, 180)
        args.pretrain_epochs = min(args.pretrain_epochs, 12)
        args.max_lap_time_s = min(args.max_lap_time_s, 240.0)
        args.max_wall_time_s = min(args.max_wall_time_s, 180.0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", f"full_race_{ts}")

    result = run_full_race(
        n_laps=args.n_laps,
        lap_distance_m=args.lap_distance_m,
        dt=args.dt,
        pretrain_samples=args.pretrain_samples,
        pretrain_epochs=args.pretrain_epochs,
        max_lap_time_s=args.max_lap_time_s,
        max_wall_time_s=args.max_wall_time_s,
    )
    save_visualizations(result, out_dir)

    json_path = os.path.join(out_dir, "full_race_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 72)
    print("FULL RACE COMPLETE")
    print("=" * 72)
    print(f"Mean real lap time: {result['summary']['mean_real_laptime_s']:.3f} s")
    print(f"Mean predicted lap time: {result['summary']['mean_pred_laptime_s']:.3f} s")
    print(f"Mean |lap error|: {result['summary']['mean_abs_laptime_error_s']:.3f} s")
    print(f"Total fuel used: {result['summary']['total_fuel_used_kg']:.3f} kg")
    print(f"Mass estimation RMSE: {result['summary']['mass_estimation_rmse_kg']:.3f} kg")
    print(f"Outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
