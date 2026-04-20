"""
Race capability test suite for IRDAS.

Purpose:
- Stress-test robustness across a wide range of race scenarios
- Evaluate response to sensor changes, tyre degradation + tyre changes,
  and minor model discrepancies between true vehicle and baseline twin
- Produce quantitative metrics and visual evidence (plots + JSON)

Run:
  python race_capability_test.py
  python race_capability_test.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
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

INITIAL_STATE = np.array([0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 30.0, 30.0, 30.0, 30.0, 8000.0, 4.0, 0.5], dtype=np.float64)


@dataclass
class ScenarioConfig:
    name: str
    description: str
    n_laps: int
    steps_per_lap: int
    dt: float
    mismatch_scale: float
    base_noise_scale: float
    shock_noise_lap: int | None
    shock_noise_scale: float
    tyre_change_laps: tuple[int, ...]
    expected_mean_speed_mps: float
    min_pretrain_epochs: int
    min_pretrain_samples: int
    reset_each_lap: bool = True
    metric_warmup_steps: int = 25


@dataclass
class ThresholdConfig:
    min_completion_rate: float = 0.80
    max_rmse_vx: float = 8.0
    max_rmse_vy: float = 5.5
    max_rmse_r: float = 0.8
    max_mass_rmse: float = 2.5
    max_low_speed_frac: float = 0.20
    max_mean_recovery_events: float = 25.0
    min_progress_ratio: float = 0.90


def evaluate_summary(summary: dict, thr: ThresholdConfig) -> dict:
    rmse_vx_eval = summary.get("steady_rmse_vx", summary["mean_rmse_vx"])
    rmse_vy_eval = summary.get("steady_rmse_vy", summary["mean_rmse_vy"])
    rmse_r_eval = summary.get("steady_rmse_r", summary["mean_rmse_r"])
    checks = {
        "completion_rate": summary["completion_rate"] >= thr.min_completion_rate,
        "rmse_vx": rmse_vx_eval <= thr.max_rmse_vx,
        "rmse_vy": rmse_vy_eval <= thr.max_rmse_vy,
        "rmse_r": rmse_r_eval <= thr.max_rmse_r,
        "mass_rmse": summary["mean_mass_rmse"] <= thr.max_mass_rmse,
        "low_speed_frac": summary["mean_low_speed_frac"] <= thr.max_low_speed_frac,
        "recovery_events": summary["mean_recovery_events"] <= thr.max_mean_recovery_events,
        "progress_ratio": summary["mean_progress_ratio"] >= thr.min_progress_ratio,
    }
    passed = [k for k, ok in checks.items() if ok]
    failed = [k for k, ok in checks.items() if not ok]
    return {
        "pass": len(failed) == 0,
        "pass_count": len(passed),
        "check_count": len(checks),
        "pass_ratio": float(len(passed) / max(1, len(checks))),
        "failed_checks": failed,
        "checks": checks,
    }


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


def create_true_params(baseline: dict, mismatch_scale: float) -> dict:
    true_params = baseline.copy()
    true_params["TYRE_LAT"] = baseline["TYRE_LAT"].copy()
    true_params["TYRE_LON"] = baseline["TYRE_LON"].copy()

    # Minor to moderate discrepancy knobs.
    true_params["Cd"] = baseline["Cd"] * (1.0 + 0.04 * mismatch_scale)
    true_params["Cl"] = baseline["Cl"] * (1.0 - 0.04 * mismatch_scale)
    true_params["M"] = baseline["M"] * (1.0 + 0.012 * mismatch_scale)

    true_params["TYRE_LAT"]["a2"] = baseline["TYRE_LAT"]["a2"] * (1.0 - 0.03 * mismatch_scale)
    true_params["TYRE_LON"]["a2"] = baseline["TYRE_LON"]["a2"] * (1.0 - 0.025 * mismatch_scale)
    true_params["TYRE_LAT"]["B"] = baseline["TYRE_LAT"]["B"] * (1.0 + 0.03 * mismatch_scale)
    true_params["TYRE_LON"]["B"] = baseline["TYRE_LON"]["B"] * (1.0 + 0.02 * mismatch_scale)

    return true_params


def set_sensor_noise(irdas: IRDAS, noise_scale: float) -> None:
    irdas.sensor_sim.noise.update(
        {
            "x": 0.6 * noise_scale,
            "y": 0.6 * noise_scale,
            "ax": 0.30 * noise_scale,
            "ay": 0.30 * noise_scale,
            "r": 0.006 * noise_scale,
            "vx": 0.12 * noise_scale,
            "vy": 0.12 * noise_scale,
            "rpm": 30.0 * noise_scale,
            "wheel_speed": 0.07 * noise_scale,
            "fuel_flow": 0.0022 * noise_scale,
            "mass": 2.0 * noise_scale,
        }
    )


def target_speed_profile(phase: float) -> float:
    if phase < 0.30:
        return 62.0
    if phase < 0.44:
        return 32.0
    if phase < 0.72:
        return 40.0
    return 55.0


def control_profile(step: int, steps_per_lap: int, vx_true: float,
                    stuck_timer_s: float, dt: float,
                    rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    phase = step / max(steps_per_lap, 1)
    target_speed = target_speed_profile(phase)

    # Closed-loop longitudinal control to avoid open-loop stall artifacts.
    speed_error = target_speed - vx_true
    ff = 0.12 + 0.010 * target_speed
    throttle = np.clip(ff + 0.032 * speed_error, 0.0, 1.0)
    brake = np.clip(0.055 * max(0.0, -speed_error), 0.0, 0.70)

    if phase < 0.30:  # straight
        steer = rng.normal(0.0, 0.004)
        throttle = np.clip(max(throttle, 0.72) + rng.normal(0, 0.02), 0.55, 1.00)
        brake = min(brake, 0.20)
    elif phase < 0.44:  # braking zone
        steer = np.clip(0.02 * np.sin(2 * np.pi * phase * 5), -0.06, 0.06)
        throttle = np.clip(min(throttle, 0.20) + rng.normal(0, 0.01), 0.00, 0.30)
        brake = np.clip(max(brake, 0.32) + rng.normal(0, 0.03), 0.18, 0.70)
    elif phase < 0.72:  # cornering
        steer = np.clip(0.16 * np.sin(2 * np.pi * (phase - 0.44) * 2.8), -0.24, 0.24)
        throttle = np.clip(throttle + rng.normal(0, 0.03), 0.20, 0.85)
        brake = np.clip(brake, 0.0, 0.40)
    else:  # exit
        steer = np.clip(0.08 * np.sin(2 * np.pi * (phase - 0.72) * 4), -0.12, 0.12)
        throttle = np.clip(max(throttle, 0.64) + rng.normal(0, 0.03), 0.45, 0.98)
        brake = np.clip(brake, 0.0, 0.25)

    # Recovery supervisor for low-speed lockups during stress scenarios.
    recovery_active = False
    if vx_true < 2.0 and target_speed > 16.0:
        stuck_timer_s += dt
    else:
        stuck_timer_s = max(0.0, stuck_timer_s - dt)

    if stuck_timer_s > 0.8:
        recovery_active = True
        throttle = max(throttle, 0.92)
        brake = 0.0
        steer *= 0.25
        stuck_timer_s = max(0.0, stuck_timer_s - 0.3 * dt)

    if brake > 0.05:
        throttle *= 0.25

    return np.array([steer, throttle, brake], dtype=np.float64), stuck_timer_s, float(recovery_active)


def apply_tyre_degradation(true_params: dict, baseline_true: dict, tyre_health: float) -> None:
    lat_a2_base = baseline_true["TYRE_LAT"]["a2"]
    lon_a2_base = baseline_true["TYRE_LON"]["a2"]
    lat_b_base = baseline_true["TYRE_LAT"]["B"]
    lon_b_base = baseline_true["TYRE_LON"]["B"]

    a2_scale = 0.78 + 0.22 * tyre_health
    b_scale = 1.00 + 0.12 * (1.0 - tyre_health)

    true_params["TYRE_LAT"]["a2"] = float(lat_a2_base * a2_scale)
    true_params["TYRE_LON"]["a2"] = float(lon_a2_base * a2_scale)
    true_params["TYRE_LAT"]["B"] = float(lat_b_base * b_scale)
    true_params["TYRE_LON"]["B"] = float(lon_b_base * b_scale)


def run_scenario(cfg: ScenarioConfig, pretrain_samples: int, pretrain_epochs: int,
                 thresholds: ThresholdConfig, allow_early_stop: bool,
                 seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    baseline = create_baseline_params()
    true_params = create_true_params(baseline, cfg.mismatch_scale)
    baseline_true_tyre = {
        "TYRE_LAT": true_params["TYRE_LAT"].copy(),
        "TYRE_LON": true_params["TYRE_LON"].copy(),
    }

    irdas = IRDAS(baseline, device="cpu", use_nn=True, use_rls=True)
    irdas.dt = cfg.dt
    irdas.initialize_real_vehicle(true_params=true_params, seed=seed)
    set_sensor_noise(irdas, cfg.base_noise_scale)

    eff_samples = max(pretrain_samples, cfg.min_pretrain_samples)
    eff_epochs = max(pretrain_epochs, cfg.min_pretrain_epochs)

    print(
        f"Pretraining schedule -> samples={eff_samples}, epochs={eff_epochs} "
        f"(requested: {pretrain_samples}, {pretrain_epochs})"
    )
    if allow_early_stop:
        patience = 20
        min_epochs = max(0, int(0.5 * eff_epochs))
    else:
        # Force near-full training budget unless user explicitly allows early stop.
        patience = max(eff_epochs + 5, 40)
        min_epochs = eff_epochs

    irdas.pretrain_neural_network(
        n_training_samples=eff_samples,
        epochs=eff_epochs,
        batch_size=64,
        early_stopping_patience=patience,
        min_epochs_before_stopping=min_epochs,
    )

    tyre_health = 1.0
    lap_time_s = cfg.steps_per_lap * cfg.dt
    expected_lap_distance_m = max(1.0, cfg.expected_mean_speed_mps * lap_time_s)

    metrics = {
        "lap": [],
        "noise_scale": [],
        "tyre_health": [],
        "avg_vx_true": [],
        "p05_vx_true": [],
        "rmse_vx": [],
        "rmse_vy": [],
        "rmse_r": [],
        "mass_rmse": [],
        "model_error_rms": [],
        "lat_a2_true": [],
        "lat_a2_est": [],
        "lon_a2_true": [],
        "lon_a2_est": [],
        "lap_progress_m": [],
        "lap_progress_ratio": [],
        "lap_complete": [],
        "low_speed_frac": [],
        "mean_throttle": [],
        "mean_brake": [],
        "recovery_events": [],
        "event": [],
    }

    for lap in range(1, cfg.n_laps + 1):
        event = ""

        # Isolate per-lap capability under each scenario condition.
        if cfg.reset_each_lap:
            irdas.true_state = INITIAL_STATE.copy()
            irdas.kalman_filter.reset(INITIAL_STATE.copy())
            if hasattr(irdas, "nn_learner") and irdas.nn_trained:
                irdas.nn_learner.reset_stateful_inference()

        if cfg.shock_noise_lap is not None and lap == cfg.shock_noise_lap:
            set_sensor_noise(irdas, cfg.shock_noise_scale)
            event = "sensor_shock"
        elif cfg.shock_noise_lap is not None and lap == (cfg.shock_noise_lap + 1):
            set_sensor_noise(irdas, cfg.base_noise_scale)
            event = "sensor_recover"

        if lap in cfg.tyre_change_laps:
            tyre_health = 1.0
            apply_tyre_degradation(irdas.real_simulator.true_params, baseline_true_tyre, tyre_health)
            irdas.on_tire_change(reset_parameter_adapter=True)
            event = "tyre_change" if not event else f"{event}+tyre_change"

        lap_vx_true = []
        lap_vx_est = []
        lap_vy_true = []
        lap_vy_est = []
        lap_r_true = []
        lap_r_est = []
        lap_mass_true = []
        lap_mass_est = []
        lap_model_err = []
        lap_progress = 0.0
        lap_throttle = []
        lap_brake = []
        low_speed_count = 0
        stuck_timer_s = 0.0
        recovery_events = 0

        for s in range(cfg.steps_per_lap):
            vx_now = max(0.0, float(irdas.true_state[3]))
            u, stuck_timer_s, recovery_flag = control_profile(
                s, cfg.steps_per_lap, vx_now, stuck_timer_s, cfg.dt, rng
            )
            if recovery_flag > 0.5:
                recovery_events += 1
            est = irdas.step(
                u,
                use_nn_correction=True,
                use_param_adaptation=True,
                reset_nn_memory=False,
            )

            true_s = irdas.true_state
            vxt = max(0.0, float(true_s[3]))
            lap_progress += vxt * cfg.dt

            lap_vx_true.append(vxt)
            lap_vx_est.append(max(0.0, float(est[3])))
            lap_vy_true.append(float(true_s[4]))
            lap_vy_est.append(float(est[4]))
            lap_r_true.append(float(true_s[5]))
            lap_r_est.append(float(est[5]))
            lap_mass_true.append(float(irdas.true_vehicle_mass))
            lap_mass_est.append(float(irdas.kalman_filter.get_mass_estimate()))
            lap_throttle.append(float(u[1]))
            lap_brake.append(float(u[2]))
            if vxt < 2.0:
                low_speed_count += 1
            if irdas.history["model_errors"]:
                lap_model_err.append(float(irdas.history["model_errors"][-1]))

            deg_rate = 2.6e-4 + 3.3e-4 * abs(float(u[0])) + 2.0e-4 * float(u[2])
            tyre_health = float(max(0.70, tyre_health - deg_rate * cfg.dt))
            apply_tyre_degradation(irdas.real_simulator.true_params, baseline_true_tyre, tyre_health)

        cur = irdas.current_params
        st = min(max(0, int(cfg.metric_warmup_steps)), max(0, cfg.steps_per_lap - 1))
        vx_true_eval = lap_vx_true[st:] if len(lap_vx_true) > st else lap_vx_true
        vx_est_eval = lap_vx_est[st:] if len(lap_vx_est) > st else lap_vx_est
        vy_true_eval = lap_vy_true[st:] if len(lap_vy_true) > st else lap_vy_true
        vy_est_eval = lap_vy_est[st:] if len(lap_vy_est) > st else lap_vy_est
        r_true_eval = lap_r_true[st:] if len(lap_r_true) > st else lap_r_true
        r_est_eval = lap_r_est[st:] if len(lap_r_est) > st else lap_r_est
        mass_true_eval = lap_mass_true[st:] if len(lap_mass_true) > st else lap_mass_true
        mass_est_eval = lap_mass_est[st:] if len(lap_mass_est) > st else lap_mass_est

        metrics["lap"].append(lap)
        metrics["noise_scale"].append(cfg.base_noise_scale if not (cfg.shock_noise_lap == lap) else cfg.shock_noise_scale)
        metrics["tyre_health"].append(float(tyre_health))
        metrics["avg_vx_true"].append(float(np.mean(vx_true_eval)))
        metrics["p05_vx_true"].append(float(np.percentile(vx_true_eval, 5)))
        metrics["rmse_vx"].append(float(np.sqrt(np.mean((np.array(vx_true_eval) - np.array(vx_est_eval)) ** 2))))
        metrics["rmse_vy"].append(float(np.sqrt(np.mean((np.array(vy_true_eval) - np.array(vy_est_eval)) ** 2))))
        metrics["rmse_r"].append(float(np.sqrt(np.mean((np.array(r_true_eval) - np.array(r_est_eval)) ** 2))))
        metrics["mass_rmse"].append(float(np.sqrt(np.mean((np.array(mass_true_eval) - np.array(mass_est_eval)) ** 2))))
        metrics["model_error_rms"].append(float(np.sqrt(np.mean(np.square(lap_model_err))) if lap_model_err else 0.0))
        metrics["lat_a2_true"].append(float(irdas.real_simulator.true_params["TYRE_LAT"]["a2"]))
        metrics["lat_a2_est"].append(float(cur["TYRE_LAT"]["a2"]))
        metrics["lon_a2_true"].append(float(irdas.real_simulator.true_params["TYRE_LON"]["a2"]))
        metrics["lon_a2_est"].append(float(cur["TYRE_LON"]["a2"]))
        metrics["lap_progress_m"].append(float(lap_progress))
        progress_ratio = float(lap_progress / expected_lap_distance_m)
        metrics["lap_progress_ratio"].append(progress_ratio)
        metrics["lap_complete"].append(bool(progress_ratio >= 0.90))
        metrics["low_speed_frac"].append(float(low_speed_count / max(1, cfg.steps_per_lap)))
        metrics["mean_throttle"].append(float(np.mean(lap_throttle)) if lap_throttle else 0.0)
        metrics["mean_brake"].append(float(np.mean(lap_brake)) if lap_brake else 0.0)
        metrics["recovery_events"].append(int(recovery_events))
        metrics["event"].append(event)

    completion_rate = float(np.mean(np.array(metrics["lap_complete"], dtype=float)))

    # Event transitions: mark event lap and immediate next lap as transition regime.
    events = metrics["event"]
    transition_mask = [False] * len(events)
    for i, ev in enumerate(events):
        if ev:
            transition_mask[i] = True
            if i + 1 < len(events):
                transition_mask[i + 1] = True

    steady_idx = [i for i, is_trans in enumerate(transition_mask) if not is_trans]
    trans_idx = [i for i, is_trans in enumerate(transition_mask) if is_trans]

    def _subset_mean(arr: list[float], idxs: list[int], fallback: float) -> float:
        if not idxs:
            return fallback
        vals = [arr[i] for i in idxs]
        return float(np.mean(vals)) if vals else fallback

    summary = {
        "scenario": cfg.name,
        "description": cfg.description,
        "completion_rate": completion_rate,
        "mean_avg_vx_true": float(np.mean(metrics["avg_vx_true"])),
        "mean_p05_vx_true": float(np.mean(metrics["p05_vx_true"])),
        "mean_rmse_vx": float(np.mean(metrics["rmse_vx"])),
        "mean_rmse_vy": float(np.mean(metrics["rmse_vy"])),
        "mean_rmse_r": float(np.mean(metrics["rmse_r"])),
        "mean_mass_rmse": float(np.mean(metrics["mass_rmse"])),
        "mean_model_error_rms": float(np.mean(metrics["model_error_rms"])),
        "mean_progress_ratio": float(np.mean(metrics["lap_progress_ratio"])),
        "mean_low_speed_frac": float(np.mean(metrics["low_speed_frac"])),
        "mean_throttle": float(np.mean(metrics["mean_throttle"])),
        "mean_brake": float(np.mean(metrics["mean_brake"])),
        "mean_recovery_events": float(np.mean(metrics["recovery_events"])),
        "mean_abs_lat_a2_err": float(np.mean(np.abs(np.array(metrics["lat_a2_true"]) - np.array(metrics["lat_a2_est"])))),
        "mean_abs_lon_a2_err": float(np.mean(np.abs(np.array(metrics["lon_a2_true"]) - np.array(metrics["lon_a2_est"])))),
        "effective_pretrain_samples": int(eff_samples),
        "effective_pretrain_epochs": int(eff_epochs),
        "transition_lap_count": int(len(trans_idx)),
        "steady_lap_count": int(len(steady_idx)),
        "steady_rmse_vx": _subset_mean(metrics["rmse_vx"], steady_idx, float(np.mean(metrics["rmse_vx"]))),
        "steady_rmse_vy": _subset_mean(metrics["rmse_vy"], steady_idx, float(np.mean(metrics["rmse_vy"]))),
        "steady_rmse_r": _subset_mean(metrics["rmse_r"], steady_idx, float(np.mean(metrics["rmse_r"]))),
        "transition_rmse_vx": _subset_mean(metrics["rmse_vx"], trans_idx, float(np.mean(metrics["rmse_vx"]))),
        "transition_rmse_vy": _subset_mean(metrics["rmse_vy"], trans_idx, float(np.mean(metrics["rmse_vy"]))),
        "transition_rmse_r": _subset_mean(metrics["rmse_r"], trans_idx, float(np.mean(metrics["rmse_r"]))),
    }

    verdict = evaluate_summary(summary, thresholds)
    summary["verdict"] = verdict

    return {"config": cfg.__dict__, "summary": summary, "metrics": metrics}


def plot_scenario(result: dict, out_dir: str) -> None:
    name = result["summary"]["scenario"]
    m = result["metrics"]
    laps = np.array(m["lap"])

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

    axes[0, 0].plot(laps, m["avg_vx_true"], marker="o", label="avg vx true")
    axes[0, 0].plot(laps, m["p05_vx_true"], marker="s", label="p05 vx true")
    axes[0, 0].set_title("Speed Capability")
    axes[0, 0].set_ylabel("m/s")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(laps, m["rmse_vx"], label="rmse vx")
    axes[0, 1].plot(laps, m["rmse_vy"], label="rmse vy")
    axes[0, 1].plot(laps, m["rmse_r"], label="rmse r")
    axes[0, 1].set_title("State Estimation Error")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(laps, m["lat_a2_true"], "--", label="lat a2 true")
    axes[1, 0].plot(laps, m["lat_a2_est"], label="lat a2 est")
    axes[1, 0].plot(laps, m["lon_a2_true"], "--", label="lon a2 true")
    axes[1, 0].plot(laps, m["lon_a2_est"], label="lon a2 est")
    axes[1, 0].set_title("RLS Parameter Tracking")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(laps, m["tyre_health"], marker="o", label="tyre health")
    axes[1, 1].plot(laps, m["noise_scale"], marker="s", label="noise scale")
    axes[1, 1].plot(laps, m["low_speed_frac"], marker="^", label="low-speed frac")
    axes[1, 1].set_title("Tyre and Sensor Regime")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    axes[2, 0].plot(laps, m["model_error_rms"], marker="o", color="tab:red")
    axes[2, 0].plot(laps, m["lap_progress_ratio"], marker="s", color="tab:blue", label="progress ratio")
    axes[2, 0].set_title("Model Error RMS")
    axes[2, 0].set_xlabel("Lap")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()

    comp = np.array(m["lap_complete"], dtype=float)
    axes[2, 1].bar(laps, comp, color="tab:green")
    axes[2, 1].set_ylim(-0.05, 1.05)
    axes[2, 1].set_title("Lap Completion Flag")
    axes[2, 1].set_xlabel("Lap")
    axes[2, 1].grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Race Capability: {name}")
    fig.savefig(os.path.join(out_dir, f"{name}_dashboard.png"), dpi=140)
    plt.close(fig)


def plot_aggregate(results: list[dict], out_dir: str) -> None:
    names = [r["summary"]["scenario"] for r in results]
    completion = [r["summary"]["completion_rate"] for r in results]
    rmse_vx = [r["summary"]["mean_rmse_vx"] for r in results]
    model_err = [r["summary"]["mean_model_error_rms"] for r in results]
    pass_ratio = [r["summary"]["verdict"]["pass_ratio"] for r in results]

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(x - w, completion, width=w, label="completion rate")
    ax.bar(x, rmse_vx, width=w, label="mean rmse vx")
    ax.bar(x + w, model_err, width=w, label="mean model err")
    ax.plot(x, pass_ratio, color="tab:purple", marker="o", linewidth=2, label="verdict pass ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_title("Scenario Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(out_dir, "scenario_comparison.png"), dpi=140)
    plt.close(fig)


def build_scenarios(quick: bool) -> list[ScenarioConfig]:
    n_laps = 5 if quick else 10
    steps = 200 if quick else 350
    dt = 0.05

    return [
        ScenarioConfig(
            name="baseline_robust",
            description="Nominal noise with mild mismatch and one tyre change",
            n_laps=n_laps,
            steps_per_lap=steps,
            dt=dt,
            mismatch_scale=1.0,
            base_noise_scale=1.0,
            shock_noise_lap=None,
            shock_noise_scale=1.0,
            tyre_change_laps=(max(2, n_laps // 2),),
            expected_mean_speed_mps=38.0,
            min_pretrain_epochs=24 if quick else 45,
            min_pretrain_samples=260 if quick else 850,
        ),
        ScenarioConfig(
            name="sensor_shock",
            description="Temporary high sensor noise shock, then recovery",
            n_laps=n_laps,
            steps_per_lap=steps,
            dt=dt,
            mismatch_scale=1.0,
            base_noise_scale=1.0,
            shock_noise_lap=max(2, n_laps // 2),
            shock_noise_scale=2.8,
            tyre_change_laps=(max(2, n_laps // 2 + 1),),
            expected_mean_speed_mps=36.0,
            min_pretrain_epochs=28 if quick else 55,
            min_pretrain_samples=300 if quick else 950,
        ),
        ScenarioConfig(
            name="high_mismatch_deg",
            description="Higher model mismatch, faster tyre degradation, two tyre changes",
            n_laps=n_laps,
            steps_per_lap=steps,
            dt=dt,
            mismatch_scale=1.8,
            base_noise_scale=1.2,
            shock_noise_lap=max(3, n_laps - 2),
            shock_noise_scale=2.2,
            tyre_change_laps=(max(2, n_laps // 3), max(3, 2 * n_laps // 3)),
            expected_mean_speed_mps=34.0,
            min_pretrain_epochs=32 if quick else 65,
            min_pretrain_samples=340 if quick else 1100,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Race capability test suite")
    parser.add_argument("--quick", action="store_true", help="Run shorter scenario sweep")
    parser.add_argument("--pretrain-samples", type=int, default=900)
    parser.add_argument("--pretrain-epochs", type=int, default=60)
    parser.add_argument("--n-laps", type=int, default=None, help="Optional override for laps per scenario")
    parser.add_argument("--steps-per-lap", type=int, default=None, help="Optional override for steps per lap")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Optional subset of scenario names to run (e.g., baseline_robust sensor_shock)",
    )
    parser.add_argument(
        "--allow-early-stop",
        action="store_true",
        help="Allow NN pretraining to stop early on validation plateau",
    )
    args = parser.parse_args()
    thresholds = ThresholdConfig()

    if args.quick:
        args.pretrain_samples = max(args.pretrain_samples, 260)
        args.pretrain_epochs = max(args.pretrain_epochs, 24)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", f"race_capability_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    scenarios = build_scenarios(args.quick)
    if args.n_laps is not None:
        for s in scenarios:
            s.n_laps = max(1, int(args.n_laps))
    if args.steps_per_lap is not None:
        for s in scenarios:
            s.steps_per_lap = max(10, int(args.steps_per_lap))
    if args.scenarios:
        wanted = set(args.scenarios)
        scenarios = [s for s in scenarios if s.name in wanted]
        if not scenarios:
            raise ValueError("No matching scenarios selected. Available: baseline_robust, sensor_shock, high_mismatch_deg")
    all_results = []

    for cfg in scenarios:
        print("\n" + "=" * 72)
        print(f"Running scenario: {cfg.name}")
        print(cfg.description)
        print("=" * 72)
        result = run_scenario(
            cfg,
            args.pretrain_samples,
            args.pretrain_epochs,
            thresholds,
            allow_early_stop=args.allow_early_stop,
        )
        all_results.append(result)
        plot_scenario(result, out_dir)
        print("Summary:", json.dumps(result["summary"], indent=2))

    plot_aggregate(all_results, out_dir)

    bundle = {
        "suite": "race_capability_test",
        "timestamp": ts,
        "quick": args.quick,
        "results": all_results,
    }
    with open(os.path.join(out_dir, "race_capability_results.json"), "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    print("\nOutputs saved to:", out_dir)


if __name__ == "__main__":
    main()
