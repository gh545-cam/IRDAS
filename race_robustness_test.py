"""
race_robustness_test.py — IRDAS Comprehensive Robustness Test Suite

Tests the system's capability to handle:
  1. Sensor Noise       — varying IMU/GPS noise levels
  2. Model Deviation    — parameter mismatches between baseline and real vehicle
  3. In-race Changes    — sudden mid-race events (tyre swap, fuel load, aero damage)

Structure:
  - Three test scenarios run sequentially
  - Live plots update after each lap
  - Final dashboard with full performance metrics

Run from IRDAS directory:
    python race_robustness_test.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time
import pickle

from params import (L, TF, TR, H, MX, M, Cd, Cl, Area, CX, K,
                    final_drive, tyre_radius, TYRE_LAT, TYRE_LON,
                    GEAR_RATIOS, UPSHIFT_SPEED_KPH, ENGINE_RPM, ENGINE_TORQUE_NM)
from twin_track import twin_track_model
from irdas_main import IRDAS

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

N_LAPS        = 20
STEPS_PER_LAP = 400        # 400 × 0.05s = 20s per lap
DT            = 0.05
RESULTS_DIR   = Path('results/robustness')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_STATE = np.array([0., 0., 0., 30., 0., 0.,
                           30., 30., 30., 30., 8000., 4., 0.5])

# ─────────────────────────────────────────────────────────────────────────────
# Baseline params builder
# ─────────────────────────────────────────────────────────────────────────────

def make_baseline_params():
    return {
        'L': L, 'TF': TF, 'TR': TR, 'H': H, 'MX': MX, 'M': M,
        'Cd': Cd, 'Cl': Cl, 'Area': Area, 'CX': CX,
        'K': K, 'final_drive': final_drive, 'tyre_radius': tyre_radius,
        'TYRE_LAT': TYRE_LAT.copy(), 'TYRE_LON': TYRE_LON.copy(),
        'GEAR_RATIOS': GEAR_RATIOS.copy(),
        'UPSHIFT_SPEED_KPH': UPSHIFT_SPEED_KPH.copy(),
        'ENGINE_RPM': ENGINE_RPM.copy(),
        'ENGINE_TORQUE_NM': ENGINE_TORQUE_NM.copy()
    }

# ─────────────────────────────────────────────────────────────────────────────
# Control generation
# ─────────────────────────────────────────────────────────────────────────────

def make_lap_controls(steps_per_lap, n_laps, seed=42):
    """Generate realistic race controls with cornering."""
    rng = np.random.default_rng(seed)
    total = steps_per_lap * n_laps
    controls = np.zeros((total, 3))
    throttle = 0.85
    for i in range(total):
        lap_phase    = (i % steps_per_lap) / steps_per_lap
        corner_steer = 0.12 * np.sin(2 * np.pi * lap_phase * 3)
        steer        = np.clip(corner_steer + rng.normal(0, 0.01), -0.20, 0.20)
        throttle     = np.clip(throttle + rng.normal(0, 0.02), 0.75, 1.0)
        controls[i]  = [steer, throttle, 0.0]
    return controls

# ─────────────────────────────────────────────────────────────────────────────
# Tyre degradation (shared across scenarios)
# ─────────────────────────────────────────────────────────────────────────────

def degrade_tyre_params(baseline_params, lap, n_laps, wear_rate=0.007):
    grip = max(1.0 - wear_rate * lap, 0.75)
    degraded = baseline_params.copy()
    degraded['TYRE_LAT'] = baseline_params['TYRE_LAT'].copy()
    degraded['TYRE_LON'] = baseline_params['TYRE_LON'].copy()
    degraded['TYRE_LAT']['a2'] = baseline_params['TYRE_LAT']['a2'] * grip
    degraded['TYRE_LON']['a2'] = baseline_params['TYRE_LON']['a2'] * grip
    degraded['TYRE_LAT']['B']  = baseline_params['TYRE_LAT']['B'] * (1.0 + 0.1 * (1.0 - grip))
    degraded['TYRE_LON']['B']  = baseline_params['TYRE_LON']['B'] * (1.0 + 0.1 * (1.0 - grip))
    return degraded, grip

# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LapResult:
    lap:             int
    true_grip:       float
    rls_a2:          float
    nn_residual:     float
    vx_mean:         float
    ekf_rmse_vx:     float
    ekf_rmse_vy:     float
    ekf_rmse_r:      float
    model_error_rms: float
    event:           str = ''

@dataclass
class ScenarioResult:
    name:        str
    description: str
    laps:        List[LapResult] = field(default_factory=list)

    def metric_array(self, attr):
        return np.array([getattr(l, attr) for l in self.laps])

# ─────────────────────────────────────────────────────────────────────────────
# Core lap runner
# ─────────────────────────────────────────────────────────────────────────────

def run_lap(irdas, controls, lap_idx, step_offset,
            true_params_override=None, event_label=''):
    """
    Run one lap. Returns a LapResult.
    true_params_override: if set, overrides real simulator params this lap.
    """
    lap_vx, lap_ekf_vx, lap_ekf_vy, lap_ekf_r = [], [], [], []
    lap_nn_res, lap_model_err = [], []

    reset_state = INITIAL_STATE.copy()
    irdas.true_state = reset_state.copy()
    irdas.kalman_filter.reset(reset_state)

    if true_params_override:
        irdas.real_simulator.true_params.update(true_params_override)

    for s in range(STEPS_PER_LAP):
        u = controls[step_offset + s]
        try:
            est = irdas.step(u, use_nn_correction=True, use_param_adaptation=True)
        except Exception as e:
            continue

        true_s = irdas.true_state

        lap_vx.append(est[3])
        lap_ekf_vx.append(abs(est[3] - true_s[3]))
        lap_ekf_vy.append(abs(est[4] - true_s[4]))
        lap_ekf_r.append(abs(est[5]  - true_s[5]))

        if irdas.nn_trained:
            try:
                from residual_network import extract_dynamics_states
                dyn = extract_dynamics_states(est)
                res = irdas.nn_learner.predict(dyn, u)
                lap_nn_res.append(np.linalg.norm(res))
            except Exception:
                pass

        if irdas.history['model_errors']:
            lap_model_err.append(irdas.history['model_errors'][-1])

    current = irdas.param_adapter.get_current_params()
    rls_a2  = current['TYRE_LAT']['a2']

    return LapResult(
        lap             = lap_idx,
        true_grip       = 1.0,          # filled by caller
        rls_a2          = rls_a2,
        nn_residual     = float(np.mean(lap_nn_res)) if lap_nn_res else 0.0,
        vx_mean         = float(np.mean(lap_vx)) if lap_vx else 0.0,
        ekf_rmse_vx     = float(np.sqrt(np.mean(np.array(lap_ekf_vx)**2))) if lap_ekf_vx else 0.0,
        ekf_rmse_vy     = float(np.sqrt(np.mean(np.array(lap_ekf_vy)**2))) if lap_ekf_vy else 0.0,
        ekf_rmse_r      = float(np.sqrt(np.mean(np.array(lap_ekf_r)**2)))  if lap_ekf_r  else 0.0,
        model_error_rms = float(np.sqrt(np.mean(np.array(lap_model_err)**2))) if lap_model_err else 0.0,
        event           = event_label
    )

# ─────────────────────────────────────────────────────────────────────────────
# IRDAS factory
# ─────────────────────────────────────────────────────────────────────────────

def build_irdas(baseline_params, noise_multiplier=1.0, model_seed=42,
                pretrain_samples=3000, pretrain_epochs=150):
    """Build, initialise, and pretrain an IRDAS instance."""
    irdas = IRDAS(baseline_params, device='cpu', use_nn=True, use_rls=True)

    # Scale sensor noise via SensorSimulator if it supports it
    if hasattr(irdas.sensor_sim, 'noise_scale'):
        irdas.sensor_sim.noise_scale = noise_multiplier

    irdas.initialize_real_vehicle(seed=model_seed)

    print(f"  Pretraining NN ({pretrain_samples} samples, {pretrain_epochs} epochs)...")
    irdas.pretrain_neural_network(
        n_training_samples=pretrain_samples,
        epochs=pretrain_epochs,
        batch_size=64
    )

    # RLS warmup
    print("  Warming up RLS (200 steps)...")
    irdas.true_state = INITIAL_STATE.copy()
    for i in range(200):
        u = np.array([0.08 * np.sin(i / 20.0), 0.85, 0.0])
        irdas.step(u, use_nn_correction=False, use_param_adaptation=True)
    irdas.history = {k: [] for k in irdas.history}
    irdas.time_step = 0.0

    irdas.reset(INITIAL_STATE)
    return irdas

# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 1: Sensor Noise Sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario_1(baseline_params):
    """
    Test 1: How does state estimation quality degrade with sensor noise?
    Run 3 sub-scenarios: low / nominal / high noise.
    Returns a list of ScenarioResult objects.
    """
    print("\n" + "="*60)
    print("SCENARIO 1: Sensor Noise Robustness")
    print("="*60)

    noise_configs = [
        ('Low Noise',     0.3),
        ('Nominal Noise', 1.0),
        ('High Noise',    3.0),
    ]

    results = []
    controls = make_lap_controls(STEPS_PER_LAP, N_LAPS)

    for label, noise_mult in noise_configs:
        print(f"\n  [{label}] noise_multiplier={noise_mult:.1f}")
        sr = ScenarioResult(
            name        = f'S1_{label.replace(" ", "_")}',
            description = f'Sensor noise × {noise_mult}'
        )

        irdas = build_irdas(baseline_params, noise_multiplier=noise_mult)

        step_offset = 0
        for lap in range(N_LAPS):
            degraded, grip = degrade_tyre_params(baseline_params, lap, N_LAPS)
            irdas.real_simulator.true_params.update({
                'TYRE_LAT': degraded['TYRE_LAT'],
                'TYRE_LON': degraded['TYRE_LON'],
            })

            lr = run_lap(irdas, controls, lap, step_offset)
            lr.true_grip = grip
            sr.laps.append(lr)
            step_offset += STEPS_PER_LAP

            print(f"    Lap {lap+1:>2}/{N_LAPS} | grip={grip:.3f} | "
                  f"vx={lr.vx_mean:.2f} | EKF_vx={lr.ekf_rmse_vx:.4f} | "
                  f"EKF_r={lr.ekf_rmse_r:.4f}")

        results.append(sr)
        plot_live(results, 'scenario1', 'Scenario 1: Sensor Noise')

    return results

# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 2: Model Deviation Sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario_2(baseline_params):
    """
    Test 2: How well does the system adapt to various levels of model mismatch?
    Seeds 10, 42, 99 give different mismatch magnitudes.
    """
    print("\n" + "="*60)
    print("SCENARIO 2: Model Deviation Robustness")
    print("="*60)

    mismatch_configs = [
        ('Small Mismatch',  10),
        ('Medium Mismatch', 42),
        ('Large Mismatch',  99),
    ]

    results = []
    controls = make_lap_controls(STEPS_PER_LAP, N_LAPS, seed=7)

    for label, seed in mismatch_configs:
        print(f"\n  [{label}] seed={seed}")
        sr = ScenarioResult(
            name        = f'S2_{label.replace(" ", "_")}',
            description = label
        )

        irdas = build_irdas(baseline_params, model_seed=seed)

        # Print actual mismatch for reference
        diffs = irdas.real_simulator.get_parameter_difference()
        total_mismatch = sum(abs(v) for v in diffs.values())
        print(f"    Total parameter mismatch: {total_mismatch:.4f}")

        step_offset = 0
        for lap in range(N_LAPS):
            degraded, grip = degrade_tyre_params(baseline_params, lap, N_LAPS)
            irdas.real_simulator.true_params.update({
                'TYRE_LAT': degraded['TYRE_LAT'],
                'TYRE_LON': degraded['TYRE_LON'],
            })

            lr = run_lap(irdas, controls, lap, step_offset)
            lr.true_grip = grip
            sr.laps.append(lr)
            step_offset += STEPS_PER_LAP

            print(f"    Lap {lap+1:>2}/{N_LAPS} | grip={grip:.3f} | "
                  f"model_err={lr.model_error_rms:.4f} | "
                  f"rls_a2={lr.rls_a2:.3f} | nn_res={lr.nn_residual:.3f}")

        results.append(sr)
        plot_live(results, 'scenario2', 'Scenario 2: Model Deviation')

    return results

# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 3: In-Race Events
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario_3(baseline_params):
    """
    Test 3: Sudden mid-race parameter changes.
    Events injected at specific laps:
      - Lap 5:  Tyre compound change (fresh tyres — grip reset)
      - Lap 10: Fuel burn (mass reduction -40kg)
      - Lap 15: Aero damage (Cd +25%, Cl -20%)
      - Lap 18: Second tyre degradation episode
    """
    print("\n" + "="*60)
    print("SCENARIO 3: In-Race Events")
    print("="*60)

    sr = ScenarioResult(
        name        = 'S3_InRace_Events',
        description = 'Tyre swap + fuel burn + aero damage'
    )

    irdas  = build_irdas(baseline_params)
    controls = make_lap_controls(STEPS_PER_LAP, N_LAPS, seed=13)

    # Mutable true params — we'll modify these per lap
    current_true = baseline_params.copy()
    current_true['TYRE_LAT'] = baseline_params['TYRE_LAT'].copy()
    current_true['TYRE_LON'] = baseline_params['TYRE_LON'].copy()
    current_true['M']  = baseline_params['M']
    current_true['Cd'] = baseline_params['Cd']
    current_true['Cl'] = baseline_params['Cl']

    # Track which lap each event fires
    events = {
        5:  'Tyre Swap',
        10: 'Fuel Burn (-40kg)',
        15: 'Aero Damage',
        18: 'Tyre Degradation Resumes',
    }

    wear_lap_offset = 0   # reset when tyres are swapped

    step_offset = 0
    for lap in range(N_LAPS):
        event_label = ''

        # ── Event: Tyre swap at lap 5 ────────────────────────────────────────
        if lap == 5:
            event_label = events[5]
            print(f"\n  *** LAP {lap+1}: {event_label} — fresh tyres ***")
            wear_lap_offset = lap   # reset wear counter
            current_true['TYRE_LAT'] = baseline_params['TYRE_LAT'].copy()
            current_true['TYRE_LON'] = baseline_params['TYRE_LON'].copy()

        # ── Event: Fuel burn at lap 10 ───────────────────────────────────────
        if lap == 10:
            event_label = events[10]
            print(f"\n  *** LAP {lap+1}: {event_label} ***")
            current_true['M'] = baseline_params['M'] - 40.0

        # ── Event: Aero damage at lap 15 ─────────────────────────────────────
        if lap == 15:
            event_label = events[15]
            print(f"\n  *** LAP {lap+1}: {event_label} — Cd+25%, Cl-20% ***")
            current_true['Cd'] = baseline_params['Cd'] * 1.25
            current_true['Cl'] = baseline_params['Cl'] * 0.80

        # ── Continuous tyre degradation (post-swap wear resets) ───────────────
        effective_lap = lap - wear_lap_offset
        degraded, grip = degrade_tyre_params(
            baseline_params if lap < 5 else current_true,
            effective_lap, N_LAPS
        )
        current_true['TYRE_LAT'] = degraded['TYRE_LAT']
        current_true['TYRE_LON'] = degraded['TYRE_LON']

        # Apply all current true params to simulator
        irdas.real_simulator.true_params.update({
            'TYRE_LAT': current_true['TYRE_LAT'],
            'TYRE_LON': current_true['TYRE_LON'],
            'M':        current_true['M'],
            'Cd':       current_true['Cd'],
            'Cl':       current_true['Cl'],
        })

        lr = run_lap(irdas, controls, lap, step_offset, event_label=event_label)
        lr.true_grip = grip
        sr.laps.append(lr)
        step_offset += STEPS_PER_LAP

        print(f"    Lap {lap+1:>2}/{N_LAPS} | grip={grip:.3f} | M={current_true['M']:.0f}kg | "
              f"Cd={current_true['Cd']:.3f} | Cl={current_true['Cl']:.3f} | "
              f"rls_a2={lr.rls_a2:.3f} | {event_label}")

    plot_live([sr], 'scenario3', 'Scenario 3: In-Race Events')
    return [sr]

# ─────────────────────────────────────────────────────────────────────────────
# Live plot (called after each scenario or lap batch)
# ─────────────────────────────────────────────────────────────────────────────

COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']

def plot_live(scenario_results, filename_stem, title):
    """Plot current state of one or more scenarios and save."""
    n_scenarios = len(scenario_results)
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax_grip    = fig.add_subplot(gs[0, 0])
    ax_rls     = fig.add_subplot(gs[0, 1])
    ax_ekf_vx  = fig.add_subplot(gs[0, 2])
    ax_nn      = fig.add_subplot(gs[1, 0])
    ax_model   = fig.add_subplot(gs[1, 1])
    ax_vx      = fig.add_subplot(gs[1, 2])

    for idx, sr in enumerate(scenario_results):
        if not sr.laps:
            continue
        c    = COLORS[idx % len(COLORS)]
        laps = [l.lap + 1 for l in sr.laps]
        lbl  = sr.description

        ax_grip.plot(laps, [l.true_grip for l in sr.laps], '-o',
                     color=c, lw=2, ms=4, label=f'{lbl} (true)')
        ax_grip.plot(laps, [l.rls_a2 / sr.laps[0].rls_a2 for l in sr.laps],
                     '--s', color=c, lw=1.5, ms=3, alpha=0.7, label=f'{lbl} (RLS)')

        ax_rls.plot(laps, [l.rls_a2 for l in sr.laps], '-o',
                    color=c, lw=2, ms=4, label=lbl)

        ax_ekf_vx.plot(laps, [l.ekf_rmse_vx for l in sr.laps], '-o',
                       color=c, lw=2, ms=4, label=lbl)

        ax_nn.plot(laps, [l.nn_residual for l in sr.laps], '-^',
                   color=c, lw=2, ms=4, label=lbl)

        ax_model.plot(laps, [l.model_error_rms for l in sr.laps], '-o',
                      color=c, lw=2, ms=4, label=lbl)

        ax_vx.plot(laps, [l.vx_mean for l in sr.laps], '-D',
                   color=c, lw=2, ms=4, label=lbl)

        # Mark events
        for l in sr.laps:
            if l.event:
                for ax in [ax_grip, ax_rls, ax_ekf_vx, ax_nn, ax_model, ax_vx]:
                    ax.axvline(l.lap + 1, color=c, lw=1, ls=':', alpha=0.6)

    for ax, ylabel, title_str in [
        (ax_grip,   'Grip Factor',         'Grip: True vs RLS'),
        (ax_rls,    'TYRE_LAT a2',         'RLS a2 Estimate'),
        (ax_ekf_vx, 'RMSE vx (m/s)',       'EKF vx Error'),
        (ax_nn,     '|NN Residual|',        'NN Residual Magnitude'),
        (ax_model,  'Model Error RMS',      'Baseline Model Error'),
        (ax_vx,     'Mean vx (m/s)',        'Estimated Speed'),
    ]:
        ax.set_xlabel('Lap')
        ax.set_ylabel(ylabel)
        ax.set_title(title_str, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    path = RESULTS_DIR / f'{filename_stem}.png'
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Plot saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Final dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_final_dashboard(s1_results, s2_results, s3_results):
    """Big summary dashboard combining all three scenarios."""
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle('IRDAS Robustness Test — Full Summary Dashboard',
                 fontsize=15, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.5, wspace=0.4)

    # ── Row 0: Scenario labels ────────────────────────────────────────────────
    for col, label in enumerate(['Scenario 1\nSensor Noise',
                                  'Scenario 2\nModel Deviation',
                                  'Scenario 3\nIn-Race Events',
                                  'Cross-Scenario\nComparison']):
        ax = fig.add_subplot(gs[0, col])
        ax.text(0.5, 0.5, label, ha='center', va='center',
                fontsize=11, fontweight='bold', transform=ax.transAxes)
        ax.set_axis_off()

    def plot_metric(ax, scenario_list, attr, ylabel, title, legend=True):
        for idx, sr in enumerate(scenario_list):
            if not sr.laps: continue
            laps = [l.lap + 1 for l in sr.laps]
            vals = sr.metric_array(attr)
            ax.plot(laps, vals, '-o', color=COLORS[idx % len(COLORS)],
                    lw=2, ms=3, label=sr.description)
            for l in sr.laps:
                if l.event:
                    ax.axvline(l.lap + 1, color='gray', lw=1, ls=':', alpha=0.5)
        ax.set_xlabel('Lap', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        if legend:
            ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # ── Row 1: EKF error ─────────────────────────────────────────────────────
    plot_metric(fig.add_subplot(gs[1, 0]), s1_results, 'ekf_rmse_vx',
                'RMSE vx (m/s)', 'S1: EKF vx Error vs Noise')
    plot_metric(fig.add_subplot(gs[1, 1]), s2_results, 'ekf_rmse_vx',
                'RMSE vx (m/s)', 'S2: EKF vx Error vs Mismatch')
    plot_metric(fig.add_subplot(gs[1, 2]), s3_results, 'ekf_rmse_vx',
                'RMSE vx (m/s)', 'S3: EKF vx Error during Events')

    # Cross-scenario: mean EKF error per scenario
    ax_cross = fig.add_subplot(gs[1, 3])
    all_results = s1_results + s2_results + s3_results
    means  = [np.mean(sr.metric_array('ekf_rmse_vx')) for sr in all_results]
    labels = [sr.description[:15] for sr in all_results]
    colors = [COLORS[i % len(COLORS)] for i in range(len(all_results))]
    ax_cross.barh(labels, means, color=colors, alpha=0.8)
    ax_cross.set_xlabel('Mean EKF vx RMSE (m/s)', fontsize=8)
    ax_cross.set_title('Mean EKF Error — All Scenarios', fontsize=9)
    ax_cross.tick_params(labelsize=6)
    ax_cross.grid(True, alpha=0.3, axis='x')

    # ── Row 2: RLS tracking ───────────────────────────────────────────────────
    plot_metric(fig.add_subplot(gs[2, 0]), s1_results, 'rls_a2',
                'TYRE_LAT a2', 'S1: RLS a2 vs Noise')
    plot_metric(fig.add_subplot(gs[2, 1]), s2_results, 'rls_a2',
                'TYRE_LAT a2', 'S2: RLS a2 vs Mismatch')
    plot_metric(fig.add_subplot(gs[2, 2]), s3_results, 'rls_a2',
                'TYRE_LAT a2', 'S3: RLS a2 during Events')

    # Cross: final a2 drift from baseline
    ax_rls_cross = fig.add_subplot(gs[2, 3])
    baseline_a2  = TYRE_LAT['a2']
    drifts = [abs(sr.laps[-1].rls_a2 - baseline_a2) / baseline_a2 * 100
              for sr in all_results if sr.laps]
    ax_rls_cross.barh(labels, drifts, color=colors, alpha=0.8)
    ax_rls_cross.set_xlabel('Final a2 drift from baseline (%)', fontsize=8)
    ax_rls_cross.set_title('RLS Final Drift — All Scenarios', fontsize=9)
    ax_rls_cross.tick_params(labelsize=6)
    ax_rls_cross.grid(True, alpha=0.3, axis='x')

    # ── Row 3: Model error & NN residual ─────────────────────────────────────
    plot_metric(fig.add_subplot(gs[3, 0]), s1_results, 'model_error_rms',
                'Model Error RMS', 'S1: Model Error vs Noise')
    plot_metric(fig.add_subplot(gs[3, 1]), s2_results, 'model_error_rms',
                'Model Error RMS', 'S2: Model Error vs Mismatch')
    plot_metric(fig.add_subplot(gs[3, 2]), s3_results, 'nn_residual',
                '|NN Residual|', 'S3: NN Residual during Events')

    # Cross: NN residual mean
    ax_nn_cross = fig.add_subplot(gs[3, 3])
    nn_means = [np.mean(sr.metric_array('nn_residual')) for sr in all_results]
    ax_nn_cross.barh(labels, nn_means, color=colors, alpha=0.8)
    ax_nn_cross.set_xlabel('Mean NN Residual', fontsize=8)
    ax_nn_cross.set_title('Mean NN Residual — All Scenarios', fontsize=9)
    ax_nn_cross.tick_params(labelsize=6)
    ax_nn_cross.grid(True, alpha=0.3, axis='x')

    path = RESULTS_DIR / 'final_dashboard.png'
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  → Final dashboard saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Performance metrics table
# ─────────────────────────────────────────────────────────────────────────────

def print_performance_metrics(s1_results, s2_results, s3_results):
    all_results = s1_results + s2_results + s3_results
    baseline_a2 = TYRE_LAT['a2']

    print("\n" + "="*90)
    print("FINAL PERFORMANCE METRICS")
    print("="*90)
    header = (f"{'Scenario':<30} {'EKF_vx_RMSE':>12} {'EKF_r_RMSE':>11} "
              f"{'Model_Err':>10} {'NN_Res':>8} {'a2_drift%':>10} {'Notes'}")
    print(header)
    print("-"*90)

    for sr in all_results:
        if not sr.laps: continue
        ekf_vx = np.mean(sr.metric_array('ekf_rmse_vx'))
        ekf_r  = np.mean(sr.metric_array('ekf_rmse_r'))
        merr   = np.mean(sr.metric_array('model_error_rms'))
        nn_res = np.mean(sr.metric_array('nn_residual'))
        drift  = abs(sr.laps[-1].rls_a2 - baseline_a2) / baseline_a2 * 100
        events = [l.event for l in sr.laps if l.event]
        note   = ', '.join(events[:2]) if events else '—'
        print(f"  {sr.description:<28} {ekf_vx:>12.4f} {ekf_r:>11.4f} "
              f"{merr:>10.4f} {nn_res:>8.3f} {drift:>9.1f}%  {note}")

    print("="*90)

    # Scenario-level summaries
    print("\nSCENARIO SUMMARIES:")
    print("\n  Scenario 1 — Sensor Noise Impact on EKF:")
    for sr in s1_results:
        if not sr.laps: continue
        print(f"    {sr.description:<20}: mean EKF_vx = "
              f"{np.mean(sr.metric_array('ekf_rmse_vx')):.4f} m/s  "
              f"(peak = {np.max(sr.metric_array('ekf_rmse_vx')):.4f})")

    print("\n  Scenario 2 — Model Mismatch Impact on Adaptation:")
    for sr in s2_results:
        if not sr.laps: continue
        a2_start = sr.laps[0].rls_a2
        a2_end   = sr.laps[-1].rls_a2
        print(f"    {sr.description:<22}: a2 {a2_start:.3f} → {a2_end:.3f} "
              f"(Δ={a2_end-a2_start:+.3f})  model_err_mean="
              f"{np.mean(sr.metric_array('model_error_rms')):.4f}")

    print("\n  Scenario 3 — Event Detection (RLS response lag):")
    for sr in s3_results:
        event_laps = [l for l in sr.laps if l.event]
        for el in event_laps:
            print(f"    Lap {el.lap+1:>2}: '{el.event}' | "
                  f"rls_a2={el.rls_a2:.3f} | "
                  f"model_err={el.model_error_rms:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t_start = time.time()
    print("IRDAS Robustness Test Suite")
    print("="*60)
    print(f"  Laps per scenario: {N_LAPS}")
    print(f"  Steps per lap:     {STEPS_PER_LAP} ({STEPS_PER_LAP * DT:.1f}s)")
    print(f"  Results dir:       {RESULTS_DIR}")

    baseline_params = make_baseline_params()

    # ── Run all three scenarios ───────────────────────────────────────────────
    s1_results = run_scenario_1(baseline_params)
    s2_results = run_scenario_2(baseline_params)
    s3_results = run_scenario_3(baseline_params)

    # ── Final outputs ─────────────────────────────────────────────────────────
    print("\nGenerating final dashboard...")
    plot_final_dashboard(s1_results, s2_results, s3_results)
    print_performance_metrics(s1_results, s2_results, s3_results)

    # Save raw results
    raw = {
        'scenario1': s1_results,
        'scenario2': s2_results,
        'scenario3': s3_results,
    }
    with open(RESULTS_DIR / 'raw_results.pkl', 'wb') as f:
        pickle.dump(raw, f)

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed/60:.1f} min")
    print(f"All outputs in: {RESULTS_DIR}/")
    print("\nFiles generated:")
    for p in sorted(RESULTS_DIR.iterdir()):
        print(f"  {p.name}")
