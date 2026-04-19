"""
race_scenario.py — Tyre Degradation Race Scenario

Simulates a 30-lap stint where tyre grip degrades progressively and fuel mass burns down.
Compares three prediction sources at each lap:

    1. Theoretical degradation  — what the physics model says grip should be
    2. RLS parameter adaptation — what the online adapter has estimated
    3. NN residual correction   — what the neural network is correcting for

This demonstrates the full IRDAS pipeline in a realistic racing context.

Run from the IRDAS directory:
    python race_scenario.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

from params import (L, TF, TR, H, MX, M, Cd, Cl, Area, CX, K,
                    final_drive, tyre_radius, TYRE_LAT, TYRE_LON,
                    GEAR_RATIOS, UPSHIFT_SPEED_KPH, ENGINE_RPM, ENGINE_TORQUE_NM)
from twin_track import twin_track_model
from irdas_main import IRDAS

try:
    from residual_network import generate_ou_controls
except ImportError:
    def generate_ou_controls(n, dt=0.05):
        """Fallback: simple random controls if improved module not available."""
        controls = []
        steer, throttle = 0.0, 0.6
        for _ in range(n):
            steer    = np.clip(steer    + np.random.randn() * 0.02, -0.25, 0.25)
            throttle = np.clip(throttle + np.random.randn() * 0.05,  0.3,  0.9)
            controls.append([steer, throttle, 0.0])
        return np.array(controls)


# ─────────────────────────────────────────────
#  Race configuration
# ─────────────────────────────────────────────
N_LAPS           = 30
STEPS_PER_LAP    = 400          # 400 × 0.05s = 20s per lap
DT               = 0.05
TRACK_LENGTH_M   = 5200.0       # Representative modern race-track lap length
MIN_VALID_SPEED_MPS = 0.1
DYNAMIC_STATE_START = 3
DYNAMIC_STATE_END = 10
INITIAL_STATE    = np.array([0., 0., 0., 30., 0., 0.,
                              30., 30., 30., 30., 8000., 4., 0.5])

# Tyre degradation model parameters
WEAR_RATE        = 0.007      # grip loss per lap (~1.2%)
THERMAL_OPTIMAL  = 90.0         # optimal tyre temperature (°C)
THERMAL_SIGMA    = 25.0         # temperature spread (°C)
THERMAL_PENALTY  = 0.003        # grip loss per °C from optimal


# ─────────────────────────────────────────────
#  Tyre degradation model
# ─────────────────────────────────────────────

def tyre_temperature(lap: int, n_laps: int) -> float:
    """
    Simulate tyre temperature over a stint.

    Tyres heat up in the first few laps, peak, then cool slightly
    as compound wears and thermal capacity reduces.

    Args:
        lap:    current lap (0-indexed)
        n_laps: total laps in stint

    Returns:
        tyre temperature in °C
    """
    # Ramp up to optimal in first 3 laps, then slight cooling as wear increases
    warmup   = min(lap / 3.0, 1.0)
    cooldown = 1.0 - 0.15 * (lap / n_laps)     # 15% cooling over stint
    return THERMAL_OPTIMAL * warmup * cooldown + 60.0 * (1 - warmup)


def theoretical_grip(lap: int, n_laps: int) -> float:
    """
    Theoretical grip factor from tyre degradation model.

    Combines wear degradation and thermal effects.

    Args:
        lap:    current lap (0-indexed)
        n_laps: total laps

    Returns:
        grip factor (1.0 = baseline, lower = degraded)
    """
    # Wear factor: linear grip loss per lap
    wear_factor    = max(1.0 - WEAR_RATE * lap, 0.75)

    # Thermal factor: grip peaks at THERMAL_OPTIMAL
    temp           = tyre_temperature(lap, n_laps)
    temp_factor    = 1.0 - THERMAL_PENALTY * abs(temp - THERMAL_OPTIMAL)
    temp_factor    = max(temp_factor, 0.7)

    return wear_factor * temp_factor


def degrade_tyre_params(baseline_params: dict, lap: int, n_laps: int) -> dict:
    """
    Return a copy of params with degraded tyre coefficients.

    Modifies the Pacejka D parameter (peak friction) and B parameter
    (stiffness) to simulate real tyre wear behaviour:
      - D decreases with wear (less peak grip)
      - B increases slightly with wear (stiffer but narrower peak)

    Args:
        baseline_params: original vehicle parameters
        lap:             current lap
        n_laps:          total laps

    Returns:
        modified params dict with degraded tyre parameters
    """
    grip = theoretical_grip(lap, n_laps)

    degraded = baseline_params.copy()
    degraded['TYRE_LAT'] = baseline_params['TYRE_LAT'].copy()
    degraded['TYRE_LON'] = baseline_params['TYRE_LON'].copy()

    # Peak friction degrades with wear
    degraded['TYRE_LAT']['a2'] = baseline_params['TYRE_LAT']['a2'] * grip
    degraded['TYRE_LON']['a2'] = baseline_params['TYRE_LON']['a2'] * grip

    # Stiffness increases slightly as rubber hardens
    stiffness_factor = 1.0 + 0.1 * (1.0 - grip)
    degraded['TYRE_LAT']['B'] = baseline_params['TYRE_LAT']['B'] * stiffness_factor
    degraded['TYRE_LON']['B'] = baseline_params['TYRE_LON']['B'] * stiffness_factor

    return degraded


# ─────────────────────────────────────────────
#  Race simulation
# ─────────────────────────────────────────────

def run_race_scenario(irdas: IRDAS, n_laps: int = N_LAPS,
                      steps_per_lap: int = STEPS_PER_LAP) -> dict:
    """
    Run a full race stint with progressive tyre degradation.

    At each lap boundary, records:
      - Theoretical grip from degradation model
      - RLS-estimated tyre parameters (what IRDAS adapted to)
      - NN residual magnitude (proxy for what NN is correcting)

    Args:
        irdas:         initialised IRDAS object
        n_laps:        number of laps
        steps_per_lap: simulation steps per lap

    Returns:
        results dict with per-lap data
    """
    print(f"\nRunning {n_laps}-lap race stint ({n_laps * steps_per_lap * DT:.0f}s total)")
    print("=" * 60)

    # Per-lap tracking
    laps                  = list(range(n_laps))
    theoretical_grips     = []
    rls_lat_B             = []
    rls_lat_a2            = []
    rls_lon_B             = []
    nn_residual_magnitude = []
    lap_times             = []
    temperatures          = []
    vx_history            = []
    model_error_rmse      = []
    ekf_error_rmse        = []
    true_mass_history     = []
    estimated_mass_history = []
    fuel_used_history     = []

    # Full-race state continuity (no per-lap reset) for realistic accumulation effects.
    irdas.reset(INITIAL_STATE)
    lap_start_mass = irdas.true_vehicle_mass

    # Generate realistic race controls: straights, braking zones and corner exits.
    rng = np.random.default_rng(42)
    for lap in range(n_laps):
        lap_residuals = []
        lap_vx = []
        lap_model_error = []
        lap_ekf_error = []

        # Apply lap-level tyre degradation to the "real" vehicle model.
        degraded_params = degrade_tyre_params(irdas.baseline_params, lap, n_laps)
        irdas.real_simulator.true_params.update({
            'TYRE_LAT': degraded_params['TYRE_LAT'],
            'TYRE_LON': degraded_params['TYRE_LON'],
        })

        for step in range(steps_per_lap):
            lap_phase = step / steps_per_lap
            if lap_phase < 0.35:  # long straight
                steer = rng.normal(0.0, 0.005)
                throttle = np.clip(0.92 + rng.normal(0, 0.02), 0.8, 1.0)
                brake = 0.0
            elif lap_phase < 0.48:  # heavy braking
                steer = np.clip(0.02 * np.sin(2 * np.pi * lap_phase * 6), -0.06, 0.06)
                throttle = np.clip(0.15 + rng.normal(0, 0.02), 0.0, 0.25)
                brake = np.clip(0.45 + rng.normal(0, 0.05), 0.25, 0.65)
            elif lap_phase < 0.72:  # medium-speed corner
                steer = np.clip(0.17 * np.sin(2 * np.pi * (lap_phase - 0.48) * 2.5), -0.22, 0.22)
                throttle = np.clip(0.55 + rng.normal(0, 0.03), 0.35, 0.75)
                brake = 0.0
            else:  # corner exit and short straight
                steer = np.clip(0.08 * np.sin(2 * np.pi * (lap_phase - 0.72) * 4), -0.12, 0.12)
                throttle = np.clip(0.82 + rng.normal(0, 0.03), 0.65, 0.95)
                brake = 0.0

            u = np.array([steer, throttle, brake], dtype=float)

            try:
                estimated_state = irdas.step(u, use_nn_correction=True, use_param_adaptation=True)

                if irdas.nn_trained:
                    from residual_network import extract_dynamics_states
                    dyn_state = extract_dynamics_states(estimated_state)
                    residual = irdas.nn_learner.predict(dyn_state, u)
                    lap_residuals.append(np.linalg.norm(residual))

                lap_vx.append(float(estimated_state[3]))
                lap_model_error.append(float(irdas.history['model_errors'][-1]))
                # EKF accuracy metric focused on dynamic states (vx, vy, r, 4 wheel speeds).
                lap_ekf_error.append(float(np.linalg.norm(
                    (irdas.true_state - estimated_state)[DYNAMIC_STATE_START:DYNAMIC_STATE_END]
                )))
            except Exception as e:
                print(f"  Warning lap {lap+1}, step {step+1}: {e}")
                continue

        grip = theoretical_grip(lap, n_laps)
        temp = tyre_temperature(lap, n_laps)
        theoretical_grips.append(grip)
        temperatures.append(temp)

        current = irdas.param_adapter.get_current_params()
        rls_lat_B.append(current['TYRE_LAT']['B'])
        rls_lat_a2.append(current['TYRE_LAT']['a2'])
        rls_lon_B.append(current['TYRE_LON']['B'])

        nn_residual_magnitude.append(np.mean(lap_residuals) if lap_residuals else 0.0)
        mean_vx = np.mean(lap_vx) if lap_vx else 0.0
        vx_history.append(mean_vx)
        # Treat near-standstill laps as invalid for lap-time estimation.
        if mean_vx <= MIN_VALID_SPEED_MPS:
            lap_times.append(np.nan)
        else:
            lap_times.append(TRACK_LENGTH_M / mean_vx)

        model_error_rmse.append(float(np.sqrt(np.mean(np.square(lap_model_error)))) if lap_model_error else 0.0)
        ekf_error_rmse.append(float(np.sqrt(np.mean(np.square(lap_ekf_error)))) if lap_ekf_error else 0.0)

        true_mass = float(irdas.true_vehicle_mass)
        est_mass = float(irdas.kalman_filter.get_mass_estimate())
        true_mass_history.append(true_mass)
        estimated_mass_history.append(est_mass)
        fuel_used_history.append(max(0.0, lap_start_mass - true_mass))
        lap_start_mass = true_mass

        print(f"  Lap {lap+1:>2}/{n_laps} | Grip {grip:.3f} | RLS a2 {rls_lat_a2[-1]:.3f} | "
              f"EKF RMSE {ekf_error_rmse[-1]:.3f} | Fuel used {fuel_used_history[-1]:.2f} kg")

    results = {
        'laps':                   laps,
        'theoretical_grip':       theoretical_grips,
        'temperatures':           temperatures,
        'rls_lat_B':              rls_lat_B,
        'rls_lat_a2':             rls_lat_a2,
        'rls_lon_B':              rls_lon_B,
        'nn_residual_magnitude':  nn_residual_magnitude,
        'vx_history':             vx_history,
        'lap_times':              lap_times,
        'model_error_rmse':       model_error_rmse,
        'ekf_error_rmse':         ekf_error_rmse,
        'true_mass_history':      true_mass_history,
        'estimated_mass_history': estimated_mass_history,
        'fuel_used_history':      fuel_used_history,
        'baseline_lat_a2':        irdas.baseline_params['TYRE_LAT']['a2'],
        'baseline_lat_B':         irdas.baseline_params['TYRE_LAT']['B'],
        'initial_mass':           float(irdas.baseline_params['M']),
        'n_laps':                 n_laps,
    }

    return results


# ─────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────

def plot_race_results(results: dict, save_path: str = 'results/race_scenario.png'):
    """
    Plot comparison of theoretical vs RLS vs NN predictions across the stint.

    Four panels:
      1. Grip factor: theoretical vs RLS-estimated
      2. Tyre stiffness B: baseline vs RLS-estimated
      3. NN residual magnitude — what the network is correcting
      4. Mean speed per lap — performance consequence of degradation
    """
    Path(save_path).parent.mkdir(exist_ok=True)

    laps     = np.array(results['laps']) + 1     # 1-indexed for display
    n_laps   = results['n_laps']
    baseline_a2 = results['baseline_lat_a2']
    baseline_B  = results['baseline_lat_B']

    # Compute RLS grip proxy: ratio of adapted a2 to baseline a2
    rls_grip_proxy = np.array(results['rls_lat_a2']) / baseline_a2

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle('IRDAS Race Scenario: 30-Lap Tyre Degradation Stint',
                 fontsize=14, fontweight='bold', y=1.01)

    colors = {'theoretical': '#2196F3',   # blue
              'rls':         '#F44336',   # red
              'nn':          '#4CAF50',   # green
              'speed':       '#FF9800'}   # orange

    # ── Panel 1: Grip factor ─────────────────────────────────
    ax = axes[0, 0]
    ax.plot(laps, results['theoretical_grip'], '-o',
            color=colors['theoretical'], linewidth=2, markersize=4,
            label='Theoretical (degradation model)')
    ax.plot(laps, rls_grip_proxy, '-s',
            color=colors['rls'], linewidth=2, markersize=4,
            label='RLS estimate (adapted a2 ratio)')
    ax.axhline(1.0, color='gray', linewidth=0.8, linestyle='--', label='Baseline')
    ax.set_xlabel('Lap')
    ax.set_ylabel('Grip Factor')
    ax.set_title('Tyre Grip: Theoretical vs RLS Estimate')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.1])

    # ── Panel 2: Stiffness B parameter ───────────────────────
    ax = axes[0, 1]
    # Theoretical B: increases as tyre hardens
    theoretical_B = [baseline_B * (1.0 + 0.1 * (1.0 - theoretical_grip(l, n_laps)))
                     for l in results['laps']]
    ax.plot(laps, theoretical_B, '-o',
            color=colors['theoretical'], linewidth=2, markersize=4,
            label='Theoretical B (hardening model)')
    ax.plot(laps, results['rls_lat_B'], '-s',
            color=colors['rls'], linewidth=2, markersize=4,
            label='RLS adapted B')
    ax.axhline(baseline_B, color='gray', linewidth=0.8, linestyle='--',
               label=f'Baseline B = {baseline_B:.1f}')
    ax.set_xlabel('Lap')
    ax.set_ylabel('Lateral Stiffness B')
    ax.set_title('Tyre Stiffness: Theoretical vs RLS Estimate')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: NN residual magnitude ───────────────────────
    ax = axes[1, 0]
    ax.plot(laps, results['nn_residual_magnitude'], '-^',
            color=colors['nn'], linewidth=2, markersize=5,
            label='NN |residual|')

    # Overlay theoretical grip for comparison
    ax2 = ax.twinx()
    ax2.plot(laps, results['theoretical_grip'], '--',
             color=colors['theoretical'], linewidth=1.5, alpha=0.6,
             label='Theoretical grip (ref)')
    ax2.set_ylabel('Grip Factor', color=colors['theoretical'])
    ax2.tick_params(axis='y', labelcolor=colors['theoretical'])

    ax.set_xlabel('Lap')
    ax.set_ylabel('Mean |NN Residual|', color=colors['nn'])
    ax.tick_params(axis='y', labelcolor=colors['nn'])
    ax.set_title('NN Residual Magnitude vs Grip')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Mean speed + lap time per lap ───────────────
    ax = axes[1, 1]
    ax.plot(laps, results['vx_history'], '-D',
            color=colors['speed'], linewidth=2, markersize=5,
            label='Mean vx (estimated)')
    ax.set_xlabel('Lap')
    ax.set_ylabel('Mean Speed (m/s)')
    ax.set_title('Estimated Speed and Lap Time')
    ax2 = ax.twinx()
    ax2.plot(laps, results['lap_times'], '--', color='black', linewidth=1.5, label='Lap time (s)')
    ax2.set_ylabel('Lap Time (s)')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 5: Mass tracking and fuel consumption ──────────
    ax = axes[2, 0]
    ax.plot(laps, results['true_mass_history'], '-o', linewidth=2, markersize=4, label='True mass')
    ax.plot(laps, results['estimated_mass_history'], '-s', linewidth=2, markersize=4, label='EKF mass estimate')
    ax.set_xlabel('Lap')
    ax.set_ylabel('Vehicle Mass (kg)')
    ax.set_title('Fuel Burn and Mass Estimation')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.bar(laps, results['fuel_used_history'], alpha=0.2, color='tab:orange', label='Fuel used/lap')
    ax2.set_ylabel('Fuel Used per Lap (kg)')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # ── Panel 6: Accuracy metrics ─────────────────────────────
    ax = axes[2, 1]
    ax.plot(laps, results['model_error_rmse'], '-^', linewidth=2, markersize=4, label='Model RMSE')
    ax.plot(laps, results['ekf_error_rmse'], '-v', linewidth=2, markersize=4, label='EKF dynamics RMSE')
    ax.set_xlabel('Lap')
    ax.set_ylabel('RMSE')
    ax.set_title('Model and EKF Dynamics Accuracy per Lap')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Temperature annotation on panel 1
    axes[0, 0].twinx().plot(laps, results['temperatures'], ':',
                            color='purple', linewidth=1.2, alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    return fig


def print_stint_summary(results: dict):
    """Print a text summary of the stint analysis."""
    print("\n" + "=" * 60)
    print("STINT SUMMARY")
    print("=" * 60)

    baseline_a2 = results['baseline_lat_a2']
    final_grip  = results['theoretical_grip'][-1]
    final_rls   = results['rls_lat_a2'][-1] / baseline_a2

    print(f"\nTyre degradation over {results['n_laps']} laps:")
    print(f"  Theoretical grip loss:  {(1 - final_grip)*100:.1f}%")
    print(f"  RLS estimated loss:     {(1 - final_rls)*100:.1f}%")
    print(f"  Agreement:              {abs(final_grip - final_rls)*100:.1f}% difference")

    print(f"\nNN residual magnitude:")
    print(f"  Lap 1:   {results['nn_residual_magnitude'][0]:.4f}")
    print(f"  Lap {results['n_laps']//2}: {results['nn_residual_magnitude'][results['n_laps']//2 - 1]:.4f}")
    print(f"  Lap {results['n_laps']}: {results['nn_residual_magnitude'][-1]:.4f}")

    # Check if NN residual tracks degradation
    corr = np.corrcoef(
        results['nn_residual_magnitude'],
        [1 - g for g in results['theoretical_grip']]
    )[0, 1]
    print(f"\n  Correlation with degradation: {corr:.3f}")
    if corr > 0.5:
        print("  → NN residual tracks tyre degradation ✓")
    elif corr > 0.2:
        print("  → NN residual weakly tracks degradation")
    else:
        print("  → NN residual does not clearly track degradation")
        print("    (expected if NN was trained on fresh-tyre data only)")

    print(f"\nSpeed analysis:")
    print(f"  Mean speed lap 1:  {results['vx_history'][0]:.2f} m/s")
    print(f"  Mean speed lap {results['n_laps']}: {results['vx_history'][-1]:.2f} m/s")
    speed_drop = results['vx_history'][0] - results['vx_history'][-1]
    print(f"  Speed loss:        {speed_drop:.2f} m/s ({speed_drop/results['vx_history'][0]*100:.1f}%)")

    print(f"\nMass and fuel analysis:")
    print(f"  Start mass:        {results['initial_mass']:.2f} kg")
    print(f"  End true mass:     {results['true_mass_history'][-1]:.2f} kg")
    print(f"  End EKF mass:      {results['estimated_mass_history'][-1]:.2f} kg")
    mass_error = abs(results['true_mass_history'][-1] - results['estimated_mass_history'][-1])
    print(f"  Final mass error:  {mass_error:.2f} kg")
    print(f"  Total fuel used:   {np.sum(results['fuel_used_history']):.2f} kg")

    print(f"\nAccuracy summary:")
    print(f"  Mean model RMSE:   {np.mean(results['model_error_rmse']):.4f}")
    print(f"  Mean EKF RMSE:     {np.mean(results['ekf_error_rmse']):.4f}")


# ─────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("IRDAS Race Scenario: Tyre Degradation Analysis")
    print("=" * 60)

    # Build baseline params
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

    # ── Initialise IRDAS ──────────────────────────────────────
    print("\nInitialising IRDAS...")
    irdas = IRDAS(baseline_params, device='cpu', use_nn=True, use_rls=True)
    irdas.initialize_real_vehicle(seed=42)

    # ── Pretrain NN ───────────────────────────────────────────
    print("\nPretraining neural network (3000 samples)...")
    irdas.pretrain_neural_network(
        n_training_samples=3000,
        epochs=150,
        batch_size=64
    )

    # ── Reset for race ────────────────────────────────────────
    print("\nResetting for race simulation...")
    irdas.reset(INITIAL_STATE)

    # ── Run race ──────────────────────────────────────────────
    results = run_race_scenario(irdas, n_laps=N_LAPS, steps_per_lap=STEPS_PER_LAP)

    # ── Print summary ─────────────────────────────────────────
    print_stint_summary(results)

    # ── Plot ──────────────────────────────────────────────────
    plot_race_results(results, save_path='results/race_scenario.png')

    # ── Save raw results ──────────────────────────────────────
    Path('results').mkdir(exist_ok=True)
    with open('results/race_scenario.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Raw results saved to results/race_scenario.pkl")

    print("\nDone. Check results/race_scenario.png for the plots.")
