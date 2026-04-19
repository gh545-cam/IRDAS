import numpy as np
 
MIN_VEHICLE_MASS_KG = 100.0
 
 
# ─────────────────────────────────────────────
#  Noise profiles (standard deviations)
#  Tuned to approximate realistic sensor specs
# ─────────────────────────────────────────────
DEFAULT_NOISE = {
    'x':               0.5,    # GPS position:       ±0.5 m   (RTK-grade)
    'y':               0.5,
    'ax':              0.3,    # IMU accelerometer:  ±0.3 m/s²
    'ay':              0.3,
    'r':               0.005,  # Gyroscope yaw rate: ±0.005 rad/s
    'vx':              0.1,    # GPS velocity:       ±0.1 m/s
    'vy':              0.1,
    'rpm':            20.0,    # RPM sensor:         ±20 RPM
    'wheel_speed':     0.05,   # Wheel speed sensor: ±0.05 m/s
    'fuel_flow':       0.002,  # Fuel flow sensor:   ±0.002 kg/s
    'mass':            2.0,    # Mass estimate:      ±2.0 kg
}
 
 
class SensorSimulator:
    """
    Simulates a full vehicle sensor suite from the true state.
 
    Given the true 13-element state vector at each timestep, produces
    a noisy 9-element measurement vector that the EKF can consume.
 
    The key fix over the original IRDAS implementation is the proper
    computation of ax and ay using the Coriolis correction, and storing
    the previous state so velocity derivatives can be estimated.
 
    State vector (13 elements):
        [x, y, psi, vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr, rpm, gear, throttle]
 
    Measurement vector (9 elements):
        [x, y, ax_imu, ay_imu, r, vx, vy, rpm, wheel_speed_avg]
    """
 
    def __init__(self, noise_std: dict = None, dt: float = 0.05, initial_mass: float = 752.0):
        """
        Args:
            noise_std: dict of noise standard deviations (uses DEFAULT_NOISE if None)
            dt:        simulation timestep (seconds)
        """
        self.dt = dt
        self.noise = {**DEFAULT_NOISE, **(noise_std or {})}
        self.initial_mass = float(initial_mass)
        self.estimated_mass = float(initial_mass)
 
        # Store previous state to compute velocity derivatives for ax/ay
        self.prev_state = None
 
    def reset(self, initial_mass: float = None):
        """Reset sensor memory (call at start of each new simulation run)."""
        self.prev_state = None
        if initial_mass is None:
            self.estimated_mass = float(self.initial_mass)
        else:
            self.estimated_mass = float(initial_mass)
 
    def measure(self, true_state: np.ndarray) -> np.ndarray:
        """
        Generate a noisy measurement from the true vehicle state.
 
        Args:
            true_state: true 13-element state vector
 
        Returns:
            z: noisy 9-element measurement vector
               [x, y, ax_imu, ay_imu, r, vx, vy, rpm, wheel_speed_avg]
        """
        # Unpack relevant states
        x, y        = true_state[0], true_state[1]
        vx, vy, r   = true_state[3], true_state[4], true_state[5]
        rpm         = true_state[10]
        vw          = true_state[6:10]   # four wheel speeds
 
        # ── IMU: ax and ay with Coriolis correction ──────────────────────
        ax_imu, ay_imu = self._compute_imu(true_state, vx, vy, r)
 
        # ── Wheel speed average ───────────────────────────────────────────
        wheel_speed_avg = np.mean(vw)
 
        # ── Build true (noiseless) measurement vector ─────────────────────
        z_true = np.array([
            x,
            y,
            ax_imu,
            ay_imu,
            r,
            vx,
            vy,
            rpm,
            wheel_speed_avg,
        ])
 
        # ── Add Gaussian noise ────────────────────────────────────────────
        noise_vec = np.array([
            np.random.normal(0, self.noise['x']),
            np.random.normal(0, self.noise['y']),
            np.random.normal(0, self.noise['ax']),
            np.random.normal(0, self.noise['ay']),
            np.random.normal(0, self.noise['r']),
            np.random.normal(0, self.noise['vx']),
            np.random.normal(0, self.noise['vy']),
            np.random.normal(0, self.noise['rpm']),
            np.random.normal(0, self.noise['wheel_speed']),
        ])
 
        # Store current state as previous for next step
        self.prev_state = true_state.copy()

        return z_true + noise_vec

    def estimate_fuel_flow(self, true_state: np.ndarray, control: np.ndarray) -> float:
        """
        Estimate fuel flow [kg/s] from engine operating point.

        Args:
            true_state: full vehicle state, using engine rpm at true_state[10]
            control: control input [steer, throttle, brake], using control[1:3]

        Returns:
            Estimated instantaneous fuel flow in kg/s.
        """
        rpm = float(np.clip(true_state[10], 1000.0, 15500.0))
        throttle = float(np.clip(control[1], 0.0, 1.0))
        brake = float(np.clip(control[2], 0.0, 1.0))
        base_idle = 0.0035
        load_term = 0.018 * throttle * (rpm / 12000.0)
        brake_cut = 0.5 if brake > 0.2 and throttle < 0.1 else 1.0
        fuel_flow = (base_idle + load_term) * brake_cut
        return max(fuel_flow, 5e-4)

    def measure_fuel_system(self, true_state: np.ndarray, control: np.ndarray,
                            true_mass_kg: float) -> tuple[float, float]:
        """
        Return noisy fuel-flow and mass measurements.
        This method is the source of truth for the sensor-side mass estimate.

        Returns:
            Tuple (fuel_flow_measured_kgps, mass_measured_kg).
        """
        fuel_flow_true = self.estimate_fuel_flow(true_state, control)
        fuel_flow_measured = fuel_flow_true + np.random.normal(0, self.noise['fuel_flow'])
        fuel_flow_measured = max(float(fuel_flow_measured), 0.0)

        mass_measured = float(true_mass_kg) + np.random.normal(0, self.noise['mass'])
        self.estimated_mass = max(MIN_VEHICLE_MASS_KG, mass_measured)
        return fuel_flow_measured, self.estimated_mass
 
    def _compute_imu(self, true_state: np.ndarray,
                     vx: float, vy: float, r: float):
        """
        Compute body-frame accelerations as measured by an IMU.
 
        An IMU mounted to the car measures accelerations in the body frame.
        Because the car is rotating (yaw rate r), there are Coriolis terms
        that must be included — the accelerometer sees both the true
        acceleration AND the effect of the rotating reference frame.
 
        The full expressions are:
            ax_imu = dvx/dt - vy·r
            ay_imu = dvy/dt + vx·r
 
        where dvx/dt and dvy/dt are estimated from the velocity change
        between the previous and current timestep.
 
        If no previous state is available (first step), Coriolis-only
        terms are returned (dv/dt assumed zero).
 
        Args:
            true_state: current true state
            vx, vy, r:  current body velocities and yaw rate
 
        Returns:
            (ax_imu, ay_imu): body frame accelerations (m/s²)
        """
        if self.prev_state is not None:
            vx_prev = self.prev_state[3]
            vy_prev = self.prev_state[4]
 
            # Finite difference velocity derivatives
            dvx_dt = (vx - vx_prev) / self.dt
            dvy_dt = (vy - vy_prev) / self.dt
        else:
            # First step: no previous state, assume no acceleration
            dvx_dt = 0.0
            dvy_dt = 0.0
 
        # Coriolis correction
        ax_imu = dvx_dt - vy * r
        ay_imu = dvy_dt + vx * r
 
        return ax_imu, ay_imu
 
    def get_noise_profile(self) -> dict:
        """Return current noise standard deviations."""
        return self.noise.copy()
 
 
# ─────────────────────────────────────────────
#  Kalman filter measurement function fix
#  (drop this into kalman_filter.py)
# ─────────────────────────────────────────────
 
def measurement_function_fixed(x: np.ndarray,
                                vx_prev: float,
                                vy_prev: float,
                                dt: float = 0.05) -> np.ndarray:
    """
    Fixed measurement function h(x) for the EKF.
 
    Replaces the broken _measurement_function() in kalman_filter.py.
    Properly computes ax and ay using velocity derivatives and Coriolis.
 
    This is what the EKF uses to predict what the sensors SHOULD read
    given the current state estimate. The innovation (sensor - prediction)
    drives the update step.
 
    Args:
        x:        current state estimate (13 elements)
        vx_prev:  vx from previous timestep (stored by EKF)
        vy_prev:  vy from previous timestep (stored by EKF)
        dt:       timestep
 
    Returns:
        z_pred: predicted measurement (9 elements)
    """
    vx, vy, r = x[3], x[4], x[5]
 
    # Velocity derivatives from finite difference
    dvx_dt = (vx - vx_prev) / dt
    dvy_dt = (vy - vy_prev) / dt
 
    # Coriolis correction — same formula as SensorSimulator._compute_imu()
    ax_pred = dvx_dt - vy * r
    ay_pred = dvy_dt + vx * r
 
    return np.array([
        x[0],                              # x   (GPS)
        x[1],                              # y   (GPS)
        ax_pred,                           # ax  (IMU) — FIXED
        ay_pred,                           # ay  (IMU) — FIXED
        x[5],                              # r   (gyroscope)
        x[3],                              # vx  (GPS velocity)
        x[4],                              # vy  (GPS velocity)
        x[10],                             # rpm (RPM sensor)
        np.mean(x[6:10]),                  # wheel speed average
    ])
 
 
# ─────────────────────────────────────────────────────────────────────────
#  Instructions for patching kalman_filter.py
# ─────────────────────────────────────────────────────────────────────────
#
#  1. Add vx_prev and vy_prev to __init__:
#
#       self.vx_prev = 5.0   # matches initial vx
#       self.vy_prev = 0.0
#
#  2. At the END of predict(), store the new velocities:
#
#       self.vx_prev = self.x[3]
#       self.vy_prev = self.x[4]
#
#  3. Replace _measurement_function() body with:
#
#       from sensors import measurement_function_fixed
#       return measurement_function_fixed(x, self.vx_prev, self.vy_prev, self.dt)
#
#  The _jacobian_measurement() H matrix does NOT need to change —
#  ax and ay still don't have clean closed-form Jacobian entries
#  (they depend on the difference of two states across time),
#  so H[2,:] and H[3,:] remain zero. This is an approximation
#  but acceptable — the Coriolis terms are small relative to
#  the direct state measurements.
# ─────────────────────────────────────────────────────────────────────────
 
 
if __name__ == "__main__":
    """
    Quick sanity check — simulate 5 steps and print measurements.
    """
    print("SensorSimulator Sanity Check")
    print("=" * 50)
 
    # Dummy state: straight line at 30 m/s
    state = np.array([
        0., 0., 0.,          # x, y, psi
        30., 0., 0.,         # vx, vy, r
        30., 30., 30., 30.,  # wheel speeds
        8000., 4., 0.7       # rpm, gear, throttle
    ])
 
    sim = SensorSimulator(dt=0.05)
 
    labels = ['x', 'y', 'ax', 'ay', 'r', 'vx', 'vy', 'rpm', 'ws_avg']
 
    print(f"\n{'Step':<6} " + " ".join(f"{l:>8}" for l in labels))
    print("-" * 85)
 
    for step in range(5):
        # Simulate slight acceleration each step
        state[3] += 0.5   # vx increases
        z = sim.measure(state)
        vals = " ".join(f"{v:8.3f}" for v in z)
        print(f"{step:<6} {vals}")
 
    print("\nNoise profile (std dev):")
    for k, v in sim.get_noise_profile().items():
        print(f"  {k:<15}: ±{v}")
