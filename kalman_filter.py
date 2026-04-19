"""
Unscented Kalman Filter (UKF) for vehicle state estimation.
Keeps the previous ExtendedKalmanFilter API for compatibility.
"""
import numpy as np
from scipy import linalg
from twin_track import twin_track_model


class ExtendedKalmanFilter:
    """
    Backward-compatible UKF implementation (class name retained).

    State vector (13 elements):
        [x, y, psi, vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr, engine_rpm, gear, throttle]

    Sensor measurements (9 elements):
        [x_gps, y_gps, ax, ay, r, vx_gps, vy_gps, engine_rpm, wheel_speeds (avg)]
    """
    PSI_INDEX = 2
    SIGMA_CONDITION_THRESHOLD = 1e12

    def __init__(self, params, process_noise=None, measurement_noise=None,
                 alpha=1e-2, beta=2.0, kappa=0.0):
        self.params = params
        self.n_states = 13
        self.n_measurements = 9

        # UKF scaling parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = self.alpha ** 2 * (self.n_states + self.kappa) - self.n_states

        self.vx_prev = 30.0
        self.vy_prev = 0.0

        self.x = np.array([0., 0., 0., 30., 0., 0., 30., 30., 30., 30., 8000., 4., 0.5], dtype=np.float64)
        self.P = np.eye(self.n_states, dtype=np.float64) * 1.0

        if process_noise is None:
            self.Q = np.diag([
                0.01, 0.01,
                0.01,
                0.1, 0.1, 0.01,
                0.5, 0.5, 0.5, 0.5,
                10.0, 0.0, 0.01
            ]).astype(np.float64)
        else:
            self.Q = process_noise.astype(np.float64)

        if measurement_noise is None:
            self.R = np.diag([
                1.0, 1.0,
                0.5, 0.5,
                0.01,
                0.1, 0.1,
                50.0,
                0.2
            ]).astype(np.float64)
        else:
            self.R = measurement_noise.astype(np.float64)

        self.dt = 0.05
        self.history = []

        self._compute_ukf_weights()
        self._sigma_points = None

    def _compute_ukf_weights(self):
        n = self.n_states
        lam = self.lambda_
        c = n + lam

        self.Wm = np.full(2 * n + 1, 1.0 / (2.0 * c), dtype=np.float64)
        self.Wc = np.full(2 * n + 1, 1.0 / (2.0 * c), dtype=np.float64)
        self.Wm[0] = lam / c
        self.Wc[0] = lam / c + (1.0 - self.alpha ** 2 + self.beta)

    def _state_postprocess(self, x):
        x = x.copy()
        x[0:2] = np.clip(x[0:2], -1e4, 1e4)
        x[self.PSI_INDEX] = np.arctan2(np.sin(x[self.PSI_INDEX]), np.cos(x[self.PSI_INDEX]))
        x[3:6] = np.clip(x[3:6], -50, 50)
        x[6:10] = np.clip(x[6:10], 0, 100)
        x[10] = np.clip(x[10], 1000, 15500)
        x[11] = int(np.clip(round(x[11]), 1, 8))
        x[12] = np.clip(x[12], 0, 1)
        return x

    def _generate_sigma_points(self, mean, cov):
        n = self.n_states
        c = n + self.lambda_

        # Defensive symmetrization: repeated covariance updates can introduce tiny
        # floating-point asymmetry, which can break Cholesky decomposition.
        cov = 0.5 * (cov + cov.T)
        jitter = 1e-9
        # Retry a few times with increasing diagonal jitter up to ~1e-5 to recover
        # near-PSD matrices caused by numerical noise.
        for _ in range(5):
            try:
                sqrt_cov = linalg.cholesky(c * (cov + np.eye(n) * jitter), lower=True)
                break
            except linalg.LinAlgError:
                jitter *= 10.0
        else:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, 1e-9, None)
            sqrt_cov = eigvecs @ np.diag(np.sqrt(c * eigvals))

        sigma = np.zeros((2 * n + 1, n), dtype=np.float64)
        sigma[0] = mean
        for i in range(n):
            sigma[i + 1] = mean + sqrt_cov[:, i]
            sigma[n + i + 1] = mean - sqrt_cov[:, i]
        return sigma

    def _mean_from_sigma(self, sigma_points):
        mean = np.tensordot(self.Wm, sigma_points, axes=(0, 0))

        # Proper circular averaging for yaw angle psi
        sin_psi = np.tensordot(self.Wm, np.sin(sigma_points[:, self.PSI_INDEX]), axes=(0, 0))
        cos_psi = np.tensordot(self.Wm, np.cos(sigma_points[:, self.PSI_INDEX]), axes=(0, 0))
        mean[self.PSI_INDEX] = np.arctan2(sin_psi, cos_psi)
        return mean

    def _cov_from_sigma(self, sigma_points, mean, noise=None):
        n = mean.shape[0]
        cov = np.zeros((n, n), dtype=np.float64)

        for i in range(sigma_points.shape[0]):
            d = sigma_points[i] - mean
            d[self.PSI_INDEX] = np.arctan2(np.sin(d[self.PSI_INDEX]), np.cos(d[self.PSI_INDEX]))
            cov += self.Wc[i] * np.outer(d, d)

        if noise is not None:
            cov += noise

        cov = 0.5 * (cov + cov.T)
        return cov

    def predict(self, u, dt=None):
        if dt is None:
            dt = self.dt

        sigma = self._generate_sigma_points(self.x, self.P)

        propagated = np.zeros_like(sigma)
        for i in range(sigma.shape[0]):
            propagated[i] = self._state_postprocess(twin_track_model(sigma[i], u, dt, self.params))

        x_pred = self._mean_from_sigma(propagated)
        P_pred = self._cov_from_sigma(propagated, x_pred, noise=self.Q)

        self.x = self._state_postprocess(x_pred)
        self.P = P_pred
        self._sigma_points = propagated

        self.vx_prev = self.x[3]
        self.vy_prev = self.x[4]

    def update(self, z):
        if self._sigma_points is None:
            self._sigma_points = self._generate_sigma_points(self.x, self.P)

        sigma_meas = np.zeros((self._sigma_points.shape[0], self.n_measurements), dtype=np.float64)
        for i in range(self._sigma_points.shape[0]):
            sigma_meas[i] = self._measurement_function(self._sigma_points[i])

        z_pred = np.tensordot(self.Wm, sigma_meas, axes=(0, 0))

        S = np.zeros((self.n_measurements, self.n_measurements), dtype=np.float64)
        Pxz = np.zeros((self.n_states, self.n_measurements), dtype=np.float64)

        for i in range(self._sigma_points.shape[0]):
            dx = self._sigma_points[i] - self.x
            dx[self.PSI_INDEX] = np.arctan2(np.sin(dx[self.PSI_INDEX]), np.cos(dx[self.PSI_INDEX]))
            dz = sigma_meas[i] - z_pred
            S += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)

        S += self.R
        S = 0.5 * (S + S.T)

        try:
            # Use pseudoinverse for very ill-conditioned innovation covariance.
            if np.linalg.cond(S) > self.SIGMA_CONDITION_THRESHOLD:
                K = Pxz @ np.linalg.pinv(S)
            else:
                K = linalg.solve(S.T, Pxz.T, assume_a='sym').T
        except (linalg.LinAlgError, ValueError):
            K = Pxz @ np.linalg.pinv(S)

        innovation = z - z_pred
        self.x = self.x + K @ innovation
        self.x = self._state_postprocess(self.x)

        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # Keep gear uncertainty collapsed (discrete state)
        self.P[11, :] = 0.0
        self.P[:, 11] = 0.0

    def _measurement_function(self, x):
        from sensors import measurement_function_fixed
        return measurement_function_fixed(x, self.vx_prev, self.vy_prev, self.dt)

    def get_state(self):
        return self.x.copy()

    def get_covariance(self):
        return self.P.copy()

    def get_uncertainty(self):
        return np.sqrt(np.clip(np.diag(self.P), 0.0, None))

    def reset(self, initial_state=None):
        if initial_state is not None:
            self.x = np.asarray(initial_state, dtype=np.float64).copy()
        else:
            self.x = np.array([0., 0., 0., 30., 0., 0., 30., 30., 30., 30., 8000., 4., 0.5], dtype=np.float64)

        self.x = self._state_postprocess(self.x)
        self.P = np.eye(self.n_states, dtype=np.float64) * 1.0
        self.vx_prev = float(self.x[3])
        self.vy_prev = float(self.x[4])
        self._sigma_points = None


# Explicit alias for users who want UKF naming
UnscentedKalmanFilter = ExtendedKalmanFilter


def add_sensor_noise(state, measurement_noise_std):
    """
    Add realistic sensor noise to state measurements.

    Args:
        state: true state vector
        measurement_noise_std: dict with noise standard deviations

    Returns:
        noisy measurement vector
    """
    x_noise = np.random.normal(0, measurement_noise_std.get('x_gps', 1.0))
    y_noise = np.random.normal(0, measurement_noise_std.get('y_gps', 1.0))

    ax_noise = np.random.normal(0, measurement_noise_std.get('ax', 0.5))
    ay_noise = np.random.normal(0, measurement_noise_std.get('ay', 0.5))
    r_noise = np.random.normal(0, measurement_noise_std.get('r', 0.01))

    vx_noise = np.random.normal(0, measurement_noise_std.get('vx', 0.1))
    vy_noise = np.random.normal(0, measurement_noise_std.get('vy', 0.1))

    rpm_noise = np.random.normal(0, measurement_noise_std.get('rpm', 50))
    ws_noise = np.random.normal(0, measurement_noise_std.get('wheel_speed', 0.2))

    z = np.array([
        state[0] + x_noise,
        state[1] + y_noise,
        ax_noise,
        ay_noise,
        state[5] + r_noise,
        state[3] + vx_noise,
        state[4] + vy_noise,
        state[10] + rpm_noise,
        (state[6] + state[7] + state[8] + state[9]) / 4 + ws_noise
    ])

    return z


if __name__ == "__main__":
    print("Unscented Kalman Filter Test")
    print("=" * 50)

    test_params = {
        'L': 3.6, 'TF': 1.58, 'TR': 1.42, 'H': 0.295, 'MX': 0.453, 'M': 752,
        'Cd': 0.8, 'Cl': 3.5, 'Area': 1.2, 'CX': 0.52, 'K': 1.35,
        'final_drive': 6.3, 'tyre_radius': 0.330,
        'TYRE_LAT': {'B': 12.0, 'C': 1.9, 'E': -1.5, 'a1': -0.10, 'a2': 2.05},
        'TYRE_LON': {'B': 12.0, 'C': 1.7, 'E': -2.0, 'a1': -0.08, 'a2': 2.1},
        'GEAR_RATIOS': {1: 3.15, 2: 2.47, 3: 1.96, 4: 1.60, 5: 1.33, 6: 1.14, 7: 0.98, 8: 0.84},
        'UPSHIFT_SPEED_KPH': {1: 94.0, 2: 119.9, 3: 151.1, 4: 185.1, 5: 222.7, 6: 259.8, 7: 302.3},
        'ENGINE_RPM': np.array([3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 15500]),
        'ENGINE_TORQUE_NM': np.array([440, 500, 550, 540, 550, 550, 550, 540, 525, 500, 480, 440, 390, 340])
    }

    ukf = UnscentedKalmanFilter(test_params)
    print(f"UKF initialized: state dim={ukf.n_states}, measurement dim={ukf.n_measurements}")

    u = np.array([0.1, 0.5, 0.0])
    ukf.x = np.array([0., 0., 0., 5., 0., 0., 5., 5., 5., 5., 3000., 1., 0.1], dtype=np.float64)

    ukf.predict(u)
    print(f"After predict: vx={ukf.x[3]:.3f}, uncertainty={ukf.get_uncertainty()[3]:.4f}")

    noise_std = {'ax': 0.5, 'ay': 0.5, 'r': 0.01, 'vx': 0.1, 'vy': 0.1, 'rpm': 50, 'wheel_speed': 0.2}
    z = add_sensor_noise(ukf.x, noise_std)

    ukf.update(z)
    print(f"After update: vx={ukf.x[3]:.3f}, uncertainty={ukf.get_uncertainty()[3]:.4f}")
    PSI_INDEX = 2
