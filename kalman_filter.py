"""
Extended Kalman Filter (EKF) for vehicle state estimation.
Estimates vehicle state from noisy sensor measurements.
"""
import numpy as np
from scipy import linalg
from twin_track import twin_track_model


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for vehicle dynamics state estimation.
    
    State vector (13 elements):
        [x, y, psi, vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr, engine_rpm, gear, throttle]
    
    Sensor measurements (9 elements):
        [x_gps, y_gps, ax, ay, r, vx_gps, vy_gps, engine_rpm, wheel_speeds (avg)]
    """
    
    def __init__(self, params, process_noise=None, measurement_noise=None):
        """
        Initialize EKF for vehicle state estimation.
        
        Args:
            params: vehicle parameters dict
            process_noise: process noise covariance (Q matrix)
            measurement_noise: measurement noise covariance (R matrix)
        """
        self.params = params
        self.n_states = 13
        self.n_measurements = 9
        
        # Initial state estimate
        self.x = np.zeros(self.n_states)
        self.x[3] = 5.0  # initial vx = 5 m/s
        
        # State covariance (estimate uncertainty)
        self.P = np.eye(self.n_states) * 1.0
        
        # Process noise covariance (Q) - uncertainty in dynamics model
        if process_noise is None:
            self.Q = np.diag([
                0.01, 0.01,           # position (x, y)
                0.01,                 # yaw angle
                0.1, 0.1, 0.01,       # velocities (vx, vy, r)
                0.5, 0.5, 0.5, 0.5,   # wheel speeds
                10.0, 0.0, 0.01       # engine_rpm, gear, throttle
            ])
        else:
            self.Q = process_noise
        
        # Measurement noise covariance (R) - sensor noise
        if measurement_noise is None:
            self.R = np.diag([
                1.0, 1.0,             # GPS: x, y position (m)
                0.5, 0.5,             # IMU: ax, ay (m/s²)
                0.01,                 # IMU: yaw rate (rad/s)
                0.1, 0.1,             # GPS velocity (m/s)
                50.0,                 # RPM sensor
                0.2                   # wheel speeds average (m/s)
            ])
        else:
            self.R = measurement_noise
        
        self.dt = 0.05  # time step
        self.history = []
    
    def predict(self, u, dt=None):
        """
        Prediction step: propagate state estimate using dynamics model.
        
        Args:
            u: control input [steering, throttle, brake]
            dt: time step (uses self.dt if None)
        """
        if dt is None:
            dt = self.dt
        
        # Propagate mean estimate through nonlinear dynamics
        self.x = twin_track_model(self.x, u, dt, self.params)
        
        # Safety clamps
        self.x[0:2] = np.clip(self.x[0:2], -1e4, 1e4)
        self.x[2] = np.remainder(self.x[2], 2*np.pi)
        self.x[3:6] = np.clip(self.x[3:6], -50, 50)
        self.x[6:10] = np.clip(self.x[6:10], 0, 100)
        self.x[10] = np.clip(self.x[10], 1000, 15500)
        self.x[11] = int(self.x[11])
        self.x[12] = np.clip(self.x[12], 0, 1)
        
        # Linearize dynamics around current state to compute Jacobian F
        F = self._jacobian_dynamics(self.x, u, dt)
        
        # Propagate covariance: P = F*P*F^T + Q
        self.P = F @ self.P @ F.T + self.Q
        
        # Ensure P remains symmetric and positive definite
        self.P = 0.5 * (self.P + self.P.T)
    
    def update(self, z):
        """
        Update step: incorporate measurement into state estimate.
        
        Args:
            z: measurement vector [ax, ay, r, vx_gps, vy_gps, engine_rpm, wheel_speed_avg]
        """
        # Measurement function: which states we measure
        # z = h(x) = [vx, vy, r, vx, vy, rpm, (vw_fl + vw_fr + vw_rl + vw_rr)/4]
        
        # Compute innovation (measurement residual)
        z_pred = self._measurement_function(self.x)
        y = z - z_pred  # innovation
        
        # Compute measurement Jacobian H
        H = self._jacobian_measurement(self.x)
        
        # Innovation covariance: S = H*P*H^T + R
        S = H @ self.P @ H.T + self.R
        S = 0.5 * (S + S.T)  # ensure symmetry
        
        # Kalman gain: K = P*H^T * inv(S)
        try:
            K = self.P @ H.T @ linalg.inv(S)
        except linalg.LinAlgError:
            # If S is singular, use pseudoinverse
            K = self.P @ H.T @ np.linalg.pinv(S)
        
        # Update state estimate: x = x + K*y
        self.x = self.x + K @ y
        
        # Update covariance: P = (I - K*H)*P
        I = np.eye(self.n_states)
        self.P = (I - K @ H) @ self.P
        
        # Ensure P remains symmetric and positive definite
        self.P = 0.5 * (self.P + self.P.T)
        
        # Clamp state values after update
        self.x[0:2] = np.clip(self.x[0:2], -1e4, 1e4)
        self.x[2] = np.remainder(self.x[2], 2*np.pi)
        self.x[3:6] = np.clip(self.x[3:6], -50, 50)
        self.x[6:10] = np.clip(self.x[6:10], 0, 100)
        self.x[10] = np.clip(self.x[10], 1000, 15500)
        self.x[12] = np.clip(self.x[12], 0, 1)
    
    def _jacobian_dynamics(self, x, u, dt, eps=1e-5):
        """
        Compute Jacobian matrix of dynamics with respect to state.
        Uses numerical differentiation.
        
        Args:
            x: state vector
            u: control input
            dt: time step
            eps: numerical differentiation step
            
        Returns:
            F matrix (n_states × n_states)
        """
        F = np.zeros((self.n_states, self.n_states))
        
        for i in range(self.n_states):
            # Perturb state
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            
            # Compute dynamics at perturbed points
            try:
                x_dot_plus = twin_track_model(x_plus, u, dt, self.params)
                x_dot_minus = twin_track_model(x_minus, u, dt, self.params)
                
                # Central difference
                F[:, i] = (x_dot_plus - x_dot_minus) / (2 * eps)
            except:
                # If computation fails, use identity (no coupling)
                F[i, i] = 1.0
        
        return F
    
    def _measurement_function(self, x):
        """
        Measurement model: what we observe from the state.
        
        Args:
            x: state vector
            
        Returns:
            z_pred: predicted measurement
        """
        # From state, extract measurements
        # State: [x, y, psi, vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr, rpm, gear, throttle]
        
        # Compute accelerations from velocities (simple approximation)
        # In real EKF, these would come from IMU
        ax = 0.0  # would need previous velocity
        ay = 0.0
        
        z_pred = np.array([
            x[0],                             # x position (GPS) (0)
            x[1],                             # y position (GPS) (1)
            ax,                               # ax (2)
            ay,                               # ay (3)
            x[5],                             # r (yaw rate) (4)
            x[3],                             # vx (5)
            x[4],                             # vy (6)
            x[10],                            # engine_rpm (7)
            (x[6] + x[7] + x[8] + x[9]) / 4  # average wheel speed (8)
        ])
        
        return z_pred
    
    def _jacobian_measurement(self, x):
        """
        Compute Jacobian of measurement function.
        
        Returns:
            H matrix (n_measurements × n_states)
        """
        H = np.zeros((self.n_measurements, self.n_states))
        
        # x position: d(x)/dx[0] = 1
        H[0, 0] = 1.0
        
        # y position: d(y)/dx[1] = 1
        H[1, 1] = 1.0
        
        # ax: no direct dependency on state (external input)
        # ay: no direct dependency on state
        
        # r: d(r)/dx[5] = 1
        H[4, 5] = 1.0
        
        # vx: d(vx)/dx[3] = 1
        H[5, 3] = 1.0
        
        # vy: d(vy)/dx[4] = 1
        H[6, 4] = 1.0
        
        # rpm: d(rpm)/dx[10] = 1
        H[7, 10] = 1.0
        
        # avg wheel speed: d(avg_ws)/dx[6:10] = 0.25
        H[8, 6:10] = 0.25
        
        return H
    
    def get_state(self):
        """Return current state estimate."""
        return self.x.copy()
    
    def get_covariance(self):
        """Return current covariance estimate."""
        return self.P.copy()
    
    def get_uncertainty(self):
        """Return standard deviation of state estimates."""
        return np.sqrt(np.diag(self.P))
    
    def reset(self, initial_state=None):
        """Reset filter to initial state."""
        if initial_state is not None:
            self.x = initial_state.copy()
        else:
            self.x = np.zeros(self.n_states)
            self.x[3] = 5.0
        
        self.P = np.eye(self.n_states) * 1.0


def add_sensor_noise(state, measurement_noise_std):
    """
    Add realistic sensor noise to state measurements.
    
    Args:
        state: true state vector
        measurement_noise_std: dict with noise standard deviations
        
    Returns:
        noisy measurement vector
    """
    # GPS position with noise (typically 1-5 meter accuracy)
    x_noise = np.random.normal(0, measurement_noise_std.get('x_gps', 1.0))
    y_noise = np.random.normal(0, measurement_noise_std.get('y_gps', 1.0))
    
    # Simulate IMU measurements with noise
    ax_noise = np.random.normal(0, measurement_noise_std.get('ax', 0.5))
    ay_noise = np.random.normal(0, measurement_noise_std.get('ay', 0.5))
    r_noise = np.random.normal(0, measurement_noise_std.get('r', 0.01))
    
    # GPS velocity with noise
    vx_noise = np.random.normal(0, measurement_noise_std.get('vx', 0.1))
    vy_noise = np.random.normal(0, measurement_noise_std.get('vy', 0.1))
    
    # RPM sensor noise
    rpm_noise = np.random.normal(0, measurement_noise_std.get('rpm', 50))
    
    # Wheel speed sensor noise
    ws_noise = np.random.normal(0, measurement_noise_std.get('wheel_speed', 0.2))
    
    # Construct noisy measurement (9 elements: x, y, ax, ay, r, vx, vy, rpm, wheel_speed)
    z = np.array([
        state[0] + x_noise,                                          # x (GPS)
        state[1] + y_noise,                                          # y (GPS)
        ax_noise,                                                    # ax
        ay_noise,                                                    # ay
        state[5] + r_noise,                                          # r
        state[3] + vx_noise,                                         # vx
        state[4] + vy_noise,                                         # vy
        state[10] + rpm_noise,                                       # rpm
        (state[6] + state[7] + state[8] + state[9]) / 4 + ws_noise  # wheel speed avg
    ])
    
    return z


if __name__ == "__main__":
    print("Extended Kalman Filter Test")
    print("=" * 50)
    
    # Create minimal params for testing
    test_params = {
        'L': 3.6, 'TF': 1.58, 'TR': 1.42, 'H': 0.295, 'MX': 0.453, 'M': 752,
        'Cd': 0.8, 'Cl': 3.5, 'Area': 1.2, 'CX': 0.52, 'K': 1.35,
        'final_drive': 6.3, 'tyre_radius': 0.330
    }
    
    # Initialize EKF
    ekf = ExtendedKalmanFilter(test_params)
    print(f"EKF initialized: state dim={ekf.n_states}, measurement dim={ekf.n_measurements}")
    
    # Test predict and update cycle
    print("\nTesting predict-update cycle...")
    
    u = np.array([0.1, 0.5, 0.0])  # steering, throttle, brake
    initial_state = np.array([0., 0., 0., 5., 0., 0., 5., 5., 5., 5., 3000., 1., 0.1])
    ekf.x = initial_state.copy()
    
    # Predict
    ekf.predict(u)
    print(f"After predict: vx={ekf.x[3]:.3f}, uncertainty={ekf.get_uncertainty()[3]:.4f}")
    
    # Generate noisy measurement
    noise_std = {
        'ax': 0.5, 'ay': 0.5, 'r': 0.01,
        'vx': 0.1, 'vy': 0.1, 'rpm': 50, 'wheel_speed': 0.2
    }
    z = add_sensor_noise(ekf.x, noise_std)
    
    # Update
    ekf.update(z)
    print(f"After update: vx={ekf.x[3]:.3f}, uncertainty={ekf.get_uncertainty()[3]:.4f}")
    print(f"State uncertainty: {ekf.get_uncertainty()}")
