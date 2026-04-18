"""
Online parameter adaptation for real-time tuning of vehicle model parameters.
Adapts tire coefficients, mass, and aero parameters based on observed residuals.
"""
import numpy as np
from scipy.optimize import least_squares


class OnlineParameterAdapter:
    """
    Adapts vehicle model parameters online based on observed model-reality mismatch.
    Uses recursive least squares with a proper Pacejka-based regressor.
    """

    def __init__(self, baseline_params, learning_rate=0.01, memory_horizon=100):
        self.baseline_params = baseline_params.copy()
        self.current_params  = baseline_params.copy()
        self.learning_rate   = learning_rate
        self.memory_horizon  = memory_horizon

        self.adaptive_param_names = [
            'TYRE_LAT_B', 'TYRE_LAT_a2', 'TYRE_LON_B', 'TYRE_LON_a2',
            'M', 'Cd', 'Cl'
        ]

        self.param_bounds = {
            'TYRE_LAT_B':  (5.0,  21.0),
            'TYRE_LAT_a2': (0.5,  5.0),
            'TYRE_LON_B':  (5.0,  21.0),
            'TYRE_LON_a2': (1.5,  2.5),
            'M':  (baseline_params['M']  * 0.85, baseline_params['M']  * 1.15),
            'Cd': (baseline_params['Cd'] * 0.85, baseline_params['Cd'] * 1.15),
            'Cl': (baseline_params['Cl'] * 0.85, baseline_params['Cl'] * 1.15),
        }

        # RLS state: param_vector stores DEVIATIONS (dtheta = theta_true - theta_est), starting at zero
        self.P = np.eye(len(self.adaptive_param_names)) * 0.1  # Very small initial covariance for stability
        self.param_vector = np.zeros(len(self.adaptive_param_names))  # Start with no deviation

        self.residual_history = []
        self.param_history    = []
        self.adaptation_log   = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _params_to_vector(self):
        return np.array([
            self.current_params['TYRE_LAT']['B'],
            self.current_params['TYRE_LAT']['a2'],
            self.current_params['TYRE_LON']['B'],
            self.current_params['TYRE_LON']['a2'],
            self.current_params['M'],
            self.current_params['Cd'],
            self.current_params['Cl'],
        ])

    def _vector_to_params(self, vector):
        params = self.current_params.copy()
        params['TYRE_LAT'] = params['TYRE_LAT'].copy()
        params['TYRE_LON'] = params['TYRE_LON'].copy()
        params['TYRE_LAT']['B']  = vector[0]
        params['TYRE_LAT']['a2'] = vector[1]
        params['TYRE_LON']['B']  = vector[2]
        params['TYRE_LON']['a2'] = vector[3]
        params['M']  = vector[4]
        params['Cd'] = vector[5]
        params['Cl'] = vector[6]
        return params

    def _extract_vector(self, params):
        return np.array([
            params['TYRE_LAT']['B'],
            params['TYRE_LAT']['a2'],
            params['TYRE_LON']['B'],
            params['TYRE_LON']['a2'],
            params['M'],
            params['Cd'],
            params['Cl'],
        ])

    def _apply_bounds(self, vector):
        for i, name in enumerate(self.adaptive_param_names):
            lo, hi = self.param_bounds[name]
            vector[i] = np.clip(vector[i], lo, hi)
        return vector

    # ------------------------------------------------------------------
    # FIX 1: Pacejka sensitivity regressor
    # ------------------------------------------------------------------

    def _compute_pacejka_regressor(self, slip_angles, slip_ratios, Fz_kN_wheels):
        """
        Compute dF/dtheta at the current operating point.

        For Pacejka:  F = D * sin(C * atan(arg))
                      D = a1*Fz + a2
                      arg = B*s - E*(B*s - atan(B*s))

        dF/dB  = D * C * cos(C*atan(arg)) / (1+arg^2) * darg_dB
        dF/da2 = sin(C * atan(arg))     [since dD/da2 = 1]
        """
        B_lat  = self.current_params['TYRE_LAT']['B']
        C_lat  = self.current_params['TYRE_LAT'].get('C',  1.9)
        E_lat  = self.current_params['TYRE_LAT'].get('E', -1.5)
        a1_lat = self.current_params['TYRE_LAT'].get('a1', -0.10)
        a2_lat = self.current_params['TYRE_LAT']['a2']

        B_lon  = self.current_params['TYRE_LON']['B']
        C_lon  = self.current_params['TYRE_LON'].get('C',  1.65)
        E_lon  = self.current_params['TYRE_LON'].get('E', -1.5)
        a1_lon = self.current_params['TYRE_LON'].get('a1', -0.10)
        a2_lon = self.current_params['TYRE_LON']['a2']

        # --- lateral: average sensitivity over all four corners ---
        dF_dB_lat  = 0.0
        dF_da2_lat = 0.0
        for i, alpha in enumerate(slip_angles):
            Fz = Fz_kN_wheels[i]
            D  = a1_lat * Fz + a2_lat
            s  = float(alpha)
            arg      = B_lat * s - E_lat * (B_lat * s - np.arctan(B_lat * s))
            darg_dB  = s * (1 - E_lat) + E_lat * s / (1 + (B_lat * s) ** 2)
            cos_term = np.cos(C_lat * np.arctan(arg))
            dF_dB_lat  += D * C_lat * cos_term / (1 + arg ** 2) * darg_dB
            dF_da2_lat += np.sin(C_lat * np.arctan(arg))

        # --- longitudinal: average sensitivity over rear two wheels ---
        dF_dB_lon  = 0.0
        dF_da2_lon = 0.0
        for i, sigma in enumerate(slip_ratios):
            Fz = Fz_kN_wheels[2 + i]   # rear wheels
            D  = a1_lon * Fz + a2_lon
            s  = float(sigma)      # convert to % as used in twin_track
            arg      = B_lon * s - E_lon * (B_lon * s - np.arctan(B_lon * s))
            darg_dB  = s * (1 - E_lon) + E_lon * s / (1 + (B_lon * s) ** 2)
            cos_term = np.cos(C_lon * np.arctan(arg))
            dF_dB_lon  += D * C_lon * cos_term / (1 + arg ** 2) * darg_dB
            dF_da2_lon += np.sin(C_lon * np.arctan(arg))

        # phi order matches adaptive_param_names:
        # [TYRE_LAT_B, TYRE_LAT_a2, TYRE_LON_B, TYRE_LON_a2, M, Cd, Cl]
        phi = np.array([
            dF_dB_lat,
            dF_da2_lat,
            dF_dB_lon,
            dF_da2_lon,
            0.0,   # mass not estimated from tyre forces
            0.0,   # Cd  not estimated from tyre forces
            0.0,   # Cl  not estimated from tyre forces
        ])

        return phi

    # ------------------------------------------------------------------
    # FIX 1+2: corrected update_rls signature and per-parameter scalar update
    # ------------------------------------------------------------------

    def update_rls(self, slip_angles, slip_ratios, Fz_kN_wheels,
                   lateral_force_error, longitudinal_force_error,
                   acceleration_error=0.0, speed_error=0.0,
                   adaptive_factor=0.98, debug=False):
        """
        Recursive least squares update - simple scalar RLS per parameter.
        
        CORRECTED FORMULATION:
        We want to estimate dtheta = theta_true - theta_est from force errors.
        The relationship is: force_error ≈ phi * dtheta
        So we estimate dtheta, then set theta_new = theta_old + dtheta_change
        
        Args:
            slip_angles:              array [alpha_fl, alpha_fr, alpha_rl, alpha_rr] degrees
            slip_ratios:              array [sigma_rl, sigma_rr] dimensionless
            Fz_kN_wheels:             array [Fz_fl, Fz_fr, Fz_rl, Fz_rr] kN
            lateral_force_error:      scalar lateral force residual (N): true_force - predicted_force
            longitudinal_force_error: scalar longitudinal force residual (N)
            acceleration_error:       scalar acceleration error (m/s^2)
            speed_error:              scalar speed error (m/s)
            adaptive_factor:          forgetting factor (0.95-0.99)
            debug:                    print debug info
        """
        # Compute sensitivities
        phi = self._compute_pacejka_regressor(slip_angles, slip_ratios, Fz_kN_wheels)
        
        # Build measurement vector = force errors (in Newtons, not divided)
        # Keep signals in their natural units for better RLS performance
        y = np.array([
            lateral_force_error,                # [0] lat force error (N)
            lateral_force_error,                # [1] lat force error (N)
            longitudinal_force_error,           # [2] lon force error (N)
            longitudinal_force_error,           # [3] lon force error (N)
            acceleration_error,                 # [4] acceleration error (m/s^2)
            speed_error,                        # [5] speed error (m/s)
            speed_error * 0.5,                  # [6] speed error (m/s)
        ])
        
        if debug:
            print(f"[RLS] phi = {phi}")
            print(f"[RLS] y = {y}")
            print(f"[RLS] param_vector = {self.param_vector}")
        
        # Scalar RLS update for each parameter independently
        # Model: force_error = phi_i * dtheta_i + noise
        # where dtheta_i is the parameter error (true - estimated)
        for i in range(len(self.adaptive_param_names)):
            phi_i = phi[i]
            y_i   = y[i]
            
            # Skip if sensitivity is too small (except for M, Cd, Cl)
            if abs(phi_i) < 1e-8:
                if i >= 4:  # M, Cd, Cl have near-zero sensitivities from tire forces
                    phi_i = 1.0 if y_i != 0 else 0.0
                else:
                    continue
            
            # Estimate the parameter error: dtheta_est
            # We maintain dtheta_est in param_vector temporarily
            # The interpretation is: param_vector[i] represents dtheta_est
            p_i = self.P[i, i]
            
            # Gain: k = p / (λ + φ^2*p)
            denom = adaptive_factor + phi_i ** 2 * p_i
            k_i = p_i * phi_i / (denom + 1e-12)
            
            # Error: e = y - phi*dtheta_est
            error_i = y_i - phi_i * self.param_vector[i]
            
            # Update dtheta_est: dtheta += k*e
            theta_before = self.param_vector[i]
            self.param_vector[i] += k_i * error_i
            
            if debug and i == 1:
                print(f"[RLS] i=1 (TYRE_LAT_a2):")
                print(f"  phi_i={phi_i:.6f}, y_i={y_i:.6f}, dtheta_est={theta_before:.6f}")
                print(f"  p_i={p_i:.6f}, k_i={k_i:.6f}, error_i={error_i:.6f}")
                print(f"  dtheta after update: {self.param_vector[i]:.6f}")
            
            # Update covariance with light refresh for stability
            p_new = (1.0 / adaptive_factor) * (1.0 - k_i * phi_i) * p_i
            # Very conservative refresh to minimize oscillations
            # Only allow gradual re-exploration after convergence
            if i < 4:
                refresh = 0.005  # Very conservative for tire parameters
            else:
                refresh = 0.01  # Conservative for M, Cd, Cl
            self.P[i, i] = np.clip(p_new + refresh, 0.05, 20.0)
        
        # Apply bounds to parameter DEVIATIONS to prevent runaway
        # The param_vector now represents dtheta (parameter deviations)
        # We want to keep dtheta reasonable: ±20% of baseline
        for i in range(len(self.adaptive_param_names)):
            name = self.adaptive_param_names[i]
            baseline_val = self._extract_vector(self.baseline_params)[i]
            max_dev = baseline_val * 0.2  # Allow ±20% deviation
            self.param_vector[i] = np.clip(self.param_vector[i], -max_dev, max_dev)
        
        # Reconstruct actual parameters: theta_est = theta_baseline + dtheta_est
        baseline_vector = self._extract_vector(self.baseline_params)
        actual_params_vector = baseline_vector + self.param_vector
        
        # Also apply absolute bounds
        actual_params_vector = self._apply_bounds(actual_params_vector)
        
        # Convert back to params dict
        self.current_params = self._vector_to_params(actual_params_vector)
        
        # Store history (store the actual params, not deltas)
        self.param_history.append(actual_params_vector.copy())

    # ------------------------------------------------------------------
    # Gradient-based update (unchanged)
    # ------------------------------------------------------------------

    def update_gradient_based(self, residual, gradient_estimate, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
        residual_norm = np.linalg.norm(residual)
        self.param_vector = self.param_vector - learning_rate * gradient_estimate
        self.param_vector = self._apply_bounds(self.param_vector)
        self.current_params = self._vector_to_params(self.param_vector)
        self.adaptation_log.append({
            'residual_norm': residual_norm,
            'params': self.param_vector.copy(),
            'method': 'gradient_based'
        })

    # ------------------------------------------------------------------
    # FIX 3: batch adaptation — objective now depends on param_vec
    # ------------------------------------------------------------------

    def adapt_from_residuals(self, residuals, controls, dt=0.05):
        """
        Batch adaptation using a linear sensitivity model.
        objective() now actually depends on param_vec.
        """
        if len(residuals) == 0:
            return

        residuals_flat = residuals.flatten()
        residuals = np.atleast_2d(residuals)

        lat_err = float(np.mean(np.abs(residuals[:, 4]))) if residuals.shape[1] > 4 else 0.0
        lon_err = float(np.mean(np.abs(residuals[:, 3]))) if residuals.shape[1] > 3 else 0.0

        def objective(param_vec):
            param_delta   = param_vec - self._extract_vector(self.baseline_params)
            regularisation = 0.1 * np.dot(param_delta, param_delta)
            residual_term  = (abs(lat_err) * abs(param_vec[0] - self.param_vector[0]) +
                              abs(lon_err) * abs(param_vec[2] - self.param_vector[2]))
            return residual_term + regularisation

        x0 = self.param_vector.copy()
        bounds = (
            np.array([self.param_bounds[n][0] for n in self.adaptive_param_names]),
            np.array([self.param_bounds[n][1] for n in self.adaptive_param_names])
        )
        result = least_squares(objective, x0, bounds=bounds, max_nfev=20)

        self.param_vector   = result.x
        self.current_params = self._vector_to_params(self.param_vector)
        self.adaptation_log.append({
            'residual_norm': np.linalg.norm(residuals_flat),
            'params': self.param_vector.copy(),
            'method': 'batch_optimization'
        })

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_current_params(self):
        return self.current_params.copy()

    def get_parameter_changes(self):
        changes  = {}
        base_vec = self._extract_vector(self.baseline_params)
        curr_vec = self.param_vector
        for i, name in enumerate(self.adaptive_param_names):
            change_pct = ((curr_vec[i] - base_vec[i]) / base_vec[i]) * 100
            changes[name] = {
                'baseline':   base_vec[i],
                'current':    curr_vec[i],
                'change_pct': change_pct
            }
        return changes

    def reset_to_baseline(self):
        self.current_params   = self.baseline_params.copy()
        self.param_vector     = np.zeros(len(self.adaptive_param_names))  # Reset to no deviation
        self.P                = np.eye(len(self.adaptive_param_names)) * 0.1
        self.residual_history = []
        self.param_history    = []
        self.adaptation_log   = []


# -------------------------------------------------------------------------
class TireParameterEstimator:
    """Specialized estimator for tire parameters using force measurements."""

    def __init__(self, baseline_tire_params):
        self.baseline_params = baseline_tire_params.copy()
        self.current_params  = baseline_tire_params.copy()
        self.measurements    = []
        self.measured_forces = []

    def add_measurement(self, slip, normal_force, measured_force):
        self.measurements.append((slip, normal_force))
        self.measured_forces.append(measured_force)

    def estimate_parameters(self, tire_type='lateral'):
        if len(self.measurements) < 3:
            return self.current_params
        slips  = np.array([m[0] for m in self.measurements])
        forces = np.array(self.measured_forces)
        mask   = np.abs(slips) < 5
        if np.sum(mask) > 2:
            stiffness = np.polyfit(slips[mask], forces[mask], 1)[0]
            self.current_params['B'] = self.baseline_params['B'] * (stiffness / 100.0)
        return self.current_params.copy()

    def reset(self):
        self.measurements    = []
        self.measured_forces = []
        self.current_params  = self.baseline_params.copy()


# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("Online Parameter Adaptation Test")
    print("=" * 50)

    baseline_params = {
        'M': 752, 'Cd': 0.8, 'Cl': 3.5,
        'TYRE_LAT': {'B': 12.0, 'a2': 2.05},
        'TYRE_LON': {'B': 12.0, 'a2': 2.1},
    }

    adapter = OnlineParameterAdapter(baseline_params)
    print("Adapter initialized")
    print(f"\nBaseline — Mass: {adapter.baseline_params['M']} kg, "
          f"Tire LAT B: {adapter.baseline_params['TYRE_LAT']['B']}")

    print("\nSimulating adaptation...")
    for i in range(10):
        slip_angles = np.random.uniform(-5,   5,   4)
        slip_ratios = np.random.uniform(-0.1, 0.1, 2)
        Fz_kN       = np.array([2.0, 2.0, 2.2, 2.2])
        lat_err     = np.random.randn() * 50
        lon_err     = np.random.randn() * 50
        adapter.update_rls(slip_angles, slip_ratios, Fz_kN, lat_err, lon_err,
                           adaptive_factor=0.995)

    print("\nAdapted parameters:")
    for name, info in adapter.get_parameter_changes().items():
        print(f"  {name}: {info['current']:.4f} ({info['change_pct']:+.2f}%)")

    print("\nTire Parameter Estimator Test")
    tire_est = TireParameterEstimator({'B': 12.0, 'C': 1.9, 'D': 1.8})
    for slip in range(-10, 11, 2):
        tire_est.add_measurement(slip, 2.0, slip * 100)
    print(f"Estimated tire B: {tire_est.estimate_parameters()['B']:.4f}")
