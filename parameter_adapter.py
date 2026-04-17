"""
Online parameter adaptation for real-time tuning of vehicle model parameters.
Adapts tire coefficients, mass, and aero parameters based on observed residuals.
"""
import numpy as np
from scipy.optimize import least_squares


class OnlineParameterAdapter:
    """
    Adapts vehicle model parameters online based on observed model-reality mismatch.
    Uses recursive least squares or gradient-based optimization.
    """
    
    def __init__(self, baseline_params, learning_rate=0.01, memory_horizon=100):
        """
        Initialize parameter adapter.
        
        Args:
            baseline_params: baseline vehicle parameters dict
            learning_rate: gradient update step size
            memory_horizon: number of recent samples to consider for adaptation
        """
        self.baseline_params = baseline_params.copy()
        self.current_params = baseline_params.copy()
        self.learning_rate = learning_rate
        self.memory_horizon = memory_horizon
        
        # Adaptive parameters (which ones we tune)
        self.adaptive_param_names = [
            'TYRE_LAT_B', 'TYRE_LAT_a2', 'TYRE_LON_B', 'TYRE_LON_a2',
            'M', 'Cd', 'Cl'
        ]
        
        # Parameter bounds (prevent unrealistic values)
        self.param_bounds = {
            'TYRE_LAT_B': (8.0, 16.0),
            'TYRE_LAT_a2': (1.5, 2.5),
            'TYRE_LON_B': (8.0, 16.0),
            'TYRE_LON_a2': (1.5, 2.5),
            'M': (baseline_params['M'] * 0.85, baseline_params['M'] * 1.15),
            'Cd': (baseline_params['Cd'] * 0.85, baseline_params['Cd'] * 1.15),
            'Cl': (baseline_params['Cl'] * 0.85, baseline_params['Cl'] * 1.15),
        }
        
        # Recursive least squares
        self.P = np.eye(len(self.adaptive_param_names)) * 100  # covariance
        self.param_vector = self._params_to_vector()
        
        # History for tracking
        self.residual_history = []
        self.param_history = []
        self.adaptation_log = []
    
    def _params_to_vector(self):
        """Convert adaptive parameters to vector."""
        vector = np.array([
            self.current_params['TYRE_LAT']['B'],
            self.current_params['TYRE_LAT']['a2'],
            self.current_params['TYRE_LON']['B'],
            self.current_params['TYRE_LON']['a2'],
            self.current_params['M'],
            self.current_params['Cd'],
            self.current_params['Cl'],
        ])
        return vector
    
    def _vector_to_params(self, vector):
        """Convert parameter vector back to dict."""
        params = self.current_params.copy()
        params['TYRE_LAT'] = params['TYRE_LAT'].copy()
        params['TYRE_LON'] = params['TYRE_LON'].copy()
        
        params['TYRE_LAT']['B'] = vector[0]
        params['TYRE_LAT']['a2'] = vector[1]
        params['TYRE_LON']['B'] = vector[2]
        params['TYRE_LON']['a2'] = vector[3]
        params['M'] = vector[4]
        params['Cd'] = vector[5]
        params['Cl'] = vector[6]
        
        return params
    
    def _apply_bounds(self, vector):
        """Apply parameter bounds."""
        for i, name in enumerate(self.adaptive_param_names):
            bounds = self.param_bounds[name]
            vector[i] = np.clip(vector[i], bounds[0], bounds[1])
        return vector
    
    def update_gradient_based(self, residual, gradient_estimate, learning_rate=None):
        """
        Update parameters using gradient-based optimization.
        
        Args:
            residual: observed residual (model error) vector
            gradient_estimate: estimated gradient of residual w.r.t. parameters
            learning_rate: override default learning rate
        """
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        # Simple gradient descent on residual norm
        residual_norm = np.linalg.norm(residual)
        
        # Update parameter vector
        self.param_vector = self.param_vector - learning_rate * gradient_estimate
        
        # Apply bounds
        self.param_vector = self._apply_bounds(self.param_vector)
        
        # Update current params
        self.current_params = self._vector_to_params(self.param_vector)
        
        # Log
        self.adaptation_log.append({
            'residual_norm': residual_norm,
            'params': self.param_vector.copy(),
            'method': 'gradient_based'
        })
    
    def update_rls(self, slip_angles, slip_ratios, Fz_kN_wheels, lateral_force_error,
                longitudinal_force_error, adaptive_factor=0.99):
        """
        Correct RLS update. phi is the Pacejka sensitivity (dF/dB, dF/da2) at current
        operating point — not raw state values.

        Args:
            slip_angles:           array [alpha_fl, alpha_fr, alpha_rl, alpha_rr] in degrees
            slip_ratios:           array [sigma_rl, sigma_rr] (longitudinal, rear only)
            Fz_kN_wheels:          array [Fz_fl, Fz_fr, Fz_rl, Fz_rr] in kN
            lateral_force_error:   scalar, sum of lateral force residuals (N)
            longitudinal_force_error: scalar, sum of longitudinal force residuals (N)
            adaptive_factor:       forgetting factor lambda
        """
        phi = self._compute_pacejka_regressor(
            slip_angles, slip_ratios, Fz_kN_wheels
        )

        # scalar measurement: total force error projected onto parameter space
        y = np.array([lateral_force_error, longitudinal_force_error,
                    lateral_force_error, longitudinal_force_error,
                    0.0, 0.0, 0.0])  # mass/aero not updated from force error directly

        # Standard RLS scalar update per parameter (decoupled for stability)
        for i in range(len(self.adaptive_param_names)):
            phi_i = phi[i]
            if abs(phi_i) < 1e-6:
                continue  # skip if parameter has no influence at this operating point

            k_i = self.P[i, i] * phi_i / (adaptive_factor + phi_i * self.P[i, i] * phi_i)
            error_i = y[i] - phi_i * self.param_vector[i]
            self.param_vector[i] = self.param_vector[i] + k_i * error_i
            self.P[i, i] = (1.0 / adaptive_factor) * (1 - k_i * phi_i) * self.P[i, i]
            self.P[i, i] = np.clip(self.P[i, i], 1e-6, 1e4)  # prevent covariance explosion

        self.param_vector = self._apply_bounds(self.param_vector)
        self.current_params = self._vector_to_params(self.param_vector)
        self.residual_history.append(y.copy())
        self.param_history.append(self.param_vector.copy())


    def _compute_pacejka_regressor(self, slip_angles, slip_ratios, Fz_kN_wheels):
        """
        Compute dF/dtheta for each adaptive parameter at the current operating point.
        This is the sensitivity of the tyre force output to each parameter — the correct phi.

        For Pacejka: F = D * sin(C * atan(B*s - E*(B*s - atan(B*s))))
        where D = a1*Fz + a2

        dF/dB  = D * C * cos(C*atan(arg)) * s*(1-E) / (1 + arg^2)
        dF/da2 = sin(C * atan(arg))   [since D = a1*Fz + a2, dD/da2 = 1]
        """
        B_lat  = self.current_params['TYRE_LAT']['B']
        C_lat  = self.current_params['TYRE_LAT'].get('C', 1.9)
        E_lat  = self.current_params['TYRE_LAT'].get('E', -1.5)
        a1_lat = self.current_params['TYRE_LAT'].get('a1', -0.10)
        a2_lat = self.current_params['TYRE_LAT']['a2']

        B_lon  = self.current_params['TYRE_LON']['B']
        C_lon  = self.current_params['TYRE_LON'].get('C', 1.65)
        E_lon  = self.current_params['TYRE_LON'].get('E', -1.5)
        a1_lon = self.current_params['TYRE_LON'].get('a1', -0.10)
        a2_lon = self.current_params['TYRE_LON']['a2']

        # Average over all four corners for lateral sensitivity
        dF_dB_lat_total  = 0.0
        dF_da2_lat_total = 0.0
        for i, alpha in enumerate(slip_angles):
            Fz = Fz_kN_wheels[i]
            D = a1_lat * Fz + a2_lat
            s = alpha  # slip angle in degrees
            arg = B_lat * s - E_lat * (B_lat * s - np.arctan(B_lat * s))
            darg_dB = s * (1 - E_lat) + E_lat * s / (1 + (B_lat * s) ** 2)
            cos_term = np.cos(C_lat * np.arctan(arg))
            dF_dB_lat  = D * C_lat * cos_term / (1 + arg ** 2) * darg_dB
            dF_da2_lat = np.sin(C_lat * np.arctan(arg))
            dF_dB_lat_total  += dF_dB_lat
            dF_da2_lat_total += dF_da2_lat

        # Average over rear two wheels for longitudinal sensitivity
        dF_dB_lon_total  = 0.0
        dF_da2_lon_total = 0.0
        for i, sigma in enumerate(slip_ratios):
            Fz = Fz_kN_wheels[2 + i]  # rear wheels
            D = a1_lon * Fz + a2_lon
            s = sigma * 100  # convert to % as used in twin_track
            arg = B_lon * s - E_lon * (B_lon * s - np.arctan(B_lon * s))
            darg_dB = s * (1 - E_lon) + E_lon * s / (1 + (B_lon * s) ** 2)
            cos_term = np.cos(C_lon * np.arctan(arg))
            dF_dB_lon  = D * C_lon * cos_term / (1 + arg ** 2) * darg_dB
            dF_da2_lon = np.sin(C_lon * np.arctan(arg))
            dF_dB_lon_total  += dF_dB_lon
            dF_da2_lon_total += dF_da2_lon

        # phi vector matches self.adaptive_param_names order:
        # ['TYRE_LAT_B', 'TYRE_LAT_a2', 'TYRE_LON_B', 'TYRE_LON_a2', 'M', 'Cd', 'Cl']
        phi = np.array([
            dF_dB_lat_total,
            dF_da2_lat_total,
            dF_dB_lon_total,
            dF_da2_lon_total,
            0.0,   # mass: not estimated from tyre forces here
            0.0,   # Cd: not estimated from tyre forces here
            0.0,   # Cl: not estimated from tyre forces here
        ])
        return phi
    
    def adapt_from_residuals(self, residuals, controls, dt=0.05):
        """
        Adapt parameters based on batch of residuals.
        Uses least squares to find parameters that minimize residuals.
        
        Args:
            residuals: array of residuals [N, state_dim]
            controls: array of controls [N, control_dim]
            dt: time step
        """
        # Flatten residuals and controls
        residuals_flat = residuals.flatten()
        
        # Simple least squares: minimize ||residuals||^2 by adjusting parameters
        def objective(param_vec):
            param_delta = param_vec - self._extract_vector(self.baseline_params)
            regularisation = 0.1 * np.dot(param_delta, param_delta)
            lat_err = float(np.mean(np.abs(residuals[:, 4]))) if residuals.ndim > 1 else 0.0
            lon_err = float(np.mean(np.abs(residuals[:, 3]))) if residuals.ndim > 1 else 0.0
            residual_term = (abs(lat_err) * abs(param_vec[0] - self.param_vector[0]) +
                            abs(lon_err) * abs(param_vec[2] - self.param_vector[2]))
            return residual_term + regularisation
        
        # Initial guess
        x0 = self.param_vector.copy()
        
        # Bounds for optimization
        bounds = (
            np.array([self.param_bounds[name][0] for name in self.adaptive_param_names]),
            np.array([self.param_bounds[name][1] for name in self.adaptive_param_names])
        )
        
        # Optimize (lightweight version for online use)
        result = least_squares(
            objective,
            x0,
            bounds=bounds,
            max_nfev=20  # Very limited for online use
        )
        
        # Update parameters
        self.param_vector = result.x
        self.current_params = self._vector_to_params(self.param_vector)
        
        self.adaptation_log.append({
            'residual_norm': np.linalg.norm(residuals_flat),
            'params': self.param_vector.copy(),
            'method': 'batch_optimization'
        })
    
    def get_current_params(self):
        """Return current adapted parameters."""
        return self.current_params.copy()
    
    def get_parameter_changes(self):
        """Return changes in adaptive parameters from baseline."""
        changes = {}
        base_vec = self._extract_vector(self.baseline_params)
        curr_vec = self.param_vector
        
        for i, name in enumerate(self.adaptive_param_names):
            change_pct = ((curr_vec[i] - base_vec[i]) / base_vec[i]) * 100
            changes[name] = {
                'baseline': base_vec[i],
                'current': curr_vec[i],
                'change_pct': change_pct
            }
        
        return changes
    
    def _extract_vector(self, params):
        """Extract adaptive parameter vector from params dict."""
        vector = np.array([
            params['TYRE_LAT']['B'],
            params['TYRE_LAT']['a2'],
            params['TYRE_LON']['B'],
            params['TYRE_LON']['a2'],
            params['M'],
            params['Cd'],
            params['Cl'],
        ])
        return vector
    
    def reset_to_baseline(self):
        """Reset all parameters back to baseline."""
        self.current_params = self.baseline_params.copy()
        self.param_vector = self._extract_vector(self.baseline_params)
        self.P = np.eye(len(self.adaptive_param_names)) * 100
        self.residual_history = []
        self.param_history = []
        self.adaptation_log = []


class TireParameterEstimator:
    """
    Specialized estimator for tire parameters using force measurements.
    """
    
    def __init__(self, baseline_tire_params):
        """
        Initialize tire parameter estimator.
        
        Args:
            baseline_tire_params: baseline tire parameters (TYRE_LAT/TYRE_LON dicts)
        """
        self.baseline_params = baseline_tire_params.copy()
        self.current_params = baseline_tire_params.copy()
        
        # Tire slip-force measurements for fitting
        self.measurements = []
        self.measured_forces = []
        self.param_bounds = {
            'TYRE_LAT_B':  (10.0, 14.0),                              # was (8, 16)
            'TYRE_LAT_a2': (1.8, 2.3),                                # was (1.5, 2.5)
            'TYRE_LON_B':  (10.0, 14.0),
            'TYRE_LON_a2': (1.8, 2.3),
            'M':  (baseline_params['M'] * 0.97, baseline_params['M'] * 1.03),   # ±3% not ±15%
            'Cd': (baseline_params['Cd'] * 0.97, baseline_params['Cd'] * 1.03),
            'Cl': (baseline_params['Cl'] * 0.97, baseline_params['Cl'] * 1.03),
        }

    
    def add_measurement(self, slip, normal_force, measured_force):
        """
        Add a tire force measurement.
        
        Args:
            slip: slip angle (deg) or slip ratio (%)
            normal_force: normal force (kN)
            measured_force: measured tire force (N)
        """
        self.measurements.append((slip, normal_force))
        self.measured_forces.append(measured_force)
    
    def estimate_parameters(self, tire_type='lateral'):
        """
        Estimate tire parameters from accumulated measurements.
        
        Args:
            tire_type: 'lateral' or 'longitudinal'
            
        Returns:
            estimated tire parameters dict
        """
        if len(self.measurements) < 3:
            return self.current_params
        
        # Convert measurements to arrays
        slips = np.array([m[0] for m in self.measurements])
        normal_forces = np.array([m[1] for m in self.measurements])
        forces = np.array(self.measured_forces)
        
        # Simple linear regression to estimate stiffness
        # Assuming small slip: F ≈ C_s * slip (linear region)
        mask = np.abs(slips) < 5  # small slip region
        
        if np.sum(mask) > 2:
            slips_linear = slips[mask]
            forces_linear = forces[mask]
            
            # Estimate cornering stiffness
            # F = C_s * alpha, where C_s = BCD_a3 * exp(...)
            stiffness = np.polyfit(slips_linear, forces_linear, 1)[0]
            
            # Update B parameter (stiffness factor)
            # This is a simplified adaptation
            base_stiffness = self.baseline_params['B']
            self.current_params['B'] = base_stiffness * (stiffness / 100.0)
        
        return self.current_params.copy()
    
    def reset(self):
        """Reset measurements and parameters."""
        self.measurements = []
        self.measured_forces = []
        self.current_params = self.baseline_params.copy()


if __name__ == "__main__":
    print("Online Parameter Adaptation Test")
    print("=" * 50)
    
    # Create baseline params
    baseline_params = {
        'M': 752,
        'Cd': 0.8,
        'Cl': 3.5,
        'TYRE_LAT': {'B': 12.0, 'a2': 2.05},
        'TYRE_LON': {'B': 12.0, 'a2': 2.1},
    }
    
    # Create adapter
    adapter = OnlineParameterAdapter(baseline_params)
    print("Adapter initialized")
    
    # Test parameter changes
    print("\nBaseline parameters:")
    print(f"  Mass: {adapter.baseline_params['M']} kg")
    print(f"  Tire LAT B: {adapter.baseline_params['TYRE_LAT']['B']}")
    
    # Simulate some residuals and adapt
    print("\nSimulating adaptation...")
    for i in range(10):
        slip_angles  = np.random.uniform(-5, 5, 4)       # degrees
        slip_ratios  = np.random.uniform(-0.1, 0.1, 2)   # dimensionless
        Fz_kN        = np.array([2.0, 2.0, 2.2, 2.2])    # kN per wheel
        lat_err      = np.random.randn() * 50             # Newtons
        lon_err      = np.random.randn() * 50             # Newtons

        adapter.update_rls(slip_angles, slip_ratios, Fz_kN, lat_err, lon_err,
                           adaptive_factor=0.995)
    
    print("\nAdapted parameters:")
    changes = adapter.get_parameter_changes()
    for param_name, change_info in changes.items():
        print(f"  {param_name}: {change_info['current']:.4f} ({change_info['change_pct']:+.2f}%)")
    
    # Test tire estimator
    print("\nTire Parameter Estimator Test")
    tire_params = {'B': 12.0, 'C': 1.9, 'D': 1.8}
    tire_est = TireParameterEstimator(tire_params)
    
    # Add some measurements
    for slip in range(-10, 11, 2):
        tire_est.add_measurement(slip, 2.0, slip * 100)  # simplified force model
    
    est_params = tire_est.estimate_parameters()
    print(f"Estimated tire B: {est_params['B']:.4f}")
