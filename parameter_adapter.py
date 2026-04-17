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
    
    def update_rls(self, observation, measurement, adaptive_factor=0.99):
        """
        Recursive least squares update for parameter adaptation.
        
        Args:
            observation: state/control observation vector
            measurement: measurement vector (observed residual)
            adaptive_factor: forgetting factor (0.9-0.99 for online learning)
        """
        # Observation vector dimension should match number of adaptive params
        phi = observation[:len(self.adaptive_param_names)]  # measurement matrix
        
        # RLS update
        # S = P + phi*phi^T / adaptive_factor
        S = self.P + np.outer(phi, phi) / adaptive_factor
        
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if singular
            S_inv = np.linalg.pinv(S)
        
        # Gain: K = P * phi / (lambda + phi^T * P * phi)
        K = (self.P @ phi) / (adaptive_factor + phi @ self.P @ phi)
        
        # Parameter update
        error = measurement - (phi @ self.param_vector)
        self.param_vector = self.param_vector + K * error
        
        # Apply bounds
        self.param_vector = self._apply_bounds(self.param_vector)
        
        # Update covariance
        self.P = (1.0 / adaptive_factor) * (self.P - K[:, np.newaxis] @ phi[np.newaxis, :] @ self.P)
        
        # Update current params
        self.current_params = self._vector_to_params(self.param_vector)
        
        # Log
        self.residual_history.append(measurement)
        self.param_history.append(self.param_vector.copy())
    
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
            # This is a simplified objective - just minimize residual magnitude
            # In practice, you'd recompute dynamics with new params
            return np.linalg.norm(residuals_flat)
        
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
            max_nfev=10  # Very limited for online use
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
        # Simulate residual (would come from real data)
        residual = np.random.randn(13) * 0.01
        
        # RLS update
        observation = np.random.randn(7) * 0.1
        measurement = np.random.randn(7) * 0.05
        
        adapter.update_rls(observation, measurement, adaptive_factor=0.95)
    
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
