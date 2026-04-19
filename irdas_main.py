"""
In-race Data Augmentation System (IRDAS)
Integrates simulator, Kalman filter, neural network, and parameter adaptation
for online model correction and state estimation.
"""
import numpy as np
import torch
from typing import Dict, Tuple

from simulator import RealVehicleSimulator
from kalman_filter import ExtendedKalmanFilter, add_sensor_noise
from residual_network import ResidualDynamicsLearner
from parameter_adapter import OnlineParameterAdapter
from twin_track import twin_track_model
from sensors import SensorSimulator

MIN_VEHICLE_MASS_KG = 100.0
def ornstein_uhlenbeck(prev, mu=0.0, theta=0.15, sigma=0.1, dt=0.05):
    """
    Temporally correlated random process.
    Generates smooth, realistic control sequences.

    Args:
        prev:  previous value
        mu:    mean (equilibrium point)
        theta: mean reversion rate (higher = snappier return to mu)
        sigma: noise amplitude
        dt:    timestep

    Returns:
        next value
    """
    noise = sigma * np.sqrt(dt) * np.random.randn()
    return prev + theta * (mu - prev) * dt + noise
def generate_ou_controls(n_samples, dt=0.05):
    """
    Generate realistic control sequences using Ornstein-Uhlenbeck process.

    Produces smooth, correlated inputs that look like real driving —
    gradual steering corrections, sustained throttle, occasional braking.
    Much better training data than pure random controls.

    Args:
        n_samples: number of timesteps
        dt:        timestep

    Returns:
        controls array (n_samples, 3): [steering, throttle, brake]
    """
    controls = []

    # Initial values
    steer    = 0.0
    throttle = 0.6
    brake    = 0.0

    for _ in range(n_samples):
        # Steering: slow oscillation around zero
        steer = ornstein_uhlenbeck(steer, mu=0.0, theta=0.3, sigma=0.15, dt=dt)
        steer = np.clip(steer, -0.25, 0.25)

        # Throttle/brake: mutually exclusive
        if brake > 0.05:
            # Currently braking — tend to release
            throttle = 0.0
            brake = ornstein_uhlenbeck(brake, mu=0.0, theta=0.5, sigma=0.1, dt=dt)
            brake = np.clip(brake, 0.0, 0.3)
        else:
            # Currently on throttle
            throttle = ornstein_uhlenbeck(throttle, mu=0.6, theta=0.2, sigma=0.15, dt=dt)
            throttle = np.clip(throttle, 0.2, 1.0)
            brake = 0.0

            # Small chance of starting to brake
            if np.random.rand() < 0.02:
                throttle = 0.0
                brake    = 0.1

        controls.append([steer, throttle, brake])

    return np.array(controls)
class IRDAS:
    """
    In-race Data Augmentation System - integrates all components:
    - Real vehicle simulator (generates mismatched data)
    - Kalman filter (state estimation from noisy sensors)
    - Neural network (learns residual dynamics)
    - Parameter adapter (tunes model parameters online)
    """
    
    def __init__(self, baseline_params, device='cpu', use_nn=True, use_rls=True):
        """
        Initialize IRDAS.
        
        Args:
            baseline_params: baseline vehicle parameters dict
            device: 'cpu' or 'cuda'
            use_nn: whether to use neural network for residual learning
            use_rls: whether to use recursive least squares for parameter adaptation
        """
        self.baseline_params = baseline_params.copy()
        self.true_state = np.array([0., 0., 0., 30., 0., 0., 30., 30., 30., 30., 8000., 4., 0.5])
        self.true_vehicle_mass = float(self.baseline_params.get('M', 752.0))
        self.nn_trained = False 
        self.device = device
        self.use_nn = use_nn
        self.use_rls = use_rls
        # Components
        self.real_simulator = None  # initialized with true params
        self.kalman_filter = ExtendedKalmanFilter(baseline_params)
        
        if use_nn:
            self.nn_learner = ResidualDynamicsLearner(state_dim=7, control_dim=3, device=device)
            self.nn_trained = False
        
        if use_rls:
            self.param_adapter = OnlineParameterAdapter(baseline_params)
        
        # Current model parameters (adapt during run)
        self.current_params = baseline_params.copy()
        
        # History tracking
        self.history = {
            'true_states': [],
            'estimated_states': [],
            'measured_states': [],
            'fuel_flow_measured': [],
            'true_vehicle_mass': [],
            'estimated_vehicle_mass': [],
            'controls': [],
            'residuals': [],
            'model_errors': [],
            'param_changes': [],
            'timestamps': []
        }
        
        self.time_step = 0.0
        self.dt = 0.05
        self.sensor_sim = SensorSimulator(dt=self.dt, initial_mass=self.true_vehicle_mass)

    def initialize_real_vehicle(self, true_params=None, seed=None):
        """
        Initialize the real vehicle simulator with possibly mismatched parameters.
        
        Args:
            true_params: dict of true vehicle parameters (mismatches)
            seed: random seed for reproducibility
        """
        self.real_simulator = RealVehicleSimulator(true_params=true_params, seed=seed)
        self.true_vehicle_mass = float(self.real_simulator.true_params.get('M', self.true_vehicle_mass))
        self.sensor_sim.reset(initial_mass=self.true_vehicle_mass)
        print(f"Real vehicle initialized with parameter differences:")
        for name, diff in self.real_simulator.get_parameter_difference().items():
            print(f"  {name}: {diff:+.4f}")



    def pretrain_neural_network(self, n_training_samples=5000, epochs=100, batch_size=32):
        """
        Pretrain neural network on generated data from simulator.
        Uses reduced 7-state space for dynamics-relevant learning.
        
        Args:
            n_training_samples: number of training samples
            epochs: training epochs
            batch_size: batch size
        """
        if not self.use_nn:
            print("Neural network disabled")
            return
        
        if self.real_simulator is None:
            raise ValueError("Real vehicle simulator not initialized. Call initialize_real_vehicle first.")
        
        print("\n" + "="*50)
        print("Pretraining Neural Network (7-state reduced space)")
        print("="*50)
        
        print(f"Generating {n_training_samples} training samples...")
        
        # Generate trajectories from both models
        states_baseline = []
        states_real = []
        controls_list = []
        
        state_baseline = np.array([0., 0., 0., 30., 0., 0., 30., 30., 30., 30., 8000., 4., 0.5])
        state_real = state_baseline.copy()
        self.real_simulator.reset_history()   # clear pretraining steps from history
        self.real_simulator.reset_history()
        print(f"History length after reset: {len(self.real_simulator.state_history)}")
        ou_controls = generate_ou_controls(n_training_samples)
        for i in range(n_training_samples):
            # Random control
            u = ou_controls[i]
            
            try:
                # Baseline model
                state_baseline_next = twin_track_model(state_baseline, u, self.dt, self.baseline_params)
                
                # Real model
                state_real_next = self.real_simulator.step(state_real, u, self.dt)
                
                states_baseline.append(state_baseline.copy())
                states_real.append(state_real.copy())
                controls_list.append(u.copy())
                
                state_baseline = state_baseline_next.copy()
                state_real = state_real_next.copy()
                
            except Exception as e:
                print(f"Warning at sample {i}: {e}")
                continue
        
        states_baseline = np.array(states_baseline)
        states_real = np.array(states_real)
        controls_list = np.array(controls_list)
        
        # Extract 7 dynamics states from full 13-state vectors
        # Indices: [vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr] = [3, 4, 5, 6, 7, 8, 9]
        states_baseline_7 = states_baseline[:, [3, 4, 5, 6, 7, 8, 9]]
        states_real_7 = states_real[:, [3, 4, 5, 6, 7, 8, 9]]
        
        # Compute residuals in 7-state space
        residuals = states_real_7 - states_baseline_7
        
        # Split into train/val
        split = int(0.8 * len(states_baseline_7))
        train_states = states_baseline_7[:split]
        train_controls = controls_list[:split]
        train_residuals = residuals[:split]
        
        val_states = states_baseline_7[split:]
        val_controls = controls_list[split:]
        val_residuals = residuals[split:]
        
        # Train NN
        print("\nTraining neural network with normalization and L2 regularization...")
        self.nn_learner.fit(
            train_states, train_controls, train_residuals,
            val_states, val_controls, val_residuals,
            epochs=epochs, batch_size=batch_size, verbose=True
        )
        
        self.nn_trained = True
        print("Neural network training complete!")
    
    def _apply_nn_residual_correction(self, full_state, control, correction_scale=0.1):
        """
        Apply neural network residual correction and reconstruct full state.
        
        The NN predicts residuals for only the 7 dynamics states:
        [vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr] at indices [3,4,5,6,7,8,9]
        
        STATE RECONSTRUCTION LOGIC:
        - Dynamics states [3,4,5,6,7,8,9]: Corrected by NN
        - Position [0,1] and yaw [2]: NOT updated here (will be integrated in next KF prediction)
        - RPM [10]: UPDATED based on corrected wheel speeds (maintains consistency)
        - Gear [11] and throttle [12]: Unchanged (gear is discrete, throttle is control)
        
        Why position/yaw not updated:
          The Kalman filter naturally integrates corrected velocities in the next prediction step.
          Manually updating here would cause double-counting of changes.
        
        Why RPM updated:
          RPM is deterministically computed from wheel speeds (not a free variable).
          If wheel speeds change, RPM must change to stay physically consistent.
        
        Args:
            full_state: 13-element state vector from Kalman filter
            control: 3-element control input [steering, throttle, brake]
            correction_scale: scaling factor for residual correction (default 0.1)
            
        Returns:
            corrected_full_state: 13-element state with NN corrections applied
        """
        # Start with uncorrected full state
        corrected_state = full_state.copy()
        
        # Extract the 7 dynamics states
        dynamics_indices = [3, 4, 5, 6, 7, 8, 9]
        state_dynamics = full_state[dynamics_indices]
        
        # Get NN prediction for residuals using the proper predict method
        # This handles normalization and tensor conversions correctly
        residual_correction = self.nn_learner.predict(state_dynamics, control)
        
        # Apply residual correction to the 7 dynamics states
        corrected_state[dynamics_indices] = full_state[dynamics_indices] + residual_correction * correction_scale
        
        # UPDATE DERIVED STATE: RPM must be recomputed based on corrected wheel speeds
        # RPM is computed deterministically from wheel speeds and current gear
        # State indices: [6,7,8,9] = [vw_fl, vw_fr, vw_rl, vw_rr]
        corrected_wheel_speeds = corrected_state[[6, 7, 8, 9]]
        mean_wheel_speed = np.mean(corrected_wheel_speeds)
        
        # Get gear ratio for current gear
        current_gear = int(corrected_state[11])
        current_gear = np.clip(current_gear, 1, 8)  # Ensure valid gear (1-8)
        
        # Get parameters
        if 'GEAR_RATIOS' in self.baseline_params:
            gear_ratios = self.baseline_params['GEAR_RATIOS']
            final_drive = self.baseline_params.get('final_drive', 6.3)
        else:
            # Fallback values if not in params
            gear_ratios = {1: 3.15, 2: 2.47, 3: 2.07, 4: 1.75, 5: 1.48, 6: 1.28, 7: 1.0, 8: 0.84}
            final_drive = 6.3
        
        # Compute corrected RPM
        if current_gear in gear_ratios:
            gear_ratio = gear_ratios[current_gear]
            corrected_rpm = mean_wheel_speed * gear_ratio * final_drive * 60.0 / (2 * np.pi * 0.33)
            corrected_rpm = np.clip(corrected_rpm, 0, 15500)  # Clamp to valid RPM range
            corrected_state[10] = corrected_rpm
        
        # UNCHANGED STATES (explained above):
        # - Position [0,1] and yaw [2]: Will be integrated naturally in next KF prediction
        # - Gear [11]: Discrete, determined by check_upshift(), not modified mid-step
        # - Throttle [12]: Control input, not modified
        
        return corrected_state
    
    def step(self, control, measurement_noise_std=None, use_nn_correction=True, use_param_adaptation=True):
        """
        Execute one time step of IRDAS with full state reconstruction.
        
        STATE RECONSTRUCTION PROCESS:
        The system maintains a 13-state vector but NN operates on reduced 7-state space:
        
        Full 13-state: [x, y, psi, vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr, rpm, gear, throttle]
        Dynamics 7-state: [vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr]  (indices 3,4,5,6,7,8,9)
        Non-dynamics 6-state: [x, y, psi, rpm, gear, throttle]  (indices 0,1,2,10,11,12)
        
        During each step:
        1. Kalman filter predicts full 13-state using baseline model
        2. NN extracts 7-state dynamics subset and predicts residuals
        3. Full state is reconstructed: corrected_dynamics + unchanged_non-dynamics
        4. Kalman filter updates full 13-state with measurement
        5. Parameters adapt based on prediction errors
        
        Args:
            control: control input [steering, throttle, brake]
            measurement_noise_std: dict with sensor noise std devs
            use_nn_correction: whether to use NN for residual correction (7-state only)
            use_param_adaptation: whether to adapt parameters
            
        Returns:
            estimated_state: Kalman filter estimated full 13-state
        """
        step_num = len(self.history['true_states'])
        if step_num < 3:
            print(f"\n--- Step {step_num} RAW ---")
            print(f"Control input:     {control}")
            print(f"State going IN:    vx={self.kalman_filter.x[3]:.3f}, gear={self.kalman_filter.x[11]:.0f}, rpm={self.kalman_filter.x[10]:.0f}")
            print(f"True state going IN: {self.real_simulator.get_state_history()[-1][3] if self.real_simulator.state_history else 'EMPTY'}")
        # Default measurement noise
        if measurement_noise_std is None:
            measurement_noise_std = {
                'x_gps': 1.0, 'y_gps': 1.0,  # GPS position (meters)
                'ax': 0.5, 'ay': 0.5, 'r': 0.01,  # IMU noise
                'vx': 0.1, 'vy': 0.1, 'rpm': 50, 'wheel_speed': 0.2  # other sensors
            }
        
        # Step 1: Get true state from real vehicle simulator
        
        true_state_before = self.true_state.copy()          # snapshot BEFORE step
        # Fuel-flow sensor drives mass depletion; this mass is treated as measured (not RLS-estimated).
        fuel_flow_true = self.sensor_sim.estimate_fuel_flow(true_state_before, control)
        fuel_consumed = fuel_flow_true * self.dt
        self.true_vehicle_mass = max(MIN_VEHICLE_MASS_KG, self.true_vehicle_mass - fuel_consumed)
        if self.real_simulator is not None:
            self.real_simulator.true_params['M'] = self.true_vehicle_mass
        true_state_next = self.real_simulator.step(self.true_state, control, self.dt)
        self.true_state = true_state_next.copy()

        # Step 2: Generate noisy measurement from true state
        measurement = self.sensor_sim.measure(true_state_next)
        fuel_flow_measured, mass_sensor_kg = self.sensor_sim.measure_fuel_system(
            true_state_next, control, self.true_vehicle_mass
        )
        
        # Step 3: Kalman filter prediction (baseline model)
        self.kalman_filter.predict(control, self.dt, fuel_flow_kgps=fuel_flow_measured)
        
        # Step 4: NN-based residual correction (optional, uses reduced 7-state space)
        if use_nn_correction and self.nn_trained:
            # Apply NN residual correction with full state reconstruction
            # This ensures dynamics states [3,4,5,6,7,8,9] are corrected
            # while non-dynamics states [0,1,2,10,11,12] remain unchanged
            corrected_full_state = self._apply_nn_residual_correction(
                self.kalman_filter.x, 
                control, 
                correction_scale=0.1
            )
            self.kalman_filter.x = corrected_full_state
        
        # Step 5: Kalman filter update
        self.kalman_filter.update(measurement, mass_sensor_kg=mass_sensor_kg)
        estimated_state = self.kalman_filter.get_state()
        
        # Step 6: Compute residual for parameter adaptation
        baseline_next = twin_track_model(true_state_before, control, self.dt, self.baseline_params)
        model_error = true_state_next - baseline_next
        residual = np.linalg.norm(model_error)
        
        # Step 7: Parameter adaptation (optional)
        if use_param_adaptation and self.use_rls:
            # Only skip if error is exactly zero (no update signal)
            model_error_norm = np.linalg.norm(model_error[:7])
            if model_error_norm > 1e-12:  # Much more lenient guard

                # --- extract what we need from true_state (already defined above) ---
                vx    = true_state_before[3]
                vy    = true_state_before[4]
                r     = true_state_before[5]
                vw_rl = true_state_before[8]
                vw_rr = true_state_before[9]

                # --- slip angles (mirrors twin_track.py logic) ---
                L   = self.baseline_params.get('L',  3.6)
                MX  = self.baseline_params.get('MX', 0.453)
                Lf  = L * MX
                Lr  = L * (1.0 - MX)
                delta      = float(control[0])
                min_speed  = 0.5

                alpha_front = np.degrees(np.arctan2(
                    vy - r * Lf + delta * vx,
                    max(abs(vx), min_speed)
                ))
                alpha_rear  = np.degrees(np.arctan2(
                    vy + r * Lr,
                    max(abs(vx), min_speed)
                ))
                slip_angles = np.array([alpha_front, alpha_front,
                                        alpha_rear,  alpha_rear])
                slip_angles = np.clip(slip_angles, -20.0, 20.0)

                # --- longitudinal slip ratios (rear wheels only) ---
                v_car = max(abs(vx), min_speed)
                slip_ratios = np.array([
                    (vw_rl - v_car) / v_car,
                    (vw_rr - v_car) / v_car
                ])
                slip_ratios = np.clip(slip_ratios, -1.0, 1.0)

                # --- approximate normal forces (static split, good enough for RLS) ---
                M_veh = self.kalman_filter.get_mass_estimate()
                g     = 9.81
                Fz_front = M_veh * g * Lr / L / 2.0 / 1000.0   # kN per front wheel
                Fz_rear  = M_veh * g * Lf / L / 2.0 / 1000.0   # kN per rear wheel
                Fz_kN_wheels = np.array([Fz_front, Fz_front, Fz_rear, Fz_rear])

                # --- force residuals: vy error -> lateral, vx error -> longitudinal ---
                # Compute force errors directly from Pacejka at current slip angles
                # This is much cleaner than trying to back out forces from state errors
                from twin_track import pacejka_magic_formula

                # Lateral force difference: real params vs baseline params
                lat_force_real     = sum(
                    pacejka_magic_formula(a, Fz_kN_wheels[i],
                                         self.real_simulator.true_params['TYRE_LAT'],
                                         'lateral')
                    for i, a in enumerate(slip_angles)
                )
                lat_force_baseline = sum(
                    pacejka_magic_formula(a, Fz_kN_wheels[i],
                                         self.baseline_params['TYRE_LAT'],
                                         'lateral')
                    for i, a in enumerate(slip_angles)
                )
                lat_force_error = lat_force_real - lat_force_baseline

                # Longitudinal force difference: rear wheels only
                lon_force_real     = sum(
                    pacejka_magic_formula(slip_ratios[i] * 100, Fz_kN_wheels[2 + i],
                                         self.real_simulator.true_params['TYRE_LON'],
                                         'longitudinal')
                    for i in range(2)
                )
                lon_force_baseline = sum(
                    pacejka_magic_formula(slip_ratios[i] * 100, Fz_kN_wheels[2 + i],
                                         self.baseline_params['TYRE_LON'],
                                         'longitudinal')
                    for i in range(2)
                )
                lon_force_error = lon_force_real - lon_force_baseline
                # --- Additional signals for Cd, Cl adaptation ---
                # Cd (drag): estimated from sustained speed error
                speed_error = model_error[3]  # vx error -> drag/downforce effect
                
                # Improve drag signal with throttle state
                # If throttle is low and we're going slower than expected, could be high Cd
                throttle = float(control[1])
                if throttle < 0.5:
                    speed_error *= 1.5  # Amplify speed error signal during coasting

                self.param_adapter.update_rls(
                    slip_angles, slip_ratios, Fz_kN_wheels,
                    lat_force_error, lon_force_error,
                    speed_error=speed_error,
                    adaptive_factor=0.95
                )
                self.current_params = self.param_adapter.get_current_params()
                self.current_params['M'] = self.kalman_filter.get_mass_estimate()

                # --- Fix 4: push adapted params back into the Kalman filter ---
                self.kalman_filter.params = self.current_params
                self.kalman_filter.baseline_params = self.current_params
        
        # Step 8: Log history
        self.history['true_states'].append(true_state_next.copy())
        self.history['estimated_states'].append(estimated_state.copy())
        self.history['measured_states'].append(measurement.copy())
        self.history['fuel_flow_measured'].append(float(fuel_flow_measured))
        self.history['true_vehicle_mass'].append(float(self.true_vehicle_mass))
        self.history['estimated_vehicle_mass'].append(float(self.kalman_filter.get_mass_estimate()))
        self.history['controls'].append(control.copy())
        self.history['residuals'].append(model_error.copy())
        self.history['model_errors'].append(residual)
        self.history['timestamps'].append(self.time_step)
        
        if use_param_adaptation and self.use_rls:
            param_changes = self.param_adapter.get_parameter_changes()
            self.history['param_changes'].append(param_changes)
        
        self.time_step += self.dt
        self.time_step_count = len(self.history['true_states'])

        return estimated_state
    
    def _generate_control(self, t=0):
        """Generate physically realistic controls — no simultaneous throttle/brake."""
        vx = self.true_state[3]

        # Speed-dependent steering limit — less steering at high speed
        max_steer = np.clip(0.3 - 0.005 * vx, 0.05, 0.3)
        steering = np.random.uniform(-max_steer, max_steer)

        # Throttle/brake are mutually exclusive
        if np.random.rand() < 0.8:   # 80% chance throttle
            throttle = np.random.uniform(0.3, 0.9)
            brake    = 0.0
        else:                         # 20% chance braking
            throttle = 0.0
            brake    = np.random.uniform(0.05, 0.3)

        return np.array([steering, throttle, brake])   
    
    def simulate(self, n_steps=1000, control_strategy='random', show_progress=True):
        """
        Run full simulation of IRDAS.
        
        Args:
            n_steps: number of simulation steps
            control_strategy: 'random' or 'aggressive_maneuver'
            show_progress: whether to print progress
        """
        print("\n" + "="*50)
        print("Running IRDAS Simulation")
        print("="*50)
        
        for step in range(n_steps):
            # Generate control input
            if control_strategy == 'random':
                u = self._generate_control()
            elif control_strategy == 'aggressive_maneuver':
                # Sinusoidal steering with throttle
                t = step * self.dt
                u = np.array([
                    0.15 * np.sin(2 * np.pi * 0.1 * t),  # steering
                    0.6,  # throttle
                    0.0   # brake
                ])
            else:
                u = np.zeros(3)
            
            # Step IRDAS
            try:
                estimated_state = self.step(u)
            except Exception as e:
                print(f"Error at step {step}: {e}")
                if step > 10:
                    break
            
            if show_progress and (step % 100 == 0):
                print(f"Step {step}/{n_steps} - Time: {self.time_step:.2f}s, "
                      f"vx_est: {estimated_state[3]:.2f} m/s, "
                      f"Model error: {self.history['model_errors'][-1]:.4f}")
        
        print(f"\nSimulation complete! Ran {len(self.history['true_states'])} steps")
    def reset(self, initial_state=None):
        """Reset IRDAS between scenarios."""
        if initial_state is None:
            initial_state = np.array([0., 0., 0., 30., 0., 0., 
                                    30., 30., 30., 30., 8000., 4., 0.5])
        self.true_state = initial_state.copy()
        if self.real_simulator is not None:
            self.true_vehicle_mass = float(self.real_simulator.true_params.get('M', self.baseline_params.get('M', 752.0)))
        else:
            self.true_vehicle_mass = float(self.baseline_params.get('M', 752.0))
        self.kalman_filter.reset(initial_state)
        self.sensor_sim.reset(initial_mass=self.true_vehicle_mass)
        if self.real_simulator is not None:
            self.real_simulator.reset_history()
        if self.use_rls and hasattr(self, 'param_adapter'):
            self.param_adapter.reset_to_baseline()  # CRITICAL: reset parameter adapter
        self.history = {k: [] for k in self.history}  # clear history
        self.time_step = 0.0
        self.nn_trained = True if (self.use_nn and hasattr(self, 'nn_learner')) else False
        self.nn_trained = True if (self.use_nn and hasattr(self, 'nn_learner')) else False  # ADD THIS


    def get_metrics(self):
        """Compute and return performance metrics."""
        if len(self.history['true_states']) == 0:
            return {}
        
        true_states = np.array(self.history['true_states'])
        estimated_states = np.array(self.history['estimated_states'])
        model_errors = np.array(self.history['model_errors'])
        
        # Estimate errors (difference between true and estimated)
        estimation_errors = np.linalg.norm(true_states - estimated_states, axis=1)
        
        metrics = {
            'avg_model_error': float(np.mean(model_errors)),
            'max_model_error': float(np.max(model_errors)),
            'std_model_error': float(np.std(model_errors)),
            'avg_estimation_error': float(np.mean(estimation_errors)),
            'max_estimation_error': float(np.max(estimation_errors)),
            'std_estimation_error': float(np.std(estimation_errors)),
            'n_steps': len(true_states),
            'total_time': self.time_step
        }
        
        # Parameter adaptation metrics
        if len(self.history['param_changes']) > 0:
            last_param_changes = self.history['param_changes'][-1]
            metrics['final_param_changes'] = {
                name: change_info['change_pct'] 
                for name, change_info in last_param_changes.items()
            }
        true_arr = np.array(self.history['true_states'])
        est_arr  = np.array(self.history['estimated_states'])

        state_names = ['x','y','psi','vx','vy','r',
                    'vw_fl','vw_fr','vw_rl','vw_rr','rpm','gear','throttle']

        print("\nPer-state RMS error (true vs estimated):")
        for i, name in enumerate(state_names):
            rms = np.sqrt(np.mean((true_arr[:,i] - est_arr[:,i])**2))
            print(f"  {name:<10}: {rms:.4f}")


        return metrics
    
    def verify_state_reconstruction(self):
        """
        Verify state reconstruction logic during NN correction.
        
        Documents how the 13-state full state is reconstructed from NN corrections
        on the 7-state reduced dynamics space, with proper handling of derived states.
        
        Returns:
            dict with state component information including three categories:
            - Independent states: Not updated by NN (let KF integrate)
            - Derived states: Updated for consistency (RPM from wheel speeds)
            - Discrete/Control states: Never modified
        """
        state_info = {
            'total_states': 13,
            'dynamics_states': {
                'count': 7,
                'indices': [3, 4, 5, 6, 7, 8, 9],
                'names': ['vx', 'vy', 'r', 'vw_fl', 'vw_fr', 'vw_rl', 'vw_rr'],
                'description': 'Velocity components and wheel speeds - CORRECTED by NN'
            },
            'independent_states': {
                'count': 3,
                'indices': [0, 1, 2],
                'names': ['x', 'y', 'psi'],
                'description': 'Position and yaw - NOT updated by NN (KF integrates naturally)',
                'reason': 'Updating manually would double-count velocity corrections'
            },
            'derived_states': {
                'count': 1,
                'indices': [10],
                'names': ['rpm'],
                'description': 'Engine RPM - UPDATED for consistency',
                'reason': 'Deterministically computed from wheel speeds and gear',
                'formula': 'rpm = mean(wheel_speeds) * gear_ratio * final_drive'
            },
            'discrete_control_states': {
                'count': 2,
                'indices': [11, 12],
                'names': ['gear', 'throttle'],
                'description': 'Discrete/control states - NEVER modified',
                'reason': 'Gear is discrete decision, throttle is driver control'
            },
            'reconstruction_process': {
                'step_1': 'Kalman filter predicts full 13-state using baseline model',
                'step_2': 'NN extracts indices [3,4,5,6,7,8,9] (7 dynamics states)',
                'step_3': 'NN predicts residuals for these 7 states',
                'step_4': 'Residuals are scaled and applied to dynamics indices',
                'step_5': 'Derived states (RPM) recomputed from corrected wheel speeds',
                'step_6': 'Independent states [0,1,2] left from KF prediction (no double-counting)',
                'step_7': 'Discrete/control states [11,12] left unchanged',
                'step_8': 'Full 13-state passed to KF update with measurement'
            },
            'key_principle': 'Dynamics states learned by NN. Derived states updated for consistency. ' + \
                           'Independent states updated naturally by KF in next step (no double-counting). ' + \
                           'Discrete/control states never modified by NN.'
        }
        return state_info
    
    def save_results(self, filepath):
        """Save simulation results to file."""
        import pickle
        
        data = {
            'history': self.history,
            'baseline_params': self.baseline_params,
            'current_params': self.current_params,
            'metrics': self.get_metrics()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Results saved to {filepath}")
    
    def load_pretrained_network(self, model_path):
        """Load pretrained neural network."""
        if self.use_nn:
            self.nn_learner.load(model_path)
            self.nn_trained = True
            print(f"Loaded pretrained network from {model_path}")


if __name__ == "__main__":
    print("IRDAS Integration Test")
    print("=" * 50)
    
    # Import params
    from params import *
    
    baseline_params = {
        'L': L, 'TF': TF, 'TR': TR, 'H': H, 'MX': MX, 'M': M,
        'Cd': Cd, 'Cl': Cl, 'Area': Area, 'CX': CX,
        'K': K, 'final_drive': final_drive, 'tyre_radius': tyre_radius,
        'TYRE_LAT': TYRE_LAT.copy(), 'TYRE_LON': TYRE_LON.copy(),
        'GEAR_RATIOS': GEAR_RATIOS.copy(), 'UPSHIFT_SPEED_KPH': UPSHIFT_SPEED_KPH.copy(),
        'ENGINE_RPM': ENGINE_RPM.copy(), 'ENGINE_TORQUE_NM': ENGINE_TORQUE_NM.copy()
    }
    
    # Initialize IRDAS
    irdas = IRDAS(baseline_params, device='cpu', use_nn=True, use_rls=True)
    print("IRDAS system initialized")
    
    # Initialize real vehicle with mismatch
    irdas.initialize_real_vehicle(seed=42)

    print("Real vehicle simulator initialized with parameter mismatch")
    
    # Quick test: just a few steps
    print("\nRunning quick test (50 steps)...")
    irdas.simulate(n_steps=50, control_strategy='random', show_progress=True)
    
    # Get metrics
    metrics = irdas.get_metrics()
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v:.4f}")
        else:
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
