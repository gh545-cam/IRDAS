"""
Real vehicle data simulator with parameter mismatch.
Generates "real" vehicle data that differs from the model to simulate real-world conditions.
"""
import numpy as np
from params import *
from twin_track import twin_track_model, generate_trajectory


class RealVehicleSimulator:
    """
    Simulates a 'real' vehicle with slightly different parameters than our model.
    This creates the mismatch that the NN will learn to correct.
    """
    
    def __init__(self, true_params=None, seed=None):
        """
        Initialize simulator with true (simulated real) vehicle parameters.
        
        Args:
            true_params: dict of parameters that differ from baseline
            seed: random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Create baseline parameters
        self.baseline_params = {
            'L': L, 'TF': TF, 'TR': TR, 'H': H, 'MX': MX, 'M': M,
            'Cd': Cd, 'Cl': Cl, 'Area': Area, 'CX': CX,
            'K': K, 'final_drive': final_drive, 'tyre_radius': tyre_radius,
            'TYRE_LAT': TYRE_LAT.copy(), 'TYRE_LON': TYRE_LON.copy(),
            'GEAR_RATIOS': GEAR_RATIOS.copy(), 'UPSHIFT_SPEED_KPH': UPSHIFT_SPEED_KPH.copy(),
            'ENGINE_RPM': ENGINE_RPM.copy(), 'ENGINE_TORQUE_NM': ENGINE_TORQUE_NM.copy()
        }
        
        # True parameters (slightly different from baseline)
        self.true_params = self.baseline_params.copy()
        
        if true_params is None:
            # Randomize tire coefficients and mass to simulate real vehicle
            # Tire coefficients change with temperature, wear, etc.
            true_params = self._generate_random_mismatch()
        
        self.true_params.update(true_params)
        self.state_history = []
        self.control_history = []
    
    def _generate_random_mismatch(self):
        """Generate random parameter mismatches (±5-15% variation)."""
        mismatch = {}
        
        # Tire parameter mismatches (common in real vehicles)
        lat_params = TYRE_LAT.copy()
        lon_params = TYRE_LON.copy()
        
        # Vary tire stiffness (B coefficient)
        lat_params['B'] *= np.random.uniform(0.9, 1.1)
        lon_params['B'] *= np.random.uniform(0.9, 1.1)
        
        # Vary peak friction (a2 coefficient)
        lat_params['a2'] *= np.random.uniform(0.95, 1.05)
        lon_params['a2'] *= np.random.uniform(0.95, 1.05)
        
        # Vary shape factor (C coefficient)
        lat_params['C'] *= np.random.uniform(0.95, 1.05)
        lon_params['C'] *= np.random.uniform(0.95, 1.05)
        
        mismatch['TYRE_LAT'] = lat_params
        mismatch['TYRE_LON'] = lon_params
        
        # Mass variation (fuel load, passengers)
        mismatch['M'] = M * np.random.uniform(0.95, 1.05)
        
        # Aerodynamic variation
        mismatch['Cd'] = Cd * np.random.uniform(0.95, 1.05)
        mismatch['Cl'] = Cl * np.random.uniform(0.95, 1.05)
        
        # Engine torque variation (wear, tuning)
        mismatch['ENGINE_TORQUE_NM'] = ENGINE_TORQUE_NM * np.random.uniform(0.98, 1.02)
        
        return mismatch
    
    def step(self, state, control, dt=0.05):
        """
        Step the real vehicle simulator forward.
        
        Args:
            state: current state vector
            control: control input [steering, throttle, brake]
            dt: time step
            
        Returns:
            next state (from real vehicle model)
        """
        next_state = twin_track_model(state, control, dt, self.true_params)
        self.state_history.append(state.copy())
        self.control_history.append(control.copy())
        return next_state
    
    def generate_trajectory(self, n_steps=1000, dt=0.05):
        """
        Generate a trajectory using the real vehicle model with random controls.
        
        Args:
            n_steps: number of simulation steps
            dt: time step
            
        Returns:
            (states, controls) arrays
        """
        # Initialize state
        state = np.array([
            0., 0., 0.,           # x, y, psi
            30., 0., 0.,          # vx=30 m/s (highway speed), vy=0, r=0
            30., 30., 30., 30.,   # wheel speeds match vx
            8000., 4., 0.5        # rpm, gear 4, half throttle
        ])
        
        states = [state.copy()]
        controls = []
        
        for step in range(n_steps):
            # Generate random controls
            throttle = np.random.uniform(0.3, 0.8)
            brake    = 0.0 if throttle > 0.5 else np.random.uniform(0.0, 0.2)
            u = np.array([
                np.random.uniform(-0.15, 0.15),   # gentle steering
                throttle,
                brake
            ])
            
            controls.append(u)
            
            try:
                state = self.step(state, u, dt)
                # Safety clamps
                state[0:2] = np.clip(state[0:2], -1e4, 1e4)
                state[2] = np.remainder(state[2], 2*np.pi)
                state[3] = np.clip(state[3], -10, 95)
                state[4] = np.clip(state[4], -35, 35)
                state[5] = np.clip(state[5], -4, 4)
                state[6:10] = np.clip(state[6:10], 0, 100)
                state[10] = np.clip(state[10], 1000, 15500)
                
                states.append(state.copy())
            except Exception as e:
                print(f"Warning at step {step}: {e}")
                state = np.array([0., 0., 0., 5., 0., 0., 5., 5., 5., 5., 3000., 1., 0.1], dtype=np.float64)
                states.append(state.copy())
        
        return np.array(states), np.array(controls)
    
    def get_state_history(self):
        """Return the full state history."""
        return self.state_history
    
    def get_control_history(self):
        """Return the full control history."""
        return self.control_history
    
    def reset_history(self):
        """Clear state and control history."""
        self.state_history = []
        self.control_history = []
    
    def get_parameter_difference(self):
        """
        Return the difference between true and baseline parameters (for analysis).
        
        Returns:
            dict with parameter differences
        """
        diff = {}
        
        # Tire parameters
        diff['TYRE_LAT_B'] = self.true_params['TYRE_LAT']['B'] - self.baseline_params['TYRE_LAT']['B']
        diff['TYRE_LAT_a2'] = self.true_params['TYRE_LAT']['a2'] - self.baseline_params['TYRE_LAT']['a2']
        diff['TYRE_LON_B'] = self.true_params['TYRE_LON']['B'] - self.baseline_params['TYRE_LON']['B']
        
        # Mass
        diff['M'] = self.true_params['M'] - self.baseline_params['M']
        
        # Aero
        diff['Cd'] = self.true_params['Cd'] - self.baseline_params['Cd']
        diff['Cl'] = self.true_params['Cl'] - self.baseline_params['Cl']
        
        return diff


def compare_models(baseline_state, real_state, control, dt=0.05):
    """
    Compare baseline and real vehicle dynamics for a single step.
    
    Args:
        baseline_state: state from baseline model
        real_state: state from real vehicle
        control: control input
        dt: time step
        
    Returns:
        residual state (difference between models)
    """
    # Create baseline and real simulators
    baseline_sim = RealVehicleSimulator(true_params={})  # no mismatch
    real_sim = RealVehicleSimulator()  # with mismatch
    
    # Step both forward
    baseline_next = twin_track_model(baseline_state, control, dt, baseline_sim.baseline_params)
    real_next = real_sim.step(real_state, control, dt)
    
    # Return residual (difference)
    return real_next - baseline_next


if __name__ == "__main__":
    print("Real Vehicle Simulator Test")
    print("=" * 50)
    
    # Create real vehicle simulator
    simulator = RealVehicleSimulator(seed=42)
    print(f"True vehicle parameters generated")
    print(f"Parameter differences: {simulator.get_parameter_difference()}")
    
    # Generate a short trajectory
    print("\nGenerating trajectory...")
    states, controls = simulator.generate_trajectory(n_steps=100, dt=0.05)
    print(f"Generated {len(states)} state samples")
    print(f"State shape: {states.shape}, Control shape: {controls.shape}")
    print(f"State range - vx: [{states[:, 3].min():.2f}, {states[:, 3].max():.2f}] m/s")
    print(f"State range - vy: [{states[:, 4].min():.2f}, {states[:, 4].max():.2f}] m/s")
