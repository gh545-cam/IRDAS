import numpy as np
def pacejka_magic_formula(slip, Fz_kN, params: dict, slip_type: str = 'lateral') -> float:
    """
    Pacejka Magic Formula tire model.
    
    Args:
        slip: slip angle (deg) for lateral, or slip ratio (-) for longitudinal
        Fz_kN: normal force in kN
        params: tire parameter dictionary (TYRE_LAT or TYRE_LON)
        slip_type: 'lateral' or 'longitudinal'
    
    Returns:
        Tire force in Newtons
    """
    B = params['B']
    C = params['C']
    E = params['E']
    
    # Load-sensitive peak factor D
    a1 = params['a1']
    a2 = params['a2']
    D = a1 * Fz_kN + a2  # peak force scales with load
    D = max(D, 0.1)  # prevent negative peak
    
    # Build the magic formula: F = D * sin(C * atan(B*slip - E*(B*slip - atan(B*slip))))
    arg = B * slip - E * (B * slip - np.arctan(B * slip))
    F = D * np.sin(C * np.arctan(arg))
    
    return F


def pacejka_cornering_stiffness(Fz_kN, params: dict) -> float:
    """
    Compute cornering stiffness (N/deg) from load.
    
    Args:
        Fz_kN: normal force in kN
        params: tire parameter dictionary
    
    Returns:
        Cornering stiffness in N/deg
    """
    BCD_a3 = params['BCD_a3']
    BCD_a4 = params['BCD_a4']
    BCD_a5 = params['BCD_a5']
    
    # Simplified load sensitivity model
    Cs = BCD_a3 * np.exp(-BCD_a5 * abs(Fz_kN - BCD_a4))
    return Cs


def calculate_load_transfer(ax, ay, mass, h_com, track_width, K_roll_ratio=1.35):
    """
    Calculate load transfer on front and rear axles.
    
    Args:
        ax: longitudinal acceleration (m/s²)
        ay: lateral acceleration (m/s²)
        mass: vehicle mass (kg)
        h_com: height of center of mass (m)
        track_width: track width (m) - use front for front axle, rear for rear
        K_roll_ratio: front-to-rear roll stiffness ratio
    
    Returns:
        Tuple of (left_load_change, right_load_change) in N
    """
    g = 9.81
    
    # Lateral load transfer: ΔFz_lat = m * ay * h_com / track_width
    delta_fz_lat = mass * ay * h_com / track_width
    
    # Longitudinal load transfer: ΔFz_lon = m * ax * h_com / wheelbase
    # (proportional per axle)
    delta_fz_lon = mass * ax * h_com  # gets divided by wheelbase in main function
    
    return delta_fz_lat, delta_fz_lon


def get_engine_torque(rpm: float, params: dict = None) -> float:
    """
    Interpolate engine torque from RPM using the torque map.
    
    Args:
        rpm: engine RPM
        params: parameters dict with ENGINE_RPM and ENGINE_TORQUE_NM
    
    Returns:
        Engine torque in Nm
    """
    if params is None:
        params = globals()
    
    rpm_array = params.get('ENGINE_RPM', np.array([3000, 15500]))
    torque_array = params.get('ENGINE_TORQUE_NM', np.array([440, 340]))
    idle_rpm = float(params.get('IDLE_RPM', 1000.0))
    idle_torque = float(params.get('IDLE_TORQUE_NM', 110.0))
    
    if rpm <= idle_rpm:
        return idle_torque
    if rpm < rpm_array[0]:
        # Smoothly ramp from idle torque into torque-map region.
        return float(np.interp(rpm, [idle_rpm, rpm_array[0]], [idle_torque, torque_array[0]]))
    if rpm > rpm_array[-1]:
        return torque_array[-1]
    
    return float(np.interp(rpm, rpm_array, torque_array))


def check_upshift(speed_mps: float, current_gear: int, params: dict = None) -> int:
    """
    Check if upshift should occur based on speed.
    
    Args:
        speed_mps: vehicle speed in m/s
        current_gear: current gear number
        params: parameters dict
    
    Returns:
        New gear (upshifted if applicable)
    """
    if params is None:
        params = globals()
    
    upshift_speeds = params.get('UPSHIFT_SPEED_KPH', {})
    speed_kph = speed_mps * 3.6
    
    new_gear = current_gear
    # Auto-upshift if speed exceeds threshold
    if current_gear in upshift_speeds:
        if speed_kph > upshift_speeds[current_gear]:
            new_gear = min(current_gear + 1, 8)
    
    return new_gear


def check_downshift(speed_mps: float, current_gear: int, throttle: float,
                    engine_rpm: float, params: dict = None) -> int:
    """
    Check if downshift should occur based on speed, demand, and low-RPM lugging.
    Uses hysteresis against upshift thresholds to avoid gear hunting.
    """
    if params is None:
        params = globals()

    if current_gear <= 1:
        return 1

    upshift_speeds = params.get('UPSHIFT_SPEED_KPH', {})
    speed_kph = speed_mps * 3.6

    prev_gear = current_gear - 1
    prev_upshift = upshift_speeds.get(prev_gear, 80.0)
    # Downshift threshold below the previous upshift point (hysteresis band).
    downshift_threshold = 0.82 * prev_upshift

    high_demand = throttle > 0.35
    low_rpm = engine_rpm < 3000.0
    too_slow_for_gear = speed_kph < downshift_threshold

    # Target-speed kickdown floor: high desired pace should not stay in a tall gear.
    target_speed = float(params.get('TARGET_SPEED_MPS', speed_mps))
    if target_speed > 62.0:
        min_gear_for_target = 4
    elif target_speed > 50.0:
        min_gear_for_target = 3
    elif target_speed > 36.0:
        min_gear_for_target = 2
    else:
        min_gear_for_target = 1

    # Emergency relaunch downshift at low vehicle speed.
    if speed_mps < 12.0 and current_gear > 2:
        return current_gear - 1

    if current_gear > min_gear_for_target and high_demand:
        return current_gear - 1

    if (high_demand and low_rpm) or too_slow_for_gear:
        return prev_gear
    return current_gear


def check_grip_limit(Fx_demanded, Fy_demanded, Fz_kN,
                     params_lat: dict, params_lon: dict | None = None) -> tuple:
    """
    Check if demanded forces exceed grip limit using friction circle constraint.
    Applies combined force saturation: sqrt((Fx/F_max)^2 + (Fy/F_max)^2) <= 1
    
    Args:
        Fx_demanded: desired longitudinal force (N)
        Fy_demanded: desired lateral force (N)
        Fz_kN: normal force (kN)
        params_lat: lateral tire parameter dict
        params_lon: longitudinal tire parameter dict (optional)
    
    Returns:
        (Fx_limited, Fy_limited, utilization_factor)
        utilization_factor: 0-1, where 1 = at grip limit
    """
    # Conservative combined grip estimate from both lateral and longitudinal sets.
    if params_lon is None:
        params_lon = params_lat

    a1_lat = params_lat.get('a1', -0.10)
    a2_lat = params_lat.get('a2', 2.05)
    a1_lon = params_lon.get('a1', -0.10)
    a2_lon = params_lon.get('a2', 2.05)

    mu_lat = max(a1_lat * Fz_kN + a2_lat, 0.5)
    mu_lon = max(a1_lon * Fz_kN + a2_lon, 0.5)
    mu_peak = min(mu_lat, mu_lon)
    
    # Maximum available grip force
    Fz_N = Fz_kN * 1000
    F_max = mu_peak * Fz_N
    
    # Combined force demand (friction circle)
    F_demanded_total = np.sqrt(Fx_demanded**2 + Fy_demanded**2)
    
    # Calculate utilization factor and apply saturation
    if F_demanded_total > F_max:
        utilization_factor = F_max / F_demanded_total
        Fx_limited = Fx_demanded * utilization_factor
        Fy_limited = Fy_demanded * utilization_factor
    else:
        Fx_limited = Fx_demanded
        Fy_limited = Fy_demanded
        utilization_factor = F_demanded_total / F_max if F_max > 0 else 0
    
    return Fx_limited, Fy_limited, utilization_factor


def check_individual_grip(Fx, Fy, Fz_kN, params: dict) -> dict:
    """
    Analyze grip utilization in combined directions.
    
    Args:
        Fx: longitudinal force (N)
        Fy: lateral force (N)
        Fz_kN: normal force (kN)
        params: tire parameter dict
    
    Returns:
        dict with: {
            'Fx_utilization': ratio of Fx to max grip,
            'Fy_utilization': ratio of Fy to max grip,
            'combined_utilization': sqrt(Fx_util² + Fy_util²),
            'is_saturated': bool (True if > 0.95),
            'margin_to_slip': fraction of grip remaining
        }
    """
    # Peak friction coefficient
    a1 = params.get('a1', -0.10)
    a2 = params.get('a2', 2.05)
    mu_peak = max(a1 * Fz_kN + a2, 0.5)
    
    F_max = mu_peak * Fz_kN * 1000
    
    Fx_util = abs(Fx) / F_max if F_max > 0 else 0
    Fy_util = abs(Fy) / F_max if F_max > 0 else 0
    combined = np.sqrt(Fx_util**2 + Fy_util**2)
    
    return {
        'Fx_utilization': Fx_util,
        'Fy_utilization': Fy_util,
        'combined_utilization': combined,
        'is_saturated': combined >= 0.95,
        'margin_to_slip': max(0, 1.0 - combined)
    }


def generate_trajectory(n_steps: int = 10000, dt: float = 0.05, params: dict = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a trajectory using the twin track model with random controls.
    Returns (states, controls) as numpy arrays.
    
    Args:
        n_steps: number of simulation steps
        dt: time step (s)
        params: parameters dict (uses globals if None)
    
    Returns:
        (states, controls) where states shape is (n_steps+1, 13) and controls is (n_steps, 3)
    """
    if params is None:
        params = globals()
    
    # Initialize state: [x, y, psi, vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr, rpm, gear, throttle]
    state = np.array([
        0., 0., 0.,                    # x, y, psi
        5., 0., 0.,                    # vx, vy, r
        5., 5., 5., 5.,               # wheel speeds
        3000., 1., 0.1                # engine_rpm, gear, throttle
    ], dtype=np.float64)
    
    states = [state.copy()]
    controls = []
    
    for step in range(n_steps):
        # Random controls with realistic ranges
        u = np.array([
            np.random.uniform(-0.2, 0.2),      # steering angle (rad) ±11°
            np.random.uniform(0.0, 1.0),       # throttle pedal (0-1)
            np.random.uniform(0.0, 0.3)        # brake pedal (0-0.3)
        ], dtype=np.float64)
        
        controls.append(u)
        
        # Step the dynamics
        try:
            state = twin_track_model(state, u, dt, params)
            # Clamp state values to prevent numerical explosion
            state[0:2] = np.clip(state[0:2], -1e4, 1e4)  # position
            state[2] = np.remainder(state[2], 2*np.pi)   # wrap angle
            state[3] = np.clip(state[3], -10, 95)        # vx [m/s]
            state[4] = np.clip(state[4], -35, 35)        # vy [m/s]
            state[5] = np.clip(state[5], -4, 4)          # yaw rate [rad/s]
            state[6:10] = np.clip(state[6:10], 0, 100)   # wheel speeds
            state[10] = np.clip(state[10], 1000, 15500)  # RPM
            
            states.append(state.copy())
        except Exception as e:
            print(f"Warning: Error at step {step}: {e}")
            # Recovery: reset to safe state
            state = np.array([0., 0., 0., 5., 0., 0., 5., 5., 5., 5., 3000., 1., 0.1], dtype=np.float64)
            states.append(state.copy())
    
    return np.array(states), np.array(controls)


def twin_track_model(state: np.ndarray, u: np.ndarray, dt: float = 0.05, 
                     params: dict = None) -> np.ndarray:
    """
    Twin track vehicle dynamics model with load transfers, aero, and magic formula tires.
    
    State vector (13 elements):
        [0]  x           - global x position (m)
        [1]  y           - global y position (m)
        [2]  psi         - yaw angle (rad)
        [3]  vx          - longitudinal velocity in body frame (m/s)
        [4]  vy          - lateral velocity in body frame (m/s)
        [5]  r           - yaw rate (rad/s)
        [6]  vw_fl       - front-left wheel speed (m/s)
        [7]  vw_fr       - front-right wheel speed (m/s)
        [8]  vw_rl       - rear-left wheel speed (m/s)
        [9]  vw_rr       - rear-right wheel speed (m/s)
        [10] engine_rpm  - engine RPM
        [11] gear        - current gear (1-8)
        [12] throttle    - throttle input accumulator (0-1)
    
    Control input (3 elements):
        [0]  delta_steer - steering angle (rad)
        [1]  throttle    - throttle pedal (0-1)
        [2]  brake_pedal - brake pedal (0-1)
    
    Returns:
        Next state as np.ndarray (same dimension as state)
    """
    if params is None:
        params = globals()
    
    # Extract parameters
    L = params.get('L', 3.6)
    TF = params.get('TF', 1.58)
    TR = params.get('TR', 1.42)
    H = params.get('H', 0.295)
    MX = params.get('MX', 0.453)
    M = params.get('M', 752)
    
    # Distances from CG to axles
    Lf = MX * L  # distance from CG to front axle
    Lr = (1 - MX) * L  # distance from CG to rear axle
    
    # Aero parameters
    Cd = params.get('Cd', 0.8)
    Cl = params.get('Cl', 3.5)
    Area = params.get('Area', 1.2)
    CX = params.get('CX', 0.52)
    
    Lf_aero = CX * L  # aero center distance from front
    Lr_aero = (1 - CX) * L
    
    # Tire parameters
    tyre_radius = params.get('tyre_radius', 0.330)
    TYRE_LAT = params.get('TYRE_LAT', {})
    TYRE_LON = params.get('TYRE_LON', {})
    
    # Transmission
    final_drive = params.get('final_drive', 6.3)
    gear_ratios = params.get('GEAR_RATIOS', {})
    
    # Roll stiffness ratio
    K = params.get('K', 1.35)
    
    # Unpack state
    x, y, psi, vx, vy, r = state[0:6]
    vw_fl, vw_fr, vw_rl, vw_rr = state[6:10]
    engine_rpm, gear, throttle_input = state[10:13]
    
    # Unpack control
    delta_steer = u[0]  # steering angle (rad)
    throttle = u[1]     # throttle pedal (0-1)
    brake_pedal = u[2]  # brake pedal (0-1)
    
    g = 9.81
    
    # ======================= LOAD CALCULATIONS =======================
    # Calculate accelerations (use previous values or small estimate)
    ax = 0.0  # will be calculated from forces
    ay = vy * r + (vx * r)  # lateral accel from Coriolis term
    ay = np.clip(ay, -30, 30)  # clamp to realistic values
    
    # Longitudinal load transfer (moves load from rear to front under accel)
    Lf_val = Lf if Lf > 0 else L * 0.5
    Lr_val = Lr if Lr > 0 else L * 0.5
    
    # Base normal forces (static)
    Fz_rl_static = M * g * Lf_val / L / 2  # split equally between left and right
    Fz_rr_static = M * g * Lf_val / L / 2
    Fz_fl_static = M * g * Lr_val / L / 2
    Fz_fr_static = M * g * Lr_val / L / 2
    
    # Longitudinal load transfer (moves load to front during accel)
    delta_fz_lon = M * ax * H / L
    
    # Lateral load transfer (moves load outward in turn)
    delta_fz_lat_f, _ = calculate_load_transfer(ax, ay, M, H, TF, K)
    delta_fz_lat_r, _ = calculate_load_transfer(ax, ay, M, H, TR, 1.0)  # rear normalized
    
    # Apply load transfers to get current normal forces (in N)
    Fz_fl = Fz_fl_static + delta_fz_lon + delta_fz_lat_f  # front-left
    Fz_fr = Fz_fr_static + delta_fz_lon - delta_fz_lat_f  # front-right
    Fz_rl = Fz_rl_static - delta_fz_lon + delta_fz_lat_r  # rear-left
    Fz_rr = Fz_rr_static - delta_fz_lon - delta_fz_lat_r  # rear-right
    
    # Clamp to positive (wheels always in contact)
    Fz_fl = max(Fz_fl, 100)
    Fz_fr = max(Fz_fr, 100)
    Fz_rl = max(Fz_rl, 100)
    Fz_rr = max(Fz_rr, 100)
    
    # Convert to kN for tire formula
    Fz_fl_kN = Fz_fl / 1000
    Fz_fr_kN = Fz_fr / 1000
    Fz_rl_kN = Fz_rl / 1000
    Fz_rr_kN = Fz_rr / 1000
    
    # ======================= AERODYNAMIC FORCES =======================
    # Aero drag and downforce (quadratic in speed)
    v_sq = max(vx**2, 0)  # prevent negative values
    F_aero_drag = 0.5 * Cd * Area * v_sq  # drag force (negative, opposes motion)
    F_aero_downforce_f = 0.5 * Cl * Area * v_sq * 0.6  # 60% on front
    F_aero_downforce_r = 0.5 * Cl * Area * v_sq * 0.4  # 40% on rear
    
    # Clamp aero forces to prevent numerical issues
    F_aero_drag = np.clip(F_aero_drag, 0, 10000)
    F_aero_downforce_f = np.clip(F_aero_downforce_f, 0, 10000)
    F_aero_downforce_r = np.clip(F_aero_downforce_r, 0, 10000)
    
    # Aero adds to normal forces (downforce)
    Fz_fl += F_aero_downforce_f / 2
    Fz_fr += F_aero_downforce_f / 2
    Fz_rl += F_aero_downforce_r / 2
    Fz_rr += F_aero_downforce_r / 2
    
    # Aero drag acts on CG (creates longitudinal and small pitch moment)
    M_aero_pitch = F_aero_drag * H  # creates nose-down moment
    
    # ======================= TIRE SLIP ANGLES =======================
    # Velocity at each corner (accounting for yaw rate)
    # Front-left velocity components (accounting for steering and yaw rate)
    v_fl_x = vx + r * H  # Coriolis from yaw
    v_fl_y = vy - r * Lf  # tangential from rotation
    v_fl_steer_y = v_fl_y + delta_steer * v_fl_x  # steering effect
    
    # Front-right velocity components
    v_fr_x = vx + r * H
    v_fr_y = vy - r * Lf
    v_fr_steer_y = v_fr_y + delta_steer * v_fr_x
    
    # Rear-left velocity components
    v_rl_x = vx - r * H  # rear doesn't steer
    v_rl_y = vy + r * Lr
    
    # Rear-right velocity components
    v_rr_x = vx - r * H
    v_rr_y = vy + r * Lr
    
    # Slip angles (in degrees for magic formula), with clipping for safety
    # α = atan2(vy, vx)
    min_speed = 0.5  # threshold to avoid division issues
    alpha_fl = np.degrees(np.arctan2(v_fl_steer_y, max(abs(v_fl_x), min_speed)))
    alpha_fr = np.degrees(np.arctan2(v_fr_steer_y, max(abs(v_fr_x), min_speed)))
    alpha_rl = np.degrees(np.arctan2(v_rl_y, max(abs(v_rl_x), min_speed)))
    alpha_rr = np.degrees(np.arctan2(v_rr_y, max(abs(v_rr_x), min_speed)))
    
    # Clamp slip angles
    alpha_fl = np.clip(alpha_fl, -20, 20)
    alpha_fr = np.clip(alpha_fr, -20, 20)
    alpha_rl = np.clip(alpha_rl, -20, 20)
    alpha_rr = np.clip(alpha_rr, -20, 20)
    
    # ======================= LONGITUDINAL FORCES (FROM DRIVETRAIN) =======================
    # Automatic gear shifting (with downshift recovery + hysteresis).
    # Use rounded gear to avoid floor-bias from tiny UKF sigma-point perturbations.
    current_gear = int(np.clip(round(gear), 1, 8))
    new_gear = check_downshift(vx, current_gear, float(throttle), float(engine_rpm), params)
    if new_gear == current_gear:
        new_gear = check_upshift(vx, current_gear, params)
    
    # Engine torque from throttle and RPM
    throttle_smooth = 0.95 * throttle_input + 0.05 * throttle  # smooth input
    max_engine_torque = get_engine_torque(engine_rpm, params)
    engine_torque = max_engine_torque * throttle_smooth
    
    # Transmission: torque through gearbox
    if new_gear in gear_ratios:
        transmission_ratio = gear_ratios[new_gear] * final_drive
        wheel_torque = engine_torque * transmission_ratio  # applied torque at wheel
    else:
        wheel_torque = 0
    
    # Update engine RPM based on wheel speed (rear wheels)
    wheel_speed_rear = (vw_rl + vw_rr) / 2
    if new_gear in gear_ratios:
        engine_rpm_new = wheel_speed_rear / tyre_radius * gear_ratios[new_gear] * final_drive * 30 / np.pi
        engine_rpm = 0.9 * engine_rpm + 0.1 * max(engine_rpm_new, 1000)  # smooth update, min idle
    
    # Slip ratio: σ = (vwheel - vcar) / vcar
    # Rear wheel longitudinal forces from drivetrain
    v_car_rear = max(abs(vx), 0.5)  # threshold
    if v_car_rear < 0.5:
        slip_ratio_rl = 0.0
        slip_ratio_rr = 0.0
    else:
        slip_ratio_rl = (vw_rl - v_car_rear) / v_car_rear
        slip_ratio_rr = (vw_rr - v_car_rear) / v_car_rear
    
    # Clamp slip ratio
    slip_ratio_rl = np.clip(slip_ratio_rl, -1.0, 1.0)
    slip_ratio_rr = np.clip(slip_ratio_rr, -1.0, 1.0)
    
    # Longitudinal tire forces from magic formula
    Fx_rl = pacejka_magic_formula(slip_ratio_rl * 100, Fz_rl_kN, TYRE_LON, 'longitudinal')
    Fx_rr = pacejka_magic_formula(slip_ratio_rr * 100, Fz_rr_kN, TYRE_LON, 'longitudinal')

    # Apply driveline traction force to rear wheels so throttle generates propulsion.
    drive_force_per_wheel = wheel_torque / max(tyre_radius, 1e-3) / 2.0
    Fx_rl += drive_force_per_wheel
    Fx_rr += drive_force_per_wheel
    
    # Front wheels: assume zero slip (no engine power directly)
    Fx_fl = 0
    Fx_fr = 0
    
    # Apply brake (simple model: brake torque proportional to pedal)
    brake_torque = brake_pedal * M * 2000  # brake torque distribution to all wheels
    Fx_rl -= brake_torque / tyre_radius / 4  # distribute among wheels
    Fx_rr -= brake_torque / tyre_radius / 4
    Fx_fl -= brake_torque / tyre_radius / 4
    Fx_fr -= brake_torque / tyre_radius / 4
    
    # ======================= LATERAL TIRE FORCES =======================
    # Lateral tire forces using magic formula
    Fy_fl = pacejka_magic_formula(alpha_fl, Fz_fl_kN, TYRE_LAT, 'lateral')
    Fy_fr = pacejka_magic_formula(alpha_fr, Fz_fr_kN, TYRE_LAT, 'lateral')
    Fy_rl = pacejka_magic_formula(alpha_rl, Fz_rl_kN, TYRE_LAT, 'lateral')
    Fy_rr = pacejka_magic_formula(alpha_rr, Fz_rr_kN, TYRE_LAT, 'lateral')
    
    # ======================= GRIP LIMITING (FRICTION CIRCLE) =======================
    # Apply friction circle constraint at each wheel: combined demand must not exceed mu*Fz
    Fx_fl, Fy_fl, util_fl = check_grip_limit(Fx_fl, Fy_fl, Fz_fl_kN, TYRE_LAT, TYRE_LON)
    Fx_fr, Fy_fr, util_fr = check_grip_limit(Fx_fr, Fy_fr, Fz_fr_kN, TYRE_LAT, TYRE_LON)
    Fx_rl, Fy_rl, util_rl = check_grip_limit(Fx_rl, Fy_rl, Fz_rl_kN, TYRE_LAT, TYRE_LON)
    Fx_rr, Fy_rr, util_rr = check_grip_limit(Fx_rr, Fy_rr, Fz_rr_kN, TYRE_LAT, TYRE_LON)
    
    # Optional: log grip utilization for debugging
    # grip_util = {'FL': util_fl, 'FR': util_fr, 'RL': util_rl, 'RR': util_rr}
    
    # ======================= EQUATIONS OF MOTION =======================
    # Longitudinal (body frame)
    # Sum of longitudinal forces minus drag minus aero pitch effect
    Fx_total = Fx_fl + Fx_fr + Fx_rl + Fx_rr - F_aero_drag
    ax_dot = Fx_total / M - vy * r  # body frame: ax = Fx/m - vy*r
    
    # Lateral (body frame)
    # Sum of lateral forces from tires
    Fy_total = Fy_fl + Fy_fr + Fy_rl + Fy_rr
    ay_dot = Fy_total / M + vx * r  # body frame: ay = Fy/m + vx*r
    
    # Yaw moment
    # Tire moments
    M_z_tires = (Fy_fl + Fy_fr) * Lf * np.cos(delta_steer) + delta_steer * (Fx_fl + Fx_fr) * Lf \
                - (Fy_rl + Fy_rr) * Lr + (Fx_rl + Fx_rr) * 0  # rear wheels aligned
    
    # Moments from track offsets (left-right forces)
    M_z_lateral = (Fy_fl - Fy_fr) * TF/2 + (Fy_rl - Fy_rr) * TR/2
    
    # Yaw inertia (simplified: Iz ≈ M * L^2 / 12)
    Iz = M * (L**2 + (TF**2 + TR**2) / 4) / 12
    r_dot = (M_z_tires + M_z_lateral) / Iz
    
    # ======================= GLOBAL COORDINATES =======================
    x_dot = vx * np.cos(psi) - vy * np.sin(psi)
    y_dot = vx * np.sin(psi) + vy * np.cos(psi)
    psi_dot = r
    
    # ======================= WHEEL SPEEDS =======================
    # Simplified wheel speed dynamics (first-order lag to actual speed)
    wheel_accel_factor = 5.0  # time constant
    
    # Front wheels follow steering velocity
    v_fl_long = np.sqrt(np.clip(v_fl_x**2 + v_fl_steer_y**2, 0, 10000))
    v_fr_long = np.sqrt(np.clip(v_fr_x**2 + v_fr_steer_y**2, 0, 10000))
    vw_fl_dot = wheel_accel_factor * np.clip(v_fl_long - vw_fl, -50, 50)
    vw_fr_dot = wheel_accel_factor * np.clip(v_fr_long - vw_fr, -50, 50)
    
    # Rear wheels
    v_rl_long = np.sqrt(np.clip(v_rl_x**2 + v_rl_y**2, 0, 10000))
    v_rr_long = np.sqrt(np.clip(v_rr_x**2 + v_rr_y**2, 0, 10000))
    vw_rl_dot = wheel_accel_factor * np.clip(v_rl_long - vw_rl, -50, 50)
    vw_rr_dot = wheel_accel_factor * np.clip(v_rr_long - vw_rr, -50, 50)
    
    # ======================= ENGINE DYNAMICS =======================
    # RPM derivative (simple first-order)
    if wheel_speed_rear > 0.5:
        rpm_target = wheel_speed_rear / tyre_radius * gear_ratios.get(new_gear, 1.0) * final_drive * 30 / np.pi
    else:
        rpm_target = 3000.0  # idle
    rpm_target = np.clip(rpm_target, 1000, 15500)
    rpm_dot = 50 * np.clip(rpm_target - engine_rpm, -5000, 5000)  # time constant
    
    # ======================= GEAR LOGIC =======================
    gear_dot = 0  # gear doesn't change every step, only when upshift condition met
    if new_gear != gear:
        gear = new_gear
    
    throttle_input_dot = 5.0 * (throttle - throttle_input)  # smooth throttle input
    
    # ======================= INTEGRATION =======================
    # Update state using Euler integration with clipping for numerical stability
    state_dot = np.array([
        np.clip(x_dot, -100, 100),
        np.clip(y_dot, -100, 100),
        np.clip(psi_dot, -5, 5),
        np.clip(ax_dot, -30, 30),
        np.clip(ay_dot, -30, 30),
        np.clip(r_dot, -3, 3),
        np.clip(vw_fl_dot, -50, 50),
        np.clip(vw_fr_dot, -50, 50),
        np.clip(vw_rl_dot, -50, 50),
        np.clip(vw_rr_dot, -50, 50),
        np.clip(rpm_dot, -5000, 5000),
        gear_dot,
        throttle_input_dot
    ], dtype=np.float64)
    
    new_state = state + state_dot * dt
    
    # Final safety clamps
    new_state[0:2] = np.clip(new_state[0:2], -1e4, 1e4)  # position
    new_state[2] = np.remainder(new_state[2], 2*np.pi)   # wrap yaw angle
    new_state[3] = np.clip(new_state[3], -10, 95)        # vx [m/s]
    new_state[4] = np.clip(new_state[4], -35, 35)        # vy [m/s]
    new_state[5] = np.clip(new_state[5], -4, 4)          # yaw rate [rad/s]
    new_state[6:10] = np.clip(new_state[6:10], 0, 100)   # wheel speeds
    new_state[10] = np.clip(new_state[10], 1000, 15500)  # RPM
    new_state[11] = int(np.clip(new_gear, 1, 8))         # gear
    new_state[12] = np.clip(new_state[12], 0, 1)         # throttle
    
    return new_state

def generate_trajectory(n_steps: int = 10000, dt: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a trajectory using random controls.
    Returns (states, controls) as numpy arrays.
    """
    state = np.array([
    0., 0., 0.,
    np.random.uniform(40, 70),     # vx
    np.random.uniform(-3, 3),      # vy: moderate lateral velocity
    np.random.uniform(-0.2, 0.2),  # r: moderate yaw rate
    np.random.uniform(40, 70),     # wheel speeds
    np.random.uniform(40, 70),
    np.random.uniform(40, 70),
    np.random.uniform(40, 70),
    10000.,                        # engine_rpm: high range
    np.random.randint(5, 8),       # gear: 5-7
    np.random.uniform(0.5, 1.0)    # throttle: aggressive
]) 
    states   = [state]
    controls = []
 
    for _ in range(n_steps):
        u = np.array([
            np.random.uniform(-0.4, 0.4),   # steering angle (rad)
            np.random.uniform(0.0,1.0) ,   # throttle pedal (0-1)
            np.random.uniform(0.0,1.0)    # brake pedal (0-1)
        ])
        state = twin_track_model(state, u, dt)
        states.append(state)
        controls.append(u)
 
    return np.array(states), np.array(controls)

def tire_lateral_force(slip_angle_deg, Fz_kN, params):
    """
    Wrapper function to compute tire lateral force using Pacejka magic formula.
    
    Args:
        slip_angle_deg: slip angle in degrees
        Fz_kN: normal force in kN
        params: tire parameter dictionary (TYRE_LAT)
    
    Returns:
        Tire lateral force in Newtons
    """
    return pacejka_magic_formula(slip_angle_deg, Fz_kN, params, slip_type='lateral')


def tire_longitudinal_force(slip_ratio, Fz_kN, params):
    """
    Wrapper function to compute tire longitudinal force using Pacejka magic formula.
    
    Args:
        slip_ratio: slip ratio (dimensionless)
        Fz_kN: normal force in kN
        params: tire parameter dictionary (TYRE_LON)
    
    Returns:
        Tire longitudinal force in Newtons
    """
    return pacejka_magic_formula(slip_ratio, Fz_kN, params, slip_type='longitudinal')
