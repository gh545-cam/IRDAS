import numpy as np
L = 3.6 # wheelbase
TF = 1.58 # track front
TR = 1.42 # track rear
H = 0.295 # CoM height
MX = 0.453 # CoM distance from front axle as a percent of wheelbase
M = 752 # mass of car in kg

# Aerodynamic Parameters
Cd = 0.8 # Drag Coefficient
Cl = 3.5 # DF Coefficient
Area = 1.2 # Frontal area
CX = 0.52 # CoP distance from front axle as a percent of wheelbase


# Suspension Parameters
K = 1.35 # Front to rear roll stiffness ratio


# Transmission Parameters
final_drive = 6.3
ENGINE_RPM = np.array([
    3000, 4000, 5000, 6000, 7000, 8000, 9000,
    10000, 11000, 12000, 13000, 14000, 15000, 15500
])
ENGINE_TORQUE_NM = np.array([
    440, 500, 550, 540, 550, 550, 550,
    540, 525, 500, 480, 440, 390, 340
])
GEAR_RATIOS = {
    1: 3.15,
    2: 2.47,
    3: 1.96,
    4: 1.60,
    5: 1.33,
    6: 1.14,
    7: 0.98,
    8: 0.84,
}
UPSHIFT_SPEED_KPH = {
    1:  94.0,
    2: 119.9,
    3: 151.1,
    4: 185.1,
    5: 222.7,
    6: 259.8,
    7: 302.3,
}
tyre_radius    = 0.330

#Tyre parameters
TYRE_LAT = {
    'B':   12.0,    # stiffness factor
    'C':    1.9,    # shape factor
    'D':    1.8,    # peak factor (mu) — scales with Fz below
    'E':   -1.5,    # curvature factor (negative → more gradual falloff)

    # Load sensitivity on peak force: D_actual = (a1*Fz + a2) where Fz in kN
    'a1':  -0.10,   # degrades peak mu at high load (kN⁻¹)
    'a2':   2.05,   # base peak mu at zero load

    # Load sensitivity on cornering stiffness BCD: scales with Fz
    'BCD_a3': 110000.0,  # N/deg — peak cornering stiffness
    'BCD_a4':   3.0,  # load at peak stiffness (kN * 100 for scaling)
    'BCD_a5':    0.007, # decay rate
}

# --- Longitudinal (slip ratio) Pacejka coefficients ---
TYRE_LON = {
    'B':   12.0,    # stiffer than lateral (typical)
    'C':    1.7,
    'D':    1.85,   # slightly higher peak than lateral
    'E':   -2.0,

    # Load sensitivity (same structure as lateral)
    'a1':  -0.08,
    'a2':   2.1,
}