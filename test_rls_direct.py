#!/usr/bin/env python3
"""Direct test of RLS parameter estimation with known degradation."""
import numpy as np
from params import TYRE_LAT, TYRE_LON, M, Cd, Cl
from parameter_adapter import OnlineParameterAdapter
from twin_track import tire_lateral_force, tire_longitudinal_force

baseline = {
    'M': M, 'Cd': Cd, 'Cl': Cl,
    'TYRE_LAT': TYRE_LAT.copy(), 
    'TYRE_LON': TYRE_LON.copy(),
}

# Create adapter
adapter = OnlineParameterAdapter(baseline)

print("="*70)
print("RLS DEGRADATION TRACKING TEST")
print("="*70)
print(f"\nBaseline TYRE_LAT_a2: {baseline['TYRE_LAT']['a2']:.4f}")

# Simulate degradation: tire coefficient reduces over time
degradation_factors = [1.0, 0.95, 0.90, 0.85, 0.80]

for deg_idx, deg_factor in enumerate(degradation_factors):
    print(f"\n--- DEGRADATION SCENARIO: a2 multiplier = {deg_factor:.2f} ---")
    
    # Create degraded true parameters
    true_lat_a2 = baseline['TYRE_LAT']['a2'] * deg_factor
    true_lon_a2 = baseline['TYRE_LON']['a2'] * deg_factor
    
    # Reset adapter for this scenario
    adapter.reset_to_baseline()
    
    # Simulate several measurement cycles
    for cycle in range(30):
        # Operating point
        slip_angle = np.array([5.0, 5.0, 3.0, 3.0])  # degrees
        slip_ratio = np.array([0.02, 0.02])
        Fz_wheels = np.array([2.0, 2.0, 2.0, 2.0])  # kN
        
        # True tire force (with degraded a2)
        true_params = baseline['TYRE_LAT'].copy()
        true_params['a2'] = true_lat_a2
        F_true_lat = sum(tire_lateral_force(s, Fz_wheels[i], true_params) 
                        for i, s in enumerate(slip_angle)) / 4.0
        
        # Baseline predicted force
        F_pred_lat = sum(tire_lateral_force(s, Fz_wheels[i], baseline['TYRE_LAT']) 
                        for i, s in enumerate(slip_angle)) / 4.0
        
        # Force error
        force_error = F_true_lat - F_pred_lat
        
        # Debug on first cycle of first degradation
        if cycle == 0 and deg_idx == 0:
            print(f"  DEBUG Cycle 0: F_true={F_true_lat:.4f}, F_pred={F_pred_lat:.4f}, error={force_error:.4f}")
        
        # RLS update
        adapter.update_rls(slip_angle, slip_ratio, Fz_wheels, force_error, 0.0, debug=(cycle==0))
    
    params = adapter.get_current_params()
    rls_a2 = params['TYRE_LAT']['a2']
    change_pct = (rls_a2 - baseline['TYRE_LAT']['a2']) / baseline['TYRE_LAT']['a2'] * 100
    
    print(f"  True a2:  {true_lat_a2:.4f} ({(true_lat_a2/baseline['TYRE_LAT']['a2']-1)*100:.1f}%)")
    print(f"  RLS a2:   {rls_a2:.4f} ({change_pct:+.1f}%)")
    print(f"  Error:    {abs(rls_a2 - true_lat_a2):.4f}")

print("\n" + "="*70)
print("If RLS is working, the RLS a2 should decrease as tires degrade.")
print("="*70)
