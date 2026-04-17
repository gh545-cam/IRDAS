# State Reconstruction: Corrected Design Explanation

## The Core Question (and Answer)

**User Asked:** When the NN corrects wheel speeds, shouldn't we update Position/Yaw and RPM?

**Answer:** YES, but in DIFFERENT ways:
- **RPM**: YES, update it immediately (derived state must stay consistent)
- **Position/Yaw**: NO, leave them unchanged (KF handles naturally, avoid double-counting)

---

## Three State Categories

### 1. Dynamics States [3,4,5,6,7,8,9] - CORRECTED BY NN
```
[vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr]
Action: NN predicts residuals → Apply corrections
```
These are what the NN learns. Corrections are applied directly.

### 2. Derived States [10] - UPDATED FOR CONSISTENCY  
```
[rpm]
Formula: rpm = mean(wheel_speeds) * gear_ratio * final_drive
Action: RECOMPUTED after wheel speeds are corrected
```

**Why?** RPM is not a free variable—it's deterministically computed from wheel speeds and gear. If we correct wheel speeds but don't update RPM:
- Wheel speeds change: 5.0 → 5.01 m/s
- RPM stays: 3000 rev/min
- **Physical inconsistency!** The rotational speeds don't match.

**Solution:** Always recompute: `rpm = mean(corrected_wheel_speeds) * gear_ratio * final_drive`

### 3. Independent States [0,1,2] - NOT UPDATED
```
[x, y, psi]
Action: LEFT UNCHANGED from KF prediction
```

**Why NOT update immediately?** This is the KEY INSIGHT:

If we manually integrate corrected velocities:
```python
# WRONG (double-counting):
Step k:   x_pred = x + vx_old * dt          # KF prediction
          vx → vx_corrected (NN)
          x = x + vx_corrected * dt         # Manual integration (FIRST)
          
Step k+1: x_next = x + vx_corrected * dt   # KF prediction (SECOND)
          ↑ Now vx_corrected effect applied TWICE!
```

Correct approach:
```python
# CORRECT (no double-counting):
Step k:   x_pred = x + vx_old * dt          # KF prediction
          vx → vx_corrected (NN)
          x stays unchanged                  # No manual integration
          
Step k+1: x_next = x + vx_corrected * dt   # KF prediction (ONCE)
          ↑ vx_corrected effect applied correctly, one time
```

The **Kalman filter naturally handles integration of corrected velocities** in the next prediction step. Updating manually would count the change twice.

### 4. Discrete/Control States [11,12] - NEVER MODIFIED
```
gear [11]: Discrete decision from check_upshift() logic
throttle [12]: Control input from driver/controller
Action: Never modified by NN
```

---

## The Kalman Filter Handles It Right

After we correct velocities and leave position/yaw unchanged, the **Kalman filter update step** brings everything into consistency:

```
After NN Correction (Step k):
  - Corrected: vx, vy, r, wheel_speeds
  - Unchanged: x, y, psi (inconsistent with velocities)

Kalman Filter Update (Measurement Fusion):
  - Gets GPS measurement of x, y position
  - Adjusts estimated x, y toward measurement
  - Accounts for corrected velocities
  - Everything becomes consistent!

Result: Complete, consistent state estimate for step k+1
```

---

## State Reconstruction Algorithm

```python
def _apply_nn_residual_correction(full_state, control, correction_scale=0.1):
    corrected_state = full_state.copy()
    
    # ===== CORRECT DYNAMICS STATES =====
    dynamics_indices = [3, 4, 5, 6, 7, 8, 9]
    residual = NN(full_state[dynamics_indices], control)
    corrected_state[dynamics_indices] += residual * correction_scale
    
    # ===== UPDATE DERIVED STATES =====
    # Recompute RPM from corrected wheel speeds for consistency
    wheel_speeds = corrected_state[[6, 7, 8, 9]]
    mean_speed = np.mean(wheel_speeds)
    current_gear = int(corrected_state[11])
    
    gear_ratio = GEAR_RATIOS[current_gear]
    final_drive = 6.3
    rpm = mean_speed * gear_ratio * final_drive * 60 / (2*np.pi*0.33)
    corrected_state[10] = np.clip(rpm, 0, 15500)
    
    # ===== LEAVE INDEPENDENT STATES =====
    # corrected_state[0:3] unchanged → [x, y, psi]
    # Reason: KF will integrate in next prediction step (avoid double-counting)
    
    # ===== LEAVE DISCRETE/CONTROL STATES =====
    # corrected_state[11] unchanged → gear
    # corrected_state[12] unchanged → throttle
    
    return corrected_state  # Full 13-state, properly reconstructed
```

---

## Visual: State Flow During One Time Step

```
START OF STEP:
┌─────────────────────────────────────┐
│ State estimate from previous step   │
│ [x, y, psi, vx, vy, r, ...]        │
└────────────────┬────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
    ↓                         ↓
PHASE 1: KF PREDICTION     PHASE 2: NN CORRECTION
(baseline model)           (residual learning)

Predicts next state:       Corrects dynamics:
- Integrates velocities    - Extract [vx,vy,r,whl_sp]
  → Updated x, y, psi      - Predict residuals
- Computes new RPM         - Apply corrections
- Advances all states      - Recompute RPM
  (old RPM)                - Leave x,y,psi unchanged
                           - Return corrected full state
                
[x_pred, y_pred, psi_pred,  [x_pred, y_pred, psi_pred,
 vx_pred, vy_pred, ...]  →   vx_corr, vy_corr, ...
                             rpm_corr, ...]
                 │
                 ↓
    ┌───────────────────────┐
    │ PHASE 3: KF UPDATE    │
    │ (measurement fusion)  │
    │                       │
    │ - Gets measurement    │
    │   (GPS, IMU)          │
    │ - Fuses with          │
    │   prediction          │
    │ - Adjusts state       │
    │   toward measurement  │
    │ - Brings x,y,psi into │
    │   consistency         │
    └───────────┬───────────┘
                │
                ↓
         ┌────────────────┐
         │ FINAL STATE    │
         │ at step k      │
         │ (consistent    │
         │  and fused)    │
         └────────────────┘
                │
    (becomes input for step k+1)
```

---

## Example: Wheel Speed Correction

**Initial state (after KF prediction):**
```
vw_fl=5.0, vw_fr=5.0, vw_rl=5.0, vw_rr=5.0
x=100, y=50, psi=0.5
rpm=3000
```

**NN predicts residuals:**
```
Δvw_fl=+0.1, Δvw_fr=+0.1, Δvw_rl=+0.1, Δvw_rr=+0.1
(wheel speeds slightly higher than baseline predicted)
```

**After applying corrections (scale=0.1):**
```
vw_fl=5.01, vw_fr=5.01, vw_rl=5.01, vw_rr=5.01
x=100 (UNCHANGED)         ← Will be updated by KF in next step
y=50 (UNCHANGED)
psi=0.5 (UNCHANGED)

RPM recomputed:
  mean_speed = 5.01
  rpm = 5.01 * gear_ratio * final_drive ≈ 3006 (UPDATED)
```

**What's consistent?**
- ✓ Wheel speeds match each other
- ✓ RPM matches wheel speeds
- ✓ Position/yaw will be brought consistent by KF measurement fusion next

**What happened to the old position/yaw mismatch?**
- The KF update step (measuring position) will correct it
- In the next prediction, we'll integrate the corrected velocities naturally

---

## Guarantees

✓ **No double-counting** of velocity corrections
✓ **Derived states consistent** (RPM matches wheel speeds)
✓ **Full 13-state maintained** at all times
✓ **Measurement fusion works properly** (can correct any inconsistency)
✓ **Numerically stable** (small correction scale, clipped RPM)

---

## Files Modified

1. **irdas_main.py**:
   - `_apply_nn_residual_correction()`: Lines 173-245
     - Now includes RPM recomputation
     - Clear documentation of why each state is handled differently
   
2. **test_state_reconstruction.py**:
   - Now validates derived state updates (RPM)
   - Checks independent states unchanged
   - Verifies dynamics states corrected

3. **Documentation**:
   - COMPREHENSIVE_CODE_DOCUMENTATION.md: Updated explanation
   - STATE_RECONSTRUCTION_SUMMARY.md: Updated with state categories
   - FULL_STATE_RECONSTRUCTION_REFERENCE.md: This file

---

## Key Takeaway

**Position and yaw should NOT be manually updated because:**
1. The Kalman filter naturally integrates corrected velocities
2. Manually updating would double-count the corrections
3. The KF update step (with measurements) brings everything into consistency
4. This is how Kalman filtering is supposed to work!

**RPM SHOULD be updated because:**
1. It's derived from wheel speeds and gear
2. It must stay consistent with its source data
3. Not updating would create physical inconsistency
4. It's not an integration—it's a direct computation

This design leverages the Kalman filter's strength (optimal fusion) while maintaining physical consistency (derived states).
