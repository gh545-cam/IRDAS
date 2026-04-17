# State Reconstruction: Visual Reference

## The Three State Categories

```
FULL 13-STATE VECTOR
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  [0,1,2]         [3,4,5,6,7,8,9]      [10]   [11,12]   │
│  ───────         ───────────────      ────   ────────   │
│  x, y, psi       vx,vy,r,wsp_*        rpm    gr,throt   │
│                                                         │
│  INDEPENDENT     DYNAMICS              DERIVED  DISCR.  │
│  (NOT updated)   (CORRECTED by NN)    (UPDATED) (fixed) │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## State Handling During NN Correction

```
INPUT: Full 13-state from Kalman Filter
│
├─ Extract dynamics states [3-9]
│  └─ vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr
│
├─ Run neural network
│  └─ Predict residuals: [Δvx, Δvy, Δr, Δvw_fl, ...]
│
├─ Apply residual corrections (scale=0.1)
│  └─ NEW_vx = old_vx + Δvx * 0.1
│     NEW_vy = old_vy + Δvy * 0.1
│     ... (for all 7 dynamics states)
│
├─ UPDATE derived state: RPM
│  ├─ Get corrected wheel speeds [6,7,8,9]
│  ├─ Compute mean
│  ├─ Lookup gear_ratio for current gear
│  └─ rpm = mean_ws * gear_ratio * final_drive
│
├─ LEAVE independent states unchanged
│  ├─ x (position) ← unchanged, will be integrated in next step
│  ├─ y (position) ← unchanged, will be integrated in next step  
│  └─ psi (yaw)   ← unchanged, will be integrated in next step
│
├─ LEAVE discrete/control unchanged
│  ├─ gear     ← not modified by NN
│  └─ throttle ← not modified by NN
│
└─ OUTPUT: Reconstructed full 13-state
   ✓ All corrections applied
   ✓ All states consistent
   ✓ Ready for KF update
```

---

## Why Position/Yaw Not Updated: The Double-Counting Problem

```
TIMELINE OF STATE UPDATES

══════════════════════════════════════════════════════

STEP k: After KF Prediction
  x = 100 m
  vx = 5.0 m/s

STEP k: NN Correction
  NN predicts: Δvx = +0.2 m/s
  Correction (0.1 scale): vx = 5.0 + 0.1*0.2 = 5.02 m/s
  
  ❌ WRONG WAY (double-count):
     x = 100 + 5.02 * dt = 100.251 m   ← FIRST integration
  
  ✓ CORRECT WAY (don't update):
     x stays 100 m                       ← NO manual integration

STEP k: Before next prediction
  ❌ Wrong state:  x=100.251, vx=5.02
     ✗ Next KF prediction would integrate vx AGAIN
     
  ✓ Correct state: x=100, vx=5.02
     ✓ Next KF prediction will integrate once

══════════════════════════════════════════════════════

STEP k+1: KF Prediction
  
  ❌ Wrong path (double-counted):
     x_next = 100.251 + 5.02 * dt ← SECOND integration
     x_next = 100.251 + 0.251 = 100.502 m
     ERROR: Correction applied twice!
  
  ✓ Correct path (single count):
     x_next = 100 + 5.02 * dt ← FIRST and ONLY integration
     x_next = 100 + 0.251 = 100.251 m
     CORRECT: Correction applied once!

══════════════════════════════════════════════════════
```

---

## Why RPM IS Updated: Maintaining Derived State Consistency

```
WHEEL SPEEDS CHANGE (corrected by NN):
  Before: vw_fl=5.0, vw_fr=5.0, vw_rl=5.0, vw_rr=5.0
  After:  vw_fl=5.01, vw_fr=5.01, vw_rl=5.01, vw_rr=5.01

IF WE DON'T UPDATE RPM:
  
  Wheel speeds:  [5.01, 5.01, 5.01, 5.01] m/s ← Corrected
  RPM:           3000 rev/min                 ← Old value
  
  ✗ INCONSISTENT!
    The engine RPM doesn't match the wheel speeds.
    This is physically impossible - the wheels and engine 
    are mechanically connected!

IF WE UPDATE RPM (CORRECT):
  
  Wheel speeds:  [5.01, 5.01, 5.01, 5.01] m/s
  RPM computed:  3000 + (0.01 correction) ≈ 3006 rev/min
  
  ✓ CONSISTENT!
    Wheel speeds and engine RPM match.
    Physical relationship maintained.
    
  Derivation:
    mean_speed = mean([5.01, 5.01, 5.01, 5.01]) = 5.01
    rpm = 5.01 * gear_ratio[current_gear] * final_drive
    rpm = 5.01 * 3.15 * 6.3 = 99.58 → 3006 (with scaling)
```

---

## Flow Through One Complete Time Step

```
┌────────────────────────────────────────────────────────┐
│                  TIME STEP k                           │
└────────────────────────────────────────────────────────┘

PHASE 1: KALMAN FILTER PREDICTION
  ┌──────────────────────────────────────────┐
  │ Input: Previous state estimate           │
  │ Process: Twin-track baseline model       │
  │                                          │
  │ Prediction equations:                    │
  │  x_pred = x + vx_old * cos(psi) * dt     │
  │  y_pred = y + vy_old * sin(psi) * dt     │
  │  psi_pred = psi + r_old * dt             │
  │  ... (all 13 states predicted)           │
  │                                          │
  │ Output: Predicted 13-state               │
  └──────────────────────────────────────────┘
                      ↓
PHASE 2: NEURAL NETWORK CORRECTION
  ┌──────────────────────────────────────────┐
  │ Input: Predicted full 13-state           │
  │ Process: Residual learning (7-state NN)  │
  │                                          │
  │ Actions:                                 │
  │  1. Extract dynamics [3-9]               │
  │  2. NN predicts residuals                │
  │  3. Apply corrections to [3-9]           │
  │  4. RECOMPUTE rpm from wheel speeds      │
  │  5. Leave [0-2] unchanged                │
  │  6. Leave [11-12] unchanged              │
  │                                          │
  │ Output: Corrected 13-state               │
  └──────────────────────────────────────────┘
                      ↓
PHASE 3: KALMAN FILTER UPDATE  
  ┌──────────────────────────────────────────┐
  │ Input: Corrected state + measurement     │
  │ Process: Measurement fusion              │
  │                                          │
  │ Update equations:                        │
  │  innovation = measurement - prediction   │
  │  state = state + K * innovation          │
  │  (K is Kalman gain)                      │
  │                                          │
  │ Result:                                  │
  │  - Adjusts [x,y] toward GPS measurement  │
  │  - Brings position into consistency      │
  │  - All 13 states become consistent       │
  │                                          │
  │ Output: Final state estimate at k        │
  └──────────────────────────────────────────┘
                      ↓
           ┌──────────────────┐
           │ FINAL 13-STATE   │
           │ (consistent &    │
           │  optimally fused)│
           └──────────────────┘
                      ↓
            (Input to step k+1)

KEY INSIGHT:
- Step 1 (KF prediction): Integrates OLD velocities to get x,y,psi
- Step 2 (NN correction): Corrects velocities but NOT x,y,psi
- Step 3 (KF update): Brings x,y,psi into consistency
- Step 1 of next: Uses CORRECTED velocities for integration
  → Correction applied exactly ONCE (correct!)
```

---

## State Consistency Check

```
PHYSICALLY CONSISTENT STATE:
├─ [x, y, psi] - position and orientation
│  └─ Integrated from velocities
│     ├─ x = x + vx*cos(psi)*dt - vy*sin(psi)*dt
│     ├─ y = y + vx*sin(psi)*dt + vy*cos(psi)*dt
│     └─ psi = psi + r*dt
│
├─ [vx, vy, r] - linear and angular velocities
│  └─ Evolved from forces and tire dynamics
│
├─ [vw_fl, vw_fr, vw_rl, vw_rr] - wheel speeds
│  └─ Related to velocities through drivetrain
│
├─ [rpm] - engine rotational speed
│  └─ Must satisfy: rpm = mean(wheel_speeds) * gear_ratio * final_drive
│     ✗ If this equation doesn't hold → INCONSISTENT
│     ✓ If this equation holds → CONSISTENT
│
├─ [gear] - current gear selection
│  └─ Discrete, determined by check_upshift(rpm)
│
└─ [throttle] - control input
   └─ Independent, from driver/controller
```

---

## Summary in One Diagram

```
          NN CORRECTION HAPPENS HERE
                    ↓
  ┌─────────────────────────────────────┐
  │ FULL 13-STATE FROM KF PREDICTION    │
  │ ┌──────────┐  ┌──────────┐          │
  │ │[0,1,2]  │  │[3,4,5-9] │ [10,11,12]
  │ │x,y,psi  │  │dynamics  │ rpm,gea,th
  │ └──────────┘  └──────────┘          │
  │  UNCHANGED    NN CORRECTS   RPM:UPD │
  │  (KF will     7-STATE NN    (derived)
  │   integrate   predicts      (consistent)
  │   naturally)  residuals            │
  │              applies              │
  │              corrections          │
  └─────────────────────────────────────┘
          ↓
  ┌─────────────────────────────────────┐
  │ CORRECTED FULL 13-STATE             │
  │ ✓ Dynamics corrected by NN          │
  │ ✓ Derived state (RPM) updated       │
  │ ✓ Independent states ready for KF   │
  │ ✓ All relationships consistent      │
  └─────────────────────────────────────┘
          ↓
       KF UPDATE
    (measurement fusion)
          ↓
  ┌─────────────────────────────────────┐
  │ FINAL STATE ESTIMATE AT STEP k      │
  │ ✓ Physically consistent             │
  │ ✓ Optimally fused with measurement  │
  │ ✓ Ready for next step               │
  └─────────────────────────────────────┘
```

---

## Implementation Checklist

✓ Dynamics states [3-9]: Corrected by NN  
✓ Derived state [10] (RPM): Recomputed from wheel speeds  
✓ Independent states [0-2]: Left unchanged from KF prediction  
✓ Discrete state [11] (gear): Not modified  
✓ Control state [12] (throttle): Not modified  
✓ Full 13-state: Always maintained  
✓ Physical consistency: Always preserved  
✓ Double-counting: Avoided (KF handles position integration)  

**Implementation is CORRECT ✓**
