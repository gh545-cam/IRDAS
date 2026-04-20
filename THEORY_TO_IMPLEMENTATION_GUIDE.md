# IRDAS Theory to Implementation Guide

## Who This Is For
This guide is for an aspiring engineer who wants to understand both:
- Why this system works (theory)
- How each theory appears in this repository (implementation)

You can start with basic calculus and linear algebra knowledge. By the end, you should be comfortable with modern model-based plus learning-based estimation and adaptation for vehicle dynamics.

## How To Use This Guide
Read in order once. Then revisit sections while reading code.

Suggested path:
1. Physical vehicle model and state-space thinking
2. Estimation with the Unscented Kalman Filter
3. Residual learning with recurrent neural networks
4. Online adaptation with recursive least squares
5. System integration and race-level behavior

---

## 1. Why IRDAS Exists
Any digital twin of a race car starts wrong in small but important ways:
- Tire behavior changes with wear and temperature
- Aero coefficients differ from nominal values
- Vehicle mass changes due to fuel burn
- Sensors are noisy

A pure physics model drifts away from reality.
A pure neural network is data-hungry and can violate physics.

IRDAS solves this by combining:
- Physics-based dynamics model
- Bayesian state estimation (UKF)
- Learned residual correction (GRU network)
- Online parameter adaptation (RLS)

This is a hybrid architecture, often called gray-box modeling.

---

## 2. Core Modeling Language: State Space
The system uses a discrete-time nonlinear state-space form:

x_(k+1) = f(x_k, u_k, theta_k) + w_k
z_k = h(x_k) + v_k

Where:
- x is the hidden vehicle state
- u is control input (steering, throttle, brake)
- theta are model parameters (tire and aero coefficients)
- z is sensor measurement
- w and v are process and measurement noise

### 2.1 Vehicle State Used Here
In the current implementation, the UKF uses 14 states:
- [x, y, psi, vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr, rpm, gear, throttle, mass]

This is implemented in:
- kalman_filter.py (class ExtendedKalmanFilter)

Important note:
- get_state() returns the first 13 states for backward compatibility
- get_augmented_state() returns all 14 including mass

### 2.2 Why Mass Is A State
Mass is not constant in racing. Fuel burn makes it time-varying.
If mass is hidden but changing, then acceleration predictions are biased.

Mass dynamics are modeled simply but effectively:
- m_(k+1) = m_k - fuel_flow_k * dt + noise

Mass is also measured by a noisy sensor channel.
So mass is estimated by fusing process model plus sensor updates, exactly like other states.

This is implemented in:
- kalman_filter.py predict(): mass process update via fuel flow
- kalman_filter.py update(): mass included in 10-dimensional measurement
- sensors.py measurement_function_fixed(): returns mass channel when state has 14 elements
- irdas_main.py step(): appends mass_sensor to measurement vector

---

## 3. Physics Backbone: Twin-Track Dynamics
The function twin_track_model is the deterministic transition model f(.).
It includes:
- Tire forces using Pacejka-type magic formula
- Load transfer (longitudinal and lateral)
- Aero drag and downforce
- Drivetrain and gear behavior
- Wheel speed and RPM dynamics

This lives in:
- twin_track.py

### 3.1 Why This Matters
The physics model gives structure and extrapolation:
- It behaves reasonably in regimes with little data
- It constrains learning to corrections rather than full dynamics replacement

### 3.2 Tire Force Principle
The tire model approximates a nonlinear map:
- Slip angle or slip ratio -> force

It captures two realities:
- Near-zero slip is near-linear
- At larger slip, saturation appears

This makes handling, braking, and traction behavior physically plausible.

---

## 4. Sensor Modeling
A filter is only as good as its sensor model h(.).
The repository models realistic noisy sensors:
- Position
- IMU accelerations
- Yaw rate
- Velocity
- RPM
- Wheel speed
- Fuel flow and mass

This lives in:
- sensors.py (SensorSimulator and measurement_function_fixed)

### 4.1 IMU Consistency
The code computes body-frame accelerations using velocity derivatives and Coriolis terms.
This is important because incorrect measurement equations can destabilize filtering even when noise tuning is good.

---

## 5. Unscented Kalman Filter (UKF) Theory
The UKF is used because dynamics are nonlinear.

### 5.1 EKF vs UKF Intuition
EKF linearizes f and h using Jacobians.
UKF avoids Jacobians and instead propagates sigma points through full nonlinear functions.

Why UKF is attractive here:
- Strong nonlinearities in tires and drivetrain
- Avoids deriving and maintaining large Jacobian code

### 5.2 UKF Steps In This Repo
At each time step:
1. Build sigma points from mean and covariance
2. Propagate each sigma point through twin_track_model and mass process
3. Recombine into predicted mean and covariance
4. Predict measurement for each sigma point
5. Build innovation covariance and cross covariance
6. Compute Kalman gain
7. Update state and covariance

Implemented in:
- kalman_filter.py
  - _generate_sigma_points
  - _mean_from_sigma
  - _cov_from_sigma
  - predict
  - update

### 5.3 Practical Numerical Stability Choices
The code includes key stability features:
- Covariance symmetrization
- Jittered Cholesky with fallback eigen decomposition
- Condition number check with pseudoinverse fallback
- Angle wrapping for yaw
- State clamping for physically valid ranges

These are not academic details. They are what make filters survive long runs.

### 5.4 Discrete-State Handling (Gear)
Gear is discrete while UKF is continuous.
A common failure mode is accidental drift of gear during updates.
This causes huge RPM errors.

The repository protects against this by:
- Postprocessing gear as rounded and clipped integer
- Preventing measurement update from changing gear and throttle
- Collapsing covariance rows and columns for discrete/control states

Key implementation:
- kalman_filter.py update() and _state_postprocess()
- twin_track.py uses rounded gear when checking shifts

### 5.5 UKF Weights and Scaling Parameters (Practical)
The UKF uses scaling parameters $(\alpha,\beta,\kappa)$ to build sigma points.
For state dimension $n$:

$$
\lambda = \alpha^2(n+\kappa) - n,
\quad
c = n + \lambda
$$

Weights are:

$$
W_0^{(m)} = \frac{\lambda}{c},
\quad
W_0^{(c)} = \frac{\lambda}{c} + (1-\alpha^2+\beta),
$$

$$
W_i^{(m)} = W_i^{(c)} = \frac{1}{2c},\; i=1..2n
$$

Interpretation:
- $\alpha$ controls sigma-point spread around mean.
- $\beta$ encodes distribution prior (commonly 2 for near-Gaussian priors).
- $\kappa$ is an additional spread-shape knob.

In this repository, these values are set conservatively to prioritize numerical robustness over aggressive nonlinear capture.

### 5.6 Innovation Diagnostics You Should Monitor
The innovation is:

$$
\nu_k = z_k - \hat z_k
$$

with innovation covariance:

$$
S_k = P_{zz,k} + R
$$

Useful checks:
1. Innovation mean near zero over time (bias check).
2. Innovation variance roughly matching predicted $S_k$ scale (consistency check).
3. Spikes correlated with maneuvers indicate either model mismatch or poor process noise design.

A scalar consistency metric is normalized innovation squared (NIS):

$$
NIS_k = \nu_k^T S_k^{-1}\nu_k
$$

Persistently high NIS means filter confidence is too optimistic or measurement model is mismatched.

### 5.7 Q and R Tuning Strategy in This Codebase
For this project, tune in this order:
1. Fix structural mismatches first (wrong measurement mapping, gear handling bugs).
2. Tune measurement noise $R$ to sensor realism.
3. Tune process noise $Q$ for transient tracking.

Heuristic:
- If estimate is noisy but unbiased: increase $R$ or reduce $Q$.
- If estimate is smooth but lags: increase relevant entries in $Q$.
- If filter diverges in aggressive maneuvers: verify nonlinear model consistency before making $Q$ huge.

### 5.8 Why Mass In UKF Improves More Than Just Mass
Adding mass state improves multiple channels indirectly:
- acceleration prediction quality ($a=F/m$)
- load transfer consistency
- tire force prediction quality
- residual-learning target quality (smaller systematic baseline bias)

So mass estimation is not an isolated feature; it stabilizes the whole estimation-learning-adaptation loop.

### 5.9 Worked Mini Example (One UKF Update)
Suppose at one step:
- predicted speed: $\hat v_x^- = 42.0$ m/s
- measured speed: $z_{vx}=41.2$ m/s
- innovation variance: $S=0.64$ (m/s)$^2$
- cross covariance: $P_{xv}=0.32$ (m/s)$^2$

Then Kalman gain for this channel is:

$$
K = \frac{P_{xv}}{S} = \frac{0.32}{0.64} = 0.5
$$

Innovation:

$$
\nu = z_{vx} - \hat v_x^- = 41.2 - 42.0 = -0.8
$$

Updated speed estimate:

$$
\hat v_x^+ = 42.0 + 0.5(-0.8) = 41.6\;\text{m/s}
$$

Interpretation:
- the filter moved halfway toward the sensor because model and sensor confidence were comparable.
- if $S$ were larger (less trust in measurement), correction would be smaller.

---

## 6. Residual Learning Theory
The residual network learns:
residual = real_next - baseline_next

So corrected prediction is:
baseline + residual_prediction

This is usually easier than learning full dynamics directly because the baseline already explains most behavior.

### 6.1 Reduced-State Learning
The residual model learns only the 7 dynamic states:
- [vx, vy, r, vw_fl, vw_fr, vw_rl, vw_rr]

This avoids overfitting to kinematic states that are naturally integrated by the filter.

### 6.2 Recurrent Memory
The network is GRU-based, so it has temporal memory.
That helps with effects that are not fully visible in one frame, such as gradual tire condition changes.

Implementation:
- residual_network.py (RecurrentResidualDynamicsNetwork)
- residual_network.py (ResidualDynamicsLearner)

### 6.3 Physics-Informed Regularization
Two regularizers are used during training:
1. Traction-inspired magnitude regularizer
   - Penalizes excessive residual acceleration magnitude
2. Left-right symmetry regularizer
   - Encourages mirrored behavior under mirrored state and steering

These regularizers bias the network toward physically plausible corrections.

Implementation:
- residual_network.py
  - _traction_circle_regularizer
  - _symmetry_regularizer

### 6.4 Stateful Inference
During runtime, the learner can keep recurrent hidden state.
This lets the model carry context across steps.

Implementation:
- residual_network.py reset_stateful_inference
- irdas_main.py use_stateful_nn_inference flag and step flow

### 6.5 Exact Residual Target Definition and Timing
Residual learning is sensitive to index alignment.
The intended target is one-step-ahead model error:

$$
r_k = x_{k+1}^{real} - x_{k+1}^{baseline}
$$

not same-step difference $x_k^{real} - x_k^{baseline}$.

If indexing is wrong, training can look numerically stable but produce poor runtime corrections.

### 6.6 Why The 7-State Output Is A Strong Design Choice
Only dynamic states are corrected:
- $v_x, v_y, r, v_{w,fl}, v_{w,fr}, v_{w,rl}, v_{w,rr}$

Benefits:
1. Keeps NN focused on dynamics where model mismatch is largest.
2. Avoids corrupting kinematic integration of $(x,y,\psi)$.
3. Improves long-horizon stability because fewer channels are directly learned.

### 6.7 Loss Decomposition With Intuition
Conceptually:

$$
\mathcal L = \underbrace{\|\hat r-r\|_2^2}_{data} + \lambda_t\underbrace{\mathcal L_{traction}}_{physical\ magnitude} + \lambda_s\underbrace{\mathcal L_{sym}}_{mirror\ consistency}
$$

Interpretation:
- Data loss makes predictions accurate on observed residuals.
- Traction regularizer discourages unrealistic residual impulses.
- Symmetry regularizer embeds structural prior from left-right vehicle symmetry.

This is a "weak physics prior" approach: physics guides learning without over-constraining it.

### 6.8 Stateful Memory Management in Long Runs
RNN hidden state improves temporal context but can accumulate stale bias.
Practical reset points:
1. start of new race/lap regime
2. after pit-stop-like discontinuities
3. after severe model mismatch events

This repository already provides mechanisms to reset stateful inference when needed.

### 6.9 Failure Modes Specific To Residual Learning
1. Residual overreach:
- symptom: corrected state oscillates more than baseline.
- action: reduce residual scale or increase regularization.

2. Residual underuse:
- symptom: corrections too small, baseline bias remains.
- action: reduce regularization or improve mismatch diversity in training.

3. Regime locking:
- symptom: good on straights, poor in corners (or vice versa).
- action: rebalance dataset excitation and sequence sampling.

### 6.10 Worked Mini Example (Residual Correction)
Assume baseline one-step prediction for dynamic 7-state is:

$$
x_{dyn,base}^{k+1} = [40.0,\;0.6,\;0.04,\;39.5,\;39.7,\;40.4,\;40.2]
$$

NN predicts residual:

$$
\hat r = [-0.8,\;0.1,\;0.02,\;-0.5,\;-0.4,\;-0.6,\;-0.5]
$$

Corrected prediction:

$$
x_{dyn,corr}^{k+1} = x_{dyn,base}^{k+1} + \hat r
$$

So corrected $v_x$ is $39.2$ m/s instead of $40.0$ m/s, indicating model over-predicted longitudinal speed in this regime.

If this repeats near a specific corner, the NN has learned a systematic local mismatch (for example unmodeled grip loss).

---

## 7. Online Parameter Adaptation Theory (RLS)
Residual learning corrects state prediction quickly.
RLS adaptation updates interpretable physical parameters over time.

### 7.1 What Is Adapted
Current adapter focuses on:
- TYRE_LAT a2
- TYRE_LON a2
- Cd
- Cl

Implementation:
- parameter_adapter.py adaptive_param_names

### 7.2 Why RLS
Recursive least squares is online and efficient.
It updates parameter estimates with each new sample without re-solving a full batch problem.

In simplified scalar form for each parameter:
- gain depends on covariance and regressor sensitivity
- parameter deviation updated by gain times prediction error
- covariance updated with forgetting factor

### 7.3 Sensitivity Regressor
The adapter computes a local sensitivity-like regressor based on tire model behavior at current operating point.
This ties adaptation to physically meaningful signals.

Implementation:
- parameter_adapter.py _compute_pacejka_regressor
- parameter_adapter.py update_rls

### 7.4 Bound Constraints
Parameters are clipped to realistic ranges to avoid runaway adaptation.
This is essential in noisy online settings.

Implementation:
- parameter_adapter.py param_bounds and _apply_bounds

### 7.5 Vector RLS Equations (Full Form)
For parameter deviation estimate $\hat{\delta\theta}$ and regressor $\phi_k$:

$$
K_k = \frac{P_{k-1}\phi_k}{\lambda + \phi_k^T P_{k-1}\phi_k}
$$

$$
e_k = y_k - \phi_k^T\hat{\delta\theta}_{k-1}
$$

$$
\hat{\delta\theta}_k = \hat{\delta\theta}_{k-1} + K_k e_k
$$

$$
P_k = \frac{1}{\lambda}(P_{k-1} - K_k\phi_k^T P_{k-1})
$$

where $\lambda\in(0,1]$ is forgetting factor.

### 7.6 Forgetting Factor and Adaptation Speed
Practical meaning:
- $\lambda \to 1$: slower but smoother adaptation.
- smaller $\lambda$: faster but noisier adaptation.

In racing applications, a moderate forgetting factor is usually best because tire and aero drift are gradual, not instantaneous.

### 7.7 Identifiability and Excitation
Not every parameter is observable in every regime.
Examples:
- Cd is weakly observable at low speed.
- tire peak parameters are weakly observable in gentle straight-line cruising.

This means adaptation confidence should be linked to excitation quality.
When excitation is low, updates should be conservative.

### 7.8 Why Bounds Are Essential (Not Optional)
Without bounds, noisy regressors can drive physically impossible parameters.
That can destabilize both predictor and controller.

Projection step:

$$
	heta_i \leftarrow \min(\theta_i^{max},\max(\theta_i^{min},\theta_i))
$$

This keeps adaptation safe and interpretable.

### 7.9 Coupling Between UKF, NN, and RLS
RLS should absorb slow structural drift.
UKF and NN should absorb fast effects.

If RLS is too aggressive, it can chase high-frequency noise already handled by UKF/NN, causing parameter chatter.
If too slow, structural bias remains and residual network is forced to compensate for everything.

Balanced multi-timescale adaptation is a core design principle in this repository.

### 7.10 Worked Mini Example (Scalar RLS Step)
Take one scalar parameter channel (e.g. deviation in tyre peak coefficient).
Suppose:

$$
P_{k-1}=10,\; \phi_k=0.2,\; \lambda=0.98,\; \hat d\theta_{k-1}=0.05,\; y_k=0.20
$$

Gain:

$$
k_k = \frac{P_{k-1}\phi_k}{\lambda + \phi_k^2P_{k-1}}
= \frac{10\cdot0.2}{0.98 + 0.04\cdot10}
= \frac{2.0}{1.38}
= 1.449
$$

Prediction error:

$$
e_k = y_k - \phi_k\hat d\theta_{k-1}
= 0.20 - 0.2\cdot0.05
= 0.19
$$

Updated estimate:

$$
\hat d\theta_k = 0.05 + 1.449\cdot0.19 \approx 0.325
$$

Interpretation:
- strong correction happens because prior covariance $P$ was large (high uncertainty).
- as $P$ shrinks over time, updates become smoother and less reactive.

---

## 8. System Integration: One Step In IRDAS
At a high level, each step in irdas_main.py does:
1. Update true mass using fuel flow and step the real simulator
2. Generate noisy measurement including mass sensor
3. UKF predict with control and measured fuel flow
4. Apply NN residual correction to dynamic states
5. UKF update with measurement
6. Compute model error signals
7. Run RLS adaptation
8. Log everything

This is implemented in:
- irdas_main.py step()

### 8.1 State Reconstruction After NN Correction
NN only modifies the 7 dynamic states.
Other states are handled carefully:
- Position and yaw are left for filter integration
- RPM is recomputed from wheel-speed consistency logic when correction is applied
- Gear and throttle remain controlled/discrete

Implementation:
- irdas_main.py _apply_nn_residual_correction

---

## 9. Data Generation and Training
Training is self-supervised through simulated mismatch:
- Baseline model trajectory
- Real simulator trajectory with perturbed parameters
- Residual target is difference between them

Implementation:
- simulator.py RealVehicleSimulator
- irdas_main.py pretrain_neural_network
- train_test.py training and evaluation scripts

This is a common digital-twin strategy when real labeled data is scarce.

---

## 10. Race-Level Scenario and Strategy
The new race script demonstrates long-horizon behavior under realistic drift:
- Sensor noise
- True vehicle mismatch vs baseline twin
- Fuel burn and mass evolution
- Tire degradation across laps
- Two pit stops with tire resets
- Parameter adaptation and lap-time comparison plots

Implementation:
- full_race_sim.py
  - run_full_race
  - save_visualizations

Outputs include:
- Real vs predicted lap times
- Fuel used per lap
- Tire health trajectory
- True vs estimated mass
- Adapted tire and aero parameter trends
- Residual RMS over laps

---

## 11. Mathematical Cheat Sheet
### 11.1 State Estimation Error
A common metric printed in scripts is RMS error for each state i:

RMS_i = sqrt( (1/N) * sum_k (x_true_i(k) - x_est_i(k))^2 )

For RPM, this is an error magnitude in RPM units, not average RPM value.

### 11.2 Innovation
Innovation at time k:

nu_k = z_k - z_hat_k

The innovation is the information content of the measurement relative to prediction.

### 11.3 Kalman Update Core
x_plus = x_minus + K * nu
P_plus = P_minus - K S K^T

Where S is innovation covariance.

### 11.4 Residual Learning
x_corrected_next ~= f_baseline(x,u) + g_theta(x,u)

Where g_theta is the neural residual model.

### 11.5 Online Adaptation View
theta_{k+1} = theta_k + Delta(theta)_k

Delta(theta)_k is computed from sensitivity, error signal, covariance, and forgetting factor.

---

## 12. How To Read The Code Efficiently
Suggested order for a first deep pass:
1. params.py
2. twin_track.py
3. sensors.py
4. kalman_filter.py
5. simulator.py
6. residual_network.py
7. parameter_adapter.py
8. irdas_main.py
9. train_test.py
10. full_race_sim.py

Then run:
- train_test.py --mode quick
- full_race_sim.py

And inspect generated plots while stepping through the related functions.

---

## 13. Common Pitfalls and Engineering Lessons
1. Nonlinear filters fail more often from model-measurement inconsistency than from noise values alone.
2. Discrete states inside continuous filters need explicit handling.
3. Pure data-driven correction without physical priors can overfit and destabilize long horizons.
4. Online adaptation needs bounds and conservative covariance handling.
5. A stable architecture is usually layered: physics model + filter + learned residual + slow parameter adaptation.

---

## 14. Where To Go Next (Advanced)
1. Add bias states for key sensors and estimate them online.
2. Move from fixed Q and R to adaptive covariance estimation.
3. Add uncertainty estimates to residual network outputs.
4. Extend RLS to additional tire shape parameters with identifiability checks.
5. Compare UKF vs square-root UKF for stronger numerical robustness.
6. Validate on real telemetry and include domain randomization during pretraining.

---

## 15. Final Mental Model
Think of IRDAS as a layered control and estimation stack:
- Physics gives structure
- UKF gives trustworthy state estimates from noisy data
- Residual network learns fast unmodeled effects
- RLS updates interpretable parameters over longer horizons

Together, this creates a digital twin that learns during operation while staying physically grounded.

---

## 16. Symbol Table (Beginner Friendly)
If equations feel dense, this table helps you decode them quickly.

- x_k: state vector at discrete time step k
- u_k: control input at time step k
- z_k: measurement vector at time step k
- f(.): state transition function (vehicle dynamics model)
- h(.): measurement function (sensor model)
- w_k: process noise (model uncertainty)
- v_k: measurement noise (sensor uncertainty)
- P: state covariance matrix (uncertainty on x)
- Q: process noise covariance matrix
- R: measurement noise covariance matrix
- S: innovation covariance
- K: Kalman gain
- nu_k: innovation, nu_k = z_k - z_hat_k
- theta: parameter vector (tire and aero coefficients)
- dt: simulation time step

Interpretation tip:
- Big P means “we are uncertain about state estimate.”
- Big R means “we trust sensors less.”
- Big Q means “we trust model propagation less.”

---

## 17. UKF Mathematics (Step by Step)
This section gives the complete UKF flow used by the repository.

### 17.1 Sigma Point Construction
Given state dimension n, mean x, covariance P, and UKF scaling lambda:

$$
\chi_0 = x,
\qquad
\chi_i = x + \left(\sqrt{(n+\lambda)P}\right)_i,
\qquad
\chi_{i+n} = x - \left(\sqrt{(n+\lambda)P}\right)_i
$$

for i = 1,...,n.

In code:
- kalman_filter.py function _generate_sigma_points

### 17.2 Predict Step
Each sigma point is propagated through nonlinear dynamics:

$$
\chi_i^- = f(\chi_i, u_k)
$$

Predicted mean:

$$
x_k^- = \sum_{i=0}^{2n} W_i^{(m)} \chi_i^-
$$

Predicted covariance:

$$
P_k^- = \sum_{i=0}^{2n} W_i^{(c)} (\chi_i^- - x_k^-)(\chi_i^- - x_k^-)^T + Q
$$

In code:
- kalman_filter.py functions _mean_from_sigma and _cov_from_sigma
- kalman_filter.py function predict

### 17.3 Measurement Prediction
Project propagated sigma points into measurement space:

$$
\gamma_i = h(\chi_i^-)
$$

Predicted measurement:

$$
\hat z_k = \sum_{i=0}^{2n} W_i^{(m)} \gamma_i
$$

Innovation covariance:

$$
S_k = \sum_{i=0}^{2n} W_i^{(c)} (\gamma_i-\hat z_k)(\gamma_i-\hat z_k)^T + R
$$

Cross covariance:

$$
P_{xz} = \sum_{i=0}^{2n} W_i^{(c)} (\chi_i^- - x_k^-)(\gamma_i-\hat z_k)^T
$$

In code:
- kalman_filter.py function update

### 17.4 Kalman Update

$$
K_k = P_{xz} S_k^{-1}
$$

$$
\nu_k = z_k - \hat z_k
$$

$$
x_k^+ = x_k^- + K_k \nu_k
$$

$$
P_k^+ = P_k^- - K_k S_k K_k^T
$$

In code:
- kalman_filter.py function update

Practical note:
- The code uses condition checks and pseudo-inverse fallbacks for numerically hard S.

---

## 18. Mass-State Augmentation (Detailed)
Mass is explicitly estimated as a state component, not just a side variable.

### 18.1 Process Model for Mass

$$
m_{k+1} = m_k - \dot m_{fuel,k} \cdot dt + w_m
$$

where:
- m_k is current mass estimate
- dot m_fuel is measured fuel flow
- w_m is process uncertainty for mass dynamics

### 18.2 Measurement Model for Mass

$$
z_{m,k} = m_k + v_m
$$

Mass update then happens inside the same UKF innovation/update pipeline as other states.

Code mapping:
- mass state index and dimensions in kalman_filter.py
- mass channel appended in irdas_main.py step
- mass measurement support in sensors.py measurement_function_fixed

Engineering benefit:
- Better acceleration prediction under fuel burn
- More consistent load transfer and tire normal force estimates

---

## 19. Tire and Traction-Circle Mechanics (Detailed)
The vehicle can request longitudinal and lateral forces simultaneously, but tire capacity is finite.

### 19.1 Combined Grip Constraint
The implementation enforces a friction-circle style cap:

$$
\sqrt{F_x^2 + F_y^2} \le F_{max},
\qquad
F_{max} = \mu_{peak} F_z
$$

If requested force exceeds F_max, both components are scaled by the same factor:

$$
\alpha = \frac{F_{max}}{\sqrt{F_x^2+F_y^2}},
\qquad
F_x \leftarrow \alpha F_x,
\qquad
F_y \leftarrow \alpha F_y
$$

### 19.2 Important Integration Order
In this repository, the order is:
1. Compute tire force demand
2. Add driveline traction force on rear wheels
3. Apply traction-circle limiting

This ensures throttle added force is still grip-limited.

Code mapping:
- twin_track.py drive_force_per_wheel addition
- twin_track.py check_grip_limit calls immediately after

---

## 20. Residual Learning Objective (Detailed)
Residual target construction:

$$
r_k = x^{real}_{k+1} - x^{baseline}_{k+1}
$$

Network prediction:

$$
\hat r_k = g_\phi(s_k, u_k)
$$

Corrected prediction:

$$
\hat x^{corr}_{k+1} = x^{baseline}_{k+1} + \hat r_k
$$

Loss conceptually:

$$
\mathcal L = \mathcal L_{data} + \lambda_t \mathcal L_{traction} + \lambda_s \mathcal L_{symmetry}
$$

where:
- data term is MSE between residual target and residual prediction
- traction term penalizes unrealistic residual acceleration magnitudes
- symmetry term penalizes left-right mirror inconsistency

Code mapping:
- residual_network.py train_epoch and validate
- residual_network.py _traction_circle_regularizer
- residual_network.py _symmetry_regularizer

---

## 21. RLS Update Equations (Detailed)
The adapter estimates deviation parameters dtheta online.

Scalar RLS form (per parameter channel):

$$
k_k = \frac{P_{k-1}\phi_k}{\lambda + \phi_k^2 P_{k-1}}
$$

$$
e_k = y_k - \phi_k \hat d\theta_{k-1}
$$

$$
\hat d\theta_k = \hat d\theta_{k-1} + k_k e_k
$$

$$
P_k = \frac{1}{\lambda}(1-k_k\phi_k)P_{k-1}
$$

where:
- phi is sensitivity-like regressor
- y is force/speed error signal
- lambda is forgetting factor

Code mapping:
- parameter_adapter.py update_rls

Practical tuning intuition:
- Lower lambda: faster adaptation, more noise sensitivity
- Higher lambda: smoother, slower adaptation

---

## 22. Controller and Drivetrain Coupling in Race Sim
Race speed behavior depends on both driver logic and drivetrain logic.

Current race controller includes:
- Feedforward + PI speed control
- Brake deadband and coast band
- Relaunch mode at very low speed

Drivetrain logic includes:
- Upshift thresholds
- Downshift recovery with hysteresis
- Target-speed-based kickdown floor
- Idle torque + low-RPM torque ramp

Code mapping:
- full_race_sim.py build_driver_input
- twin_track.py check_upshift
- twin_track.py check_downshift
- twin_track.py get_engine_torque

---

## 23. Interpreting Key Plots (How To Read Them)

### 23.1 Lap Time Real vs Predicted
- Gap shrinking over laps suggests adaptation is helping.
- Gap spikes near pit stops are expected due to regime reset.

### 23.2 Mass True vs Estimated
- Smooth tracking with small lag is healthy.
- Strong oscillation means measurement/process noise tuning issue.

### 23.3 Telemetry (Throttle/Brake/Gear/RPM/vx)
- Frequent throttle-brake toggling indicates controller chattering.
- RPM low with high gear and high throttle indicates lugging/kickdown issue.
- vx collapse with sustained throttle often points to traction/model mismatch.

---

## 24. Beginner FAQ

Q1: Is RPM RMS error the average RPM?
- No. It is RMS of estimation error in RPM units.

Q2: Why not just train one large neural net for everything?
- You lose interpretability and often robustness. Physics + filter + residuals usually generalize better in sparse-data, safety-critical settings.

Q3: Why can more aggressive adaptation hurt?
- Because noise can be mistaken as parameter drift; fast adaptation may chase noise.

Q4: Why can a model still fail even if equations are correct?
- Numerical conditioning, discrete-state handling, and controller interactions can still destabilize behavior.

---

## 25. Suggested Learning Progression (Practical)
If you are new, use this sequence:
1. Run quick tests and inspect outputs.
2. Read twin_track.py and identify each force term.
3. Read kalman_filter.py and map each UKF equation to code blocks.
4. Read residual_network.py and understand residual target generation.
5. Read parameter_adapter.py and track one parameter over time.
6. Run full_race_sim.py and interpret telemetry + adaptation plots.

If you can explain every line in the loop below, you are at solid intermediate level:
- control generation
- true-state step
- measurement generation
- UKF predict/update
- residual correction
- RLS adaptation

---

## 26. Full Vehicle Dynamics Derivation (Body Frame)
This section writes the dynamics in the same style used by the twin-track model.

### 26.1 Translational Dynamics
In body frame:

$$
m(\dot v_x - r v_y) = \sum F_x
$$

$$
m(\dot v_y + r v_x) = \sum F_y
$$

Rearranged:

$$
\dot v_x = \frac{\sum F_x}{m} + r v_y
$$

$$
\dot v_y = \frac{\sum F_y}{m} - r v_x
$$

Depending on sign convention, equivalent forms may appear with opposite signs on Coriolis terms. The important part is consistency across dynamics and measurement equations.

### 26.2 Yaw Dynamics

$$
I_z \dot r = \sum M_z
$$

where yaw moment combines front and rear lateral force lever arms and left-right track moment differences.

Code mapping:
- twin_track.py yaw-moment and Iz blocks

### 26.3 Kinematic Position Update

$$
\dot x = v_x \cos\psi - v_y \sin\psi
$$

$$
\dot y = v_x \sin\psi + v_y \cos\psi
$$

$$
\dot\psi = r
$$

Code mapping:
- twin_track.py global coordinate section

---

## 27. Tire Slip and Force Equations

### 27.1 Slip Angle (Concept)
For each wheel, slip angle is approximately:

$$
\alpha = \arctan\left(\frac{v_y^{wheel}}{\max(|v_x^{wheel}|, v_{min})}\right)
$$

The denominator floor avoids division issues at very low speed.

### 27.2 Slip Ratio (Driven Wheels)

$$
\sigma = \frac{v_{wheel} - v_{car}}{\max(|v_{car}|, v_{min})}
$$

### 27.3 Magic Formula Shape
The code uses a simplified Pacejka-style form:

$$
F = D \sin\left(C\arctan\left(Bs - E(Bs - \arctan(Bs))\right)\right)
$$

with load-sensitive peak term roughly of form:

$$
D = a_1 F_z + a_2
$$

This gives realistic nonlinear saturation and load sensitivity.

Code mapping:
- twin_track.py pacejka_magic_formula

---

## 28. Load Transfer and Aero Equations

### 28.1 Lateral Load Transfer (Simplified)

$$
\Delta F_{z,lat} \approx \frac{m a_y h_{cg}}{track}
$$

### 28.2 Longitudinal Load Transfer (Simplified)

$$
\Delta F_{z,lon} \approx \frac{m a_x h_{cg}}{L}
$$

### 28.3 Aerodynamic Forces

$$
F_{drag} = \tfrac{1}{2} \rho C_d A v^2
$$

$$
F_{down} = \tfrac{1}{2} \rho C_l A v^2
$$

In this simulator, constants are absorbed into tuned coefficients so the implemented expression may omit explicit air density.

Code mapping:
- twin_track.py load calculations
- twin_track.py aerodynamic force block

---

## 29. Traction-Circle Limiting (Detailed Math)

The combined force demand per tire is:

$$
F_{req} = \sqrt{F_x^2 + F_y^2}
$$

Peak available force:

$$
F_{max} = \mu_{peak} F_z
$$

If $F_{req} > F_{max}$, scale both components:

$$
\kappa = \frac{F_{max}}{F_{req}}
$$

$$
F_x \leftarrow \kappa F_x, \quad F_y \leftarrow \kappa F_y
$$

Important engineering detail in this repository:
- driveline throttle force is added first
- then combined grip limiting is applied

So throttle cannot generate physically impossible acceleration when lateral demand is already high.

Code mapping:
- twin_track.py driveline add and subsequent check_grip_limit calls

---

## 30. Driver Model Equations and Rationale
The race driver now follows a preview-based speed planner plus PI longitudinal controller.

### 30.1 Speed Planner
Base segment target speed is adjusted by tire health and fuel fraction, then curvature-limited:

$$
v_{target} = \min\left(v_{segment}\cdot s_{tyre}\cdot s_{fuel}, \sqrt{\frac{a_{lat,max}}{|\kappa|+\epsilon}}\right)
$$

### 30.2 PI Longitudinal Control

$$
e_v = v_{target} - v_x
$$

$$
I_v \leftarrow \text{clip}(I_v + e_v dt, I_{min}, I_{max})
$$

$$
u = u_{ff}(v_{target}) + K_p e_v + K_i I_v
$$

Throttle and brake are shaped from u with deadband/coast/relaunch logic.

### 30.3 Anti-Windup
When braking dominates, positive integral accumulation is bled off to avoid delayed over-throttle after decel.

Code mapping:
- full_race_sim.py DriverModel class

---

## 31. Start and End Lap Consistency
Discrete-time simulation can create lap-time inconsistency if the final step overshoots the finish line and full dt is counted.

This repository now uses fractional final-step integration:

If current lap progress is p, step progress is Delta p, and lap length is L:

$$
	ext{if } p + \Delta p > L,\quad \eta = \frac{L-p}{\Delta p}
$$

Use effective step:

$$
dt_{eff} = \eta dt
$$

and integrate only with $dt_{eff}$ for lap completion time and distance.

This makes lap boundary timing physically consistent and avoids artificial timing bias.

Code mapping:
- full_race_sim.py run_full_race step_frac and dt_eff logic

---

## 32. Why Speed Can Still Drop to Zero Sometimes
Even with improved control and gearing, zero-speed episodes can occur if several effects coincide:
- high drag and braking demand in a poor regime
- insufficient traction under combined force saturation
- gear transitions lagging rapid speed changes
- conservative lap safety constraints (timeouts)

This is not one bug; it is a coupled closed-loop behavior.

Recommended diagnosis order:
1. Check target speed vs actual speed traces.
2. Check throttle and brake overlap.
3. Check gear and rpm around the collapse event.
4. Check tire utilization and traction limiting status.

---

## 33. Practical Tuning Recipe (With Equations)

### 33.1 Longitudinal Controller
Start with modest gains:

$$
u = u_{ff} + K_p e_v + K_i I_v
$$

Increase $K_p$ until response is fast but non-oscillatory, then add small $K_i$ to remove steady-state error.

### 33.2 Deadband
Introduce a speed-error deadband around zero to avoid throttle-brake chatter:

$$
|e_v| < e_{db} \Rightarrow \text{coast mode}
$$

### 33.3 Relaunch Mode
If speed remains too low while target is high:

$$
v_x < v_{stall},\ v_{target} > v_{resume}\Rightarrow throttle \ge throttle_{min},\ brake=0
$$

### 33.4 Shift Policy
Use hysteresis and minimum dwell-like behavior to avoid hunting:
- upshift at higher threshold
- downshift at lower threshold

and add target-speed floor constraints for minimum usable gear under high demand.

---

## 34. Uncertainty Interpretation Beyond RMS
RMS is useful but incomplete. Also inspect:
- innovation mean (bias indicator)
- innovation covariance consistency
- normalized innovation squared trends

Normalized innovation squared (single step concept):

$$
NIS_k = \nu_k^T S_k^{-1} \nu_k
$$

Persistent high NIS suggests mismatch in noise tuning or model structure.

---

## 35. End-to-End Learning Objective View
Think of IRDAS optimization at two timescales:

Fast timescale:
- UKF updates state every step using measurements

Medium timescale:
- residual network applies dynamic correction every step

Slow timescale:
- RLS drifts interpretable parameters over many steps

This multi-timescale structure is a key reason the architecture remains stable while adapting online.

---

## 36. UKF Deep Dive (As Detailed As A Textbook Chapter)
This section expands the Kalman part to tutorial-level depth and ties each concept to this codebase.

### 36.1 Why UKF Is The Right Choice Here
Your dynamics are strongly nonlinear because of:
- Pacejka tire equations
- friction-circle clipping
- gear shifts and driveline coupling
- aerodynamic terms scaling with $v^2$

An EKF needs Jacobians of all of that. Those Jacobians are complex, fragile, and expensive to maintain.
UKF avoids analytic Jacobians and approximates nonlinear uncertainty propagation via deterministic sigma points.

### 36.2 Sigma-Point Geometry Intuition
For $n$ states, UKF picks $2n+1$ deterministic points around the mean.
These points are not random; they are chosen so mean and covariance are matched up to second order.

Define:

$$
\lambda = \alpha^2(n+\kappa)-n
$$

Sigma points:

$$
\chi_0 = x,
\quad
\chi_i = x + \left(\sqrt{(n+\lambda)P}\right)_i,
\quad
\chi_{i+n} = x - \left(\sqrt{(n+\lambda)P}\right)_i
$$

Weights:

$$
W_0^{(m)} = \frac{\lambda}{n+\lambda},
\quad
W_0^{(c)} = \frac{\lambda}{n+\lambda} + (1-\alpha^2+\beta),
$$

$$
W_i^{(m)}=W_i^{(c)}=\frac{1}{2(n+\lambda)}, \; i=1..2n
$$

Practical parameter meaning:
- $\alpha$: spread of sigma points (small means close to mean)
- $\beta$: prior knowledge of distribution (2 is common for Gaussian)
- $\kappa$: secondary scaling knob

### 36.3 Prediction in This Repository
Each sigma point goes through full nonlinear transition $f$:

$$
\chi_i^- = f(\chi_i, u_k, \theta_k)
$$

Then mean/covariance are reconstructed:

$$
x_k^- = \sum_i W_i^{(m)}\chi_i^-
$$

$$
P_k^- = \sum_i W_i^{(c)}(\chi_i^- - x_k^-)(\chi_i^- - x_k^-)^T + Q
$$

Code mapping:
- kalman_filter.py `_generate_sigma_points`
- kalman_filter.py `predict`
- twin_track.py `twin_track_model`

### 36.4 Measurement Update in This Repository
Project sigma points to measurement space:

$$
\gamma_i = h(\chi_i^-)
$$

Predict measurement:

$$
\hat z_k = \sum_i W_i^{(m)}\gamma_i
$$

Innovation covariance and cross-covariance:

$$
S_k = \sum_i W_i^{(c)}(\gamma_i-\hat z_k)(\gamma_i-\hat z_k)^T + R
$$

$$
P_{xz} = \sum_i W_i^{(c)}(\chi_i^- - x_k^-)(\gamma_i-\hat z_k)^T
$$

Gain and correction:

$$
K_k = P_{xz}S_k^{-1}
$$

$$
\nu_k = z_k - \hat z_k
$$

$$
x_k^+ = x_k^- + K_k\nu_k,
\quad
P_k^+ = P_k^- - K_kS_kK_k^T
$$

Code mapping:
- kalman_filter.py `update`
- sensors.py `measurement_function_fixed`

### 36.5 Numerical Conditioning Tricks You Are Using
Real UKF code fails without these safeguards:
- covariance symmetrization: $P\leftarrow \frac{1}{2}(P+P^T)$
- jitter on diagonal before Cholesky
- eigen fallback if Cholesky fails
- condition-number checks on $S$
- pseudo-inverse fallback for near-singular $S$

These are implementation-critical and are already present in your filter.

### 36.6 Discrete/Continuous Hybrid State Treatment
Gear is discrete; throttle command memory is quasi-discrete/actuator-like.
If allowed to drift continuously in update math, gear can become 3.47, 3.12, etc., then cast behavior introduces bias.

Your repository solves this with:
- rounded/clipped gear projection after updates
- constrained covariance for discrete/control coordinates

This is why RPM RMS dropped massively after the earlier fix.

### 36.7 Mass-Augmented UKF as Joint Estimation
You are performing joint state estimation where one component is a slowly varying parameter-like quantity (mass).
This is often more stable than trying to infer mass indirectly from residuals alone.

Mass process channel:

$$
m_{k+1} = m_k - \dot m_{fuel} dt + w_m
$$

Mass measurement channel:

$$
z_{m,k} = m_k + v_m
$$

Together they provide observability leverage from both dynamics and sensor.

### 36.8 UKF Tuning Heuristics (Practical)
1. If estimate is noisy but unbiased, decrease Q or increase R carefully.
2. If estimate lags badly during maneuvers, increase Q in dynamic states.
3. If innovations are consistently biased, fix model/measurement mismatch first, then tune noise.
4. Keep process noise on mass small but nonzero so fuel-flow mismatch can be absorbed.

---

## 37. Neural Residual Model Deep Dive
This section explains the NN part in the style of a modern hybrid-modeling tutorial.

### 37.1 Why Learn Residuals Instead of Full Dynamics
Learning full vehicle dynamics from scratch asks the network to learn:
- rigid-body mechanics
- tire saturation geometry
- aero scaling
- drivetrain behavior

That is data-hungry and brittle outside training distribution.
Residual learning reduces the target complexity:

$$
r_k = x^{real}_{k+1} - x^{base}_{k+1}
$$

$$
\hat x_{k+1} = x^{base}_{k+1} + \hat r_k
$$

So the network only learns what physics misses.

### 37.2 Sequence Modeling Rationale (GRU)
A single frame does not fully reveal latent effects like tire condition drift and transient coupling.
GRU hidden state acts as compressed memory:

$$
h_k = \mathrm{GRU}(s_k, h_{k-1})
$$

$$
\hat r_k = W_o h_k + b_o
$$

This captures temporal patterns without huge model size.

Code mapping:
- residual_network.py `RecurrentResidualDynamicsNetwork`
- residual_network.py `ResidualDynamicsLearner`

### 37.3 Input/Output Design Choices
Input uses state/control context for prediction of next-step residual.
Output only covers 7 dynamic states:
- $v_x, v_y, r, v_{w,fl}, v_{w,fr}, v_{w,rl}, v_{w,rr}$

Why not position and yaw?
- position and yaw are integrative kinematics and better handled by model + filter
- including them in residual target often injects unnecessary drift

### 37.4 Loss Engineering in This Repository
Base supervised term:

$$
\mathcal L_{data} = \|\hat r_k-r_k\|_2^2
$$

Physics-inspired regularizers:

$$
\mathcal L = \mathcal L_{data} + \lambda_t\mathcal L_{traction} + \lambda_s\mathcal L_{sym}
$$

Where:
- $\mathcal L_{traction}$ discourages unrealistically large residual accelerations
- $\mathcal L_{sym}$ encourages left-right mirror consistency

These terms are key to long-horizon stability.

### 37.5 Stateful Inference and Drift Control
During rollout, hidden state is preserved across steps for temporal continuity.
At race boundaries or domain shifts, hidden state can be reset to avoid stale context contamination.

Code mapping:
- residual_network.py `reset_stateful_inference`
- irdas_main.py `reset_nn_memory` flow

### 37.6 Where NN Correction Enters The Stack
Sequence is:
1. UKF predict gives baseline next estimate
2. NN predicts residual on 7-state slice
3. Corrected dynamic states are written back
4. UKF measurement update fuses corrected prediction with sensors

This means NN is not replacing Kalman update; it improves prior before measurement fusion.

### 37.7 Normalization and Scale Discipline
Residual targets for RPM/wheel dynamics can differ in scale from velocity channels.
Training remains stable when:
- features are normalized consistently
- output channels are balanced in loss weighting
- clipping guards prevent exploding corrections

If you later see one channel dominate loss, add per-channel weighting.

### 37.8 Failure Modes and Mitigations
1. Overconfident residuals: reduce output scale or increase regularization.
2. Mode collapse to near-zero residuals: reduce regularization and improve mismatch diversity.
3. Temporal instability: shorter sequence length or hidden-state reset policy.
4. Out-of-distribution behavior: domain randomization in pretraining scenarios.

---

## 38. RLS Parameter Adaptation Deep Dive
This section details the math behind the online parameter adaptation loop.

### 38.1 Problem Setup
Let $\theta$ be physical parameters to adapt (here: tire peaks and aero coefficients).
Represent deviation from nominal as $\delta\theta$.

Linearized local relationship:

$$
y_k \approx \phi_k^T\delta\theta_k + \epsilon_k
$$

where:
- $y_k$ is measurable model error surrogate
- $\phi_k$ is sensitivity regressor

### 38.2 Vector RLS Equations

$$
K_k = \frac{P_{k-1}\phi_k}{\lambda + \phi_k^T P_{k-1}\phi_k}
$$

$$
e_k = y_k - \phi_k^T\hat{\delta\theta}_{k-1}
$$

$$
\hat{\delta\theta}_k = \hat{\delta\theta}_{k-1} + K_k e_k
$$

$$
P_k = \frac{1}{\lambda}\left(P_{k-1} - K_k\phi_k^TP_{k-1}\right)
$$

$\lambda\in(0,1]$ is forgetting factor.

### 38.3 Forgetting Factor Meaning
- $\lambda\approx1$: slow, stable, less noise-sensitive
- smaller $\lambda$: fast tracking, more noise amplification

In racing, moderate forgetting is needed because tire/aero effective behavior does drift, but not instantly.

### 38.4 Regressor Construction In This Repo
Your adapter computes a Pacejka-informed regressor tied to current operating point.
This gives physically meaningful adaptation directions instead of arbitrary gradients.

Code mapping:
- parameter_adapter.py `_compute_pacejka_regressor`

### 38.5 Bounding and Projection
After update, parameters are clipped/projected into feasible intervals:

$$
	heta_i \leftarrow \min(\theta_i^{max},\max(\theta_i^{min},\theta_i))
$$

This prevents runaway adaptation under noisy or temporarily uninformative signals.

Code mapping:
- parameter_adapter.py `param_bounds`
- parameter_adapter.py `_apply_bounds`

### 38.6 Identifiability Caveats
Not all parameters are equally observable in all maneuvers.
Examples:
- Cd is poorly excited at low speed
- tire peak parameters are weakly observed in straight-line cruising

Implication:
- adaptation confidence should depend on excitation conditions
- if needed, gate updates by speed/lat-acc thresholds

### 38.7 Coupling With UKF and NN
The three adaptation layers operate at different frequencies:
- UKF: every step, immediate sensor fusion
- NN residual: every step, fast correction of unmodeled dynamics
- RLS: slower structural drift tracking

RLS should not chase high-frequency noise; UKF/NN should absorb that.

### 38.8 Practical RLS Tuning Workflow
1. Freeze NN and verify RLS converges directionally on synthetic mismatch.
2. Enable NN and reduce RLS aggressiveness if both over-correct simultaneously.
3. Validate bounds are wide enough for true mismatch but narrow enough for safety.
4. Inspect lap-wise parameter trajectories for smooth monotonic trends (unless pit resets).

---

## 39. Code-Level Trace: One Full Step Across Kalman, NN, RLS
Use this mental trace when reading implementation:

1. Real vehicle advances with control in simulator.
2. Sensor simulator returns noisy measurement vector (including mass sensor).
3. UKF predict propagates sigma points through nonlinear twin-track + mass process.
4. NN predicts residual on reduced dynamic subspace.
5. Corrected prediction is merged back into state estimate.
6. UKF update fuses corrected prior with measurements.
7. Error signals are computed for adaptation.
8. RLS updates selected physical parameters within bounds.
9. History logger stores state/residual/parameter traces.

If something diverges, diagnose in this order:
1. measurement consistency
2. UKF conditioning and innovation
3. NN residual magnitude and sign
4. RLS aggressiveness and bounds


