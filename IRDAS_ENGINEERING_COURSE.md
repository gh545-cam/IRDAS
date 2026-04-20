# IRDAS Engineering Course

A chaptered course to take a learner from beginner to intermediate and early advanced understanding of this repository.

Audience:
- Aspiring engineers with basic calculus and linear algebra
- Learners who want both intuition and implementation detail

Course outcomes:
- Understand nonlinear vehicle state estimation with UKF
- Understand residual learning in a physics anchored stack
- Understand online adaptation with RLS
- Understand race level behavior under noise, mismatch, fuel burn, and tire degradation

Prerequisites:
- Python fundamentals
- Introductory linear algebra (vectors, matrices)
- Basic probability and random variables

Estimated duration:
- 12 to 18 hours, including exercises

Assessment style:
- Practical coding checks
- Explain your results in words
- Compare expected trends, not exact numbers


## Chapter 1: Mental Model of the Full System

Learning goals:
- Build a clean mental map of all components
- Know what each component does and does not do

Theory core:
- Hybrid modeling combines physics plus learning plus estimation
- Physics model gives structure
- Filter gives robust state estimates from noisy sensors
- Residual network learns remaining mismatch
- RLS updates interpretable parameters online

Code walk:
- Start at irdas_main.py and read class IRDAS
- Scan simulator.py, kalman_filter.py, residual_network.py, parameter_adapter.py

Exercises:
1. Draw a block diagram of data flow for one simulation step.
2. Label each signal as state, control, measurement, residual, or parameter.
3. Explain why this is not just one big neural network.

Expected outcomes:
- You can explain the role of each subsystem in one paragraph.
- You can describe where uncertainty enters and where it is reduced.


## Chapter 2: State Space Thinking for Vehicle Dynamics

Learning goals:
- Understand state, control, measurement in this project
- Understand why mass is now included as a state

Theory core:
- Nonlinear state space form:
  x next equals f of x, u, theta plus process noise
  z equals h of x plus measurement noise
- Hidden states can be estimated if dynamics and measurements are informative
- Time varying mass should be a state when fuel is burned continuously

Code walk:
- kalman_filter.py state definition and measurement definition
- sensors.py SensorSimulator and measurement_function_fixed
- twin_track.py for f

Exercises:
1. List all 14 states and classify each as kinematic, dynamic, control/discrete, or slow varying.
2. Explain why gear is treated differently from vx and vy.
3. Explain why mass belongs in the filter and not only in external logic.

Expected outcomes:
- You can write a short equation level model for this repository.
- You can justify mass augmentation in engineering terms.


## Chapter 3: Tire and Vehicle Physics Backbone

Learning goals:
- Understand why Pacejka style tire modeling is used
- Understand load transfer and aero effects on forces

Theory core:
- Tire force is nonlinear in slip
- Peak grip and stiffness depend on load and condition
- Longitudinal and lateral dynamics are coupled through friction usage
- Aero drag and downforce alter acceleration and available grip

Code walk:
- twin_track.py pacejka_magic_formula
- twin_track.py load transfer and aero blocks
- params.py tire and aero parameters

Exercises:
1. Identify where front and rear normal loads are computed.
2. Explain how increased downforce changes cornering behavior.
3. Reduce a2 in both tire dictionaries and predict qualitative effects.

Expected outcomes:
- You can connect tire coefficients to handling behavior.
- You can explain why model mismatch appears when tires degrade.


## Chapter 4: Sensor Modeling and Measurement Consistency

Learning goals:
- Understand how simulated sensors are generated
- Understand why measurement consistency is critical for filters

Theory core:
- Sensor models define what the filter expects to observe
- Small measurement equation mistakes can create large estimation errors
- IMU terms need correct body frame treatment

Code walk:
- sensors.py measure
- sensors.py measure_fuel_system
- sensors.py measurement_function_fixed

Exercises:
1. List each measurement channel and its noise source.
2. Explain how mass measurement is fused with model based mass prediction.
3. Increase rpm noise and predict effect on rpm RMS error.

Expected outcomes:
- You can map each measurement dimension to physical sensor meaning.
- You can explain innovation as measurement minus predicted measurement.


## Chapter 5: UKF Foundations and Practical Engineering

Learning goals:
- Understand sigma point filtering without Jacobians
- Understand the predict and update flow used here

Theory core:
- UKF propagates deterministic sigma points through nonlinear dynamics
- Mean and covariance reconstructed with weighted statistics
- Kalman gain balances prediction confidence and measurement confidence

Code walk:
- kalman_filter.py _generate_sigma_points
- kalman_filter.py _mean_from_sigma
- kalman_filter.py _cov_from_sigma
- kalman_filter.py predict and update

Exercises:
1. Write a step by step pseudo algorithm for one UKF cycle in this codebase.
2. Explain what Q and R represent physically.
3. Change mass_measurement_noise and compare mass tracking smoothness versus responsiveness.

Expected outcomes:
- You can explain why UKF is used instead of EKF in this project.
- You can tune one noise term and explain observed tradeoffs.


## Chapter 6: Discrete States Inside Continuous Filters

Learning goals:
- Understand the gear handling pitfall
- Understand mitigation patterns used in this repository

Theory core:
- UKF assumes continuous states, but gear is discrete
- Naive updates can corrupt gear and explode rpm error
- Practical fix is to preserve discrete channels through measurement update and collapse covariance for those dimensions

Code walk:
- kalman_filter.py update discrete state preservation
- kalman_filter.py postprocess rounding and clipping
- twin_track.py rounded gear handling before shifting logic

Exercises:
1. Explain how a one gear error can produce very large rpm error.
2. Temporarily disable gear preservation in a branch and inspect rpm RMS behavior.
3. Restore stable behavior and document what changed.

Expected outcomes:
- You understand the difference between mathematically convenient and physically valid state treatment.
- You can articulate a robust discrete state strategy.


## Chapter 7: Residual Learning with Recurrent Networks

Learning goals:
- Understand residual learning vs full dynamics learning
- Understand why the model learns only seven dynamic channels

Theory core:
- Residual target equals real model transition minus baseline transition
- Reduced output space focuses learning on dynamics errors that matter most
- Recurrent memory captures temporal context that static models miss

Code walk:
- residual_network.py RecurrentResidualDynamicsNetwork
- residual_network.py ResidualDynamicsLearner fit and predict
- irdas_main.py pretrain_neural_network and _apply_nn_residual_correction

Exercises:
1. Explain why residual learning is data efficient here.
2. Trace exactly how the 7 state correction is written back into the full state.
3. Toggle stateful inference and compare short horizon vs long horizon behavior.

Expected outcomes:
- You can describe residual network input, output, and reconstruction path.
- You can defend the reduced state design choice.


## Chapter 8: Physics Informed Priors in Learning

Learning goals:
- Understand why priors reduce brittle learning behavior
- Understand traction and symmetry regularizers used in training

Theory core:
- Priors reduce solution space to plausible dynamics
- Traction inspired regularizer discourages unrealistically large residual accelerations
- Mirror symmetry prior encourages consistent left right behavior under mirrored steering

Code walk:
- residual_network.py _traction_circle_regularizer
- residual_network.py _symmetry_regularizer

Exercises:
1. Explain each regularizer in plain language and in math language.
2. Decrease regularizer weights and note if validation loss behavior changes.
3. Increase weights and observe if underfitting appears.

Expected outcomes:
- You can explain the bias variance tradeoff introduced by physics priors.
- You can tune regularizer weights rationally.


## Chapter 9: Recursive Least Squares Adaptation

Learning goals:
- Understand online adaptation mechanics and stability controls
- Understand what parameters are adapted and why

Theory core:
- RLS updates parameter deviations from observed force and speed error signals
- Sensitivity regressor links parameter changes to expected force effects
- Forgetting factor controls how quickly old information is discounted
- Bounds prevent nonphysical drift

Code walk:
- parameter_adapter.py adaptive_param_names and bounds
- parameter_adapter.py _compute_pacejka_regressor
- parameter_adapter.py update_rls

Exercises:
1. Explain the meaning of param_vector in the adapter.
2. Change adaptive_factor and compare adaptation responsiveness.
3. Track Cd and Cl adaptation during aggressive versus smooth driving.

Expected outcomes:
- You can explain why adaptation is online and incremental.
- You can diagnose over aggressive vs over sluggish adaptation.


## Chapter 10: End to End IRDAS Step Dynamics

Learning goals:
- Understand full sequencing in one IRDAS step
- Understand interaction between true simulator, estimator, learner, and adapter

Theory core:
- The loop alternates between truth generation and model correction
- Measurements anchor estimates
- Residual learning provides fast local corrections
- RLS provides slower interpretable drift compensation

Code walk:
- irdas_main.py step
- irdas_main.py get_metrics

Exercises:
1. Create a timeline for one step with inputs and outputs at each stage.
2. Identify where true state is used and where estimated state is used.
3. Explain why this separation is important for realistic evaluation.

Expected outcomes:
- You can reason about causality in the simulation loop.
- You can explain where each error metric comes from.


## Chapter 11: Full Race Simulation and Strategy Effects

Learning goals:
- Understand long horizon effects: fuel burn, tire degradation, pit strategy
- Understand how adaptation responds to regime shifts

Theory core:
- Long horizon simulation reveals interactions not obvious in short tests
- Tire changes reset part of the operating regime
- Two stop strategy introduces discontinuities that test adaptation robustness

Code walk:
- full_race_sim.py run_full_race
- full_race_sim.py save_visualizations

Exercises:
1. Run the full race script and inspect all plotted panels.
2. Explain lap time differences between real and predicted traces.
3. Explain behavior at each pit stop and immediately after each stop.

Expected outcomes:
- You can explain adaptation transients around pit stops.
- You can connect parameter drift and residual RMS to lap time error trends.


## Chapter 12: Diagnostics, Tuning, and Robust Engineering Practice

Learning goals:
- Build a debugging and validation workflow
- Learn practical tuning order

Theory core:
- Tuning should proceed from physics and sensing first, then learning and adaptation
- Validate one subsystem at a time before full integration

Recommended tuning order:
1. Verify sensor model and measurement dimensions
2. Stabilize UKF with realistic Q and R
3. Verify discrete state handling
4. Train residual network and check generalization
5. Enable RLS and tune adaptation pace
6. Validate race level behavior

Exercises:
1. Build a short checklist for introducing new states into UKF.
2. Create a failure catalog: three failure symptoms and likely root causes.
3. Propose one robustification improvement and implement a prototype.

Expected outcomes:
- You can debug systematically instead of by trial and error.
- You can explain which subsystem to inspect first for each symptom.


## Capstone Project

Task:
- Design and run a comparative experiment with three configurations:
  1. Physics plus UKF only
  2. Physics plus UKF plus residual network
  3. Physics plus UKF plus residual network plus RLS

Deliverables:
- A short report with plots and conclusions
- At least one table comparing:
  - Mean absolute lap time error
  - RPM RMS error
  - Mass estimation RMSE
  - Parameter drift magnitude

Rubric:
- Correctness of setup
- Clarity of explanation
- Quality of interpretation of tradeoffs
- Reproducibility


## Quick Command Plan

Use these as practical checkpoints:
- python train_test.py --mode quick
- python test_state_reconstruction.py
- python test_model_error_guard.py
- python full_race_sim.py


## Reading Map by File

Start here:
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


## Final Note

The key engineering idea in this repository is layered intelligence:
- Physics for structure
- Bayesian filtering for uncertainty handling
- Neural residuals for fast mismatch correction
- RLS for interpretable online parameter adaptation

If you can explain how those four layers cooperate, you have moved beyond beginner level and into real systems engineering competence.
