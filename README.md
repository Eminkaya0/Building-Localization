# BLOCS — Building Localization with Observer-based Control for UAVs

An EKF-based framework for localizing buildings from a UAV equipped with a monocular camera, with an uncertainty-driven altitude controller that adapts flight height to maximize localization accuracy.

## Motivation

Existing approaches to UAV-based building localization (e.g., QUASAR) use pure odometry with inverse projection at a fixed altitude. This suffers from:

- **Cumulative drift** in position estimates due to odometry errors
- **Suboptimal altitude** — too high means noisy measurements, too low means limited coverage

BLOCS solves both problems by:

1. Using an **Extended Kalman Filter (EKF)** to fuse sequential bounding box measurements into a filtered 3D building position estimate
2. Designing an **adaptive altitude controller** driven by the EKF covariance — descending when uncertain, ascending when confident

## Method Overview

```
YOLOv8 Detection → Bounding Box Center (u,v) → EKF Update → Position Estimate
                                                    ↓
                                              Covariance P
                                                    ↓
                                         Altitude Controller
                                                    ↓
                                           UAV Height Command
```

**State vector:** `x = [x_b, y_b, z_b]^T` (static building position in world frame)

**Measurement model:** Pinhole camera projection `h(x) = pi(K * T_cw * x)` where K is the camera intrinsic matrix and T_cw is the camera-to-world transform from UAV odometry.

**Control law:** `h_desired = h_min + (h_max - h_min) * exp(-alpha * trace(P))`

## Prerequisites

- **MATLAB R2023a+** (core algorithms, experiments, plotting)
- Ubuntu 22.04 (for simulation)
- ROS2 Humble + Gazebo Harmonic (simulation environment)
- Python 3.10+ with YOLOv8 (perception)
- PX4 Autopilot (SITL) or RotorS (AscTec Firefly)

## Quick Start (MATLAB)

```matlab
% Run from matlab/tests/ directory
cd matlab/tests
run_all_tests          % Unit tests for EKF, controller, observability

% Run experiments from matlab/experiments/
cd ../experiments
run_baseline_comparison       % Exp 1: Method comparison
run_observability_validation  % Exp 3: Trajectory observability
run_noise_sensitivity         % Exp 4: Noise robustness
plot_results                  % Generate publication figures
```

## Installation (Simulation)

```bash
# Clone the repository
git clone https://github.com/Eminkaya0/Building-Localization.git
cd Building-Localization

# Install Python dependencies (for YOLOv8 perception)
pip install -r requirements.txt
```

## Project Structure

```
matlab/               — Core algorithms (MATLAB)
  ekf/                — BuildingEKF class, multi-building tracker
  control/            — Adaptive altitude controller
  analysis/           — Observability Gramian, rank analysis, visualization
  experiments/        — Experiment runners, metrics, plotting
  utils/              — Camera projection, Jacobian, trajectory generation
  tests/              — Unit tests
docs/theory/          — Mathematical derivations (EKF, observability, control law)
docs/experiments/     — Experiment plans and results
docs/paper/           — LaTeX paper source
src/blocs_perception/ — YOLOv8 detection wrapper (Python/ROS2)
src/blocs_simulation/ — Gazebo worlds and launch files
data/                 — Ground truth and experiment results
```

## Experiments

| # | Experiment | Purpose |
|---|-----------|---------|
| 1 | Baseline comparison | Pure odometry vs. fixed-alt EKF vs. adaptive EKF |
| 2 | Cross-environment | Generalization across 3 world types |
| 3 | Observability validation | Circular orbit (observable) vs. straight line (degenerate) |
| 4 | Noise sensitivity | RMSE vs. pixel noise sigma |

## Citation

Paper in preparation. Targeting IEEE RA-L / ICRA / IROS.

## Author

Muhammed Emin Kaya — Yildiz Technical University, Control and Automation Engineering
