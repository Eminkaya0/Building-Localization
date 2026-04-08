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

- Ubuntu 22.04
- ROS2 Humble
- Gazebo Harmonic
- Python 3.10+
- PX4 Autopilot (SITL) or RotorS (AscTec Firefly)

## Installation

```bash
# Clone the repository
git clone <repo-url> blocs
cd blocs

# Install Python dependencies
pip install -r requirements.txt

# Build ROS2 workspace (when ROS packages are ready)
# colcon build --packages-select blocs_perception blocs_estimation blocs_control blocs_simulation
```

## Project Structure

```
docs/theory/          — Mathematical derivations (EKF, observability, control law)
docs/experiments/     — Experiment plans and results
docs/paper/           — LaTeX paper source
src/blocs_perception/ — YOLOv8 detection wrapper
src/blocs_estimation/ — EKF implementation (pure Python, no ROS dependency)
src/blocs_control/    — Adaptive altitude controller
src/blocs_simulation/ — Gazebo worlds and launch files
scripts/              — Experiment runners and plotting
data/                 — Ground truth and results
tests/                — Unit tests
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
