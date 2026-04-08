# BLOCS — Building Localization with Observer-based Control for UAVs

## What This Project Is

BLOCS is a research project that improves UAV-based building localization using:
1. An **EKF-based observer** for building position estimation (replacing pure odometry)
2. An **uncertainty-driven altitude controller** that adapts UAV height based on EKF covariance

This extends prior work (QUASAR) which used YOLOv8 + odometry-based inverse projection at a fixed 80 m altitude. BLOCS addresses QUASAR's two main weaknesses: cumulative odometry drift and suboptimal fixed altitude.

**Target venue:** IEEE RA-L, ICRA, or IROS.

## Technical Overview

- **State vector:** x = [x_b, y_b, z_b]^T (building position in world frame, static)
- **Measurement:** Pixel coordinates of YOLOv8 bounding box center + UAV pose
- **Measurement model:** Pinhole camera projection h(x) via K[R|t]
- **Control law:** h_desired = h_min + (h_max - h_min) * exp(-alpha * trace(P))

## Tech Stack

- **MATLAB** for core algorithms: EKF, altitude controller, observability analysis, experiments
- **ROS2 Humble + Gazebo Harmonic** for UAV simulation environment
- **Python + YOLOv8** (ultralytics) for building detection (ROS2 node)
- PX4 SITL or AscTec Firefly (RotorS) for UAV simulation

## Key Design Decisions

- **Theory first:** All math derivations in `docs/theory/` must be complete before implementing
- **MATLAB for core math:** EKF, controller, and analysis are in `matlab/` — pure MATLAB, no ROS deps, unit-testable in isolation
- **Hybrid architecture:** MATLAB handles estimation/control, Python/ROS2 handles perception/simulation
- **No QUASAR code reuse** — this is a fresh implementation

## Project Phases

1. Theory and math derivations (docs/theory/)
2. Simulation setup (Gazebo worlds, launch files)
3. EKF implementation + unit tests
4. Adaptive altitude controller
5. Experiments (4 experiments, 10 seeds each)
6. Real hardware validation (stretch goal)

## Repository Layout

```
matlab/
  ekf/                — BuildingEKF.m, MultiBuildingTracker.m
  control/            — altitude_controller.m, rate_limiter.m
  analysis/           — observability Gramian, rank checks, ellipsoid plots
  experiments/        — experiment runners, metrics, plotting
  utils/              — camera projection, Jacobian, trajectory generation
  tests/              — unit tests (run_all_tests.m)
docs/theory/          — EKF derivation, observability analysis, control law
docs/experiments/     — Experiment plans and results documentation
docs/paper/           — LaTeX source (IEEE template)
src/blocs_perception/ — YOLOv8 wrapper, bbox -> measurement conversion (Python/ROS2)
src/blocs_simulation/ — Gazebo worlds and ROS2 launch files
data/ground_truth/    — Building positions per world (YAML)
data/results/         — Experiment outputs (.mat, CSV)
```

## For Claude Code

- Keep theory docs and MATLAB implementation in sync
- MATLAB code uses proper docstrings (help-style comments)
- Commit after each completed phase
- Reference QUASAR as baseline, never copy its code
