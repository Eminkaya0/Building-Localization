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

- ROS2 Humble, Gazebo Harmonic, Python, YOLOv8 (ultralytics)
- PX4 SITL or AscTec Firefly (RotorS) for UAV simulation
- NumPy, SciPy, Matplotlib, Pandas

## Key Design Decisions

- **Theory first:** All math derivations in `docs/theory/` must be complete before implementing
- **EKF is ROS-independent:** `src/blocs_estimation/ekf_building.py` is pure Python, no ROS deps, unit-testable in isolation
- **ROS code lives in node wrappers only**
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
docs/theory/          — EKF derivation, observability analysis, control law
docs/experiments/     — Experiment plans and results documentation
docs/paper/           — LaTeX source (IEEE template)
src/blocs_perception/ — YOLOv8 wrapper, bbox -> measurement conversion
src/blocs_estimation/ — EKF implementation (pure Python)
src/blocs_control/    — Altitude controller, trajectory planner
src/blocs_simulation/ — Gazebo worlds and ROS2 launch files
scripts/              — Experiment runners, metrics, plotting
data/ground_truth/    — Building positions per world (YAML)
data/results/         — Experiment outputs (CSV)
tests/                — Unit tests (EKF math, etc.)
```

## For Claude Code

- Keep theory docs and implementation in sync — if the math changes, update the code and vice versa
- Type hints and docstrings everywhere in Python code
- Commit after each completed phase
- Reference QUASAR as baseline, never copy its code
