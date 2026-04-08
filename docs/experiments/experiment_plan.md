# Experiment Plan

## Experiment 1: Baseline Comparison
- Methods: pure odometry, fixed-altitude EKF (40m, 60m, 80m, 100m), adaptive EKF
- Environment: world_grid
- Seeds: 10 per configuration
- Metrics: per-building RMSE, mean RMSE, convergence time

## Experiment 2: Cross-Environment Generalization
- Method: adaptive EKF
- Environments: world_grid, world_irregular, world_cluttered
- Metrics: RMSE distribution per world

## Experiment 3: Observability Validation
- Trajectories: (a) circular orbit, (b) straight line toward buildings
- Expected: convergence in (a), failure in (b)
- Plot: eigenvalues of observability Gramian over time

## Experiment 4: Noise Sensitivity
- Pixel noise sigma: 1, 2, 3, 5, 7, 10 pixels
- Method: adaptive EKF on world_grid
- Plot: RMSE vs. noise level

## Metrics (computed by scripts/compute_metrics.py)
- RMSE per building (Euclidean distance to ground truth)
- Mean RMSE across all buildings
- Convergence time (time to reach RMSE < threshold)
- CEP (Circular Error Probable — 50th percentile error)
