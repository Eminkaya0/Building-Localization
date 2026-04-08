"""Unit tests for BLOCS EKF, controller, and observability."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from blocs_estimation.ekf_building import BuildingEKF, camera_project, camera_project_jacobian
from blocs_estimation.trajectory import generate_uav_trajectory, generate_synthetic_measurements
from blocs_estimation.observability import compute_observability_gramian, check_observability
from blocs_control.altitude_controller import altitude_controller, rate_limiter

# Shared test fixtures
K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
R_BC = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)  # cam looking down
t_BC = np.zeros(3)
SIGMA = 2.0
R_PIXEL = np.diag([SIGMA**2, SIGMA**2])
P0 = np.diag([400.0, 400.0, 100.0])
Q = np.diag([0.01, 0.01, 0.01])


def _run_ekf(x_true, traj_type, traj_kwargs, sigma=SIGMA, Q_mat=Q, duration_override=None):
    """Helper: run EKF on synthetic data, return final error and traces."""
    if duration_override:
        traj_kwargs['duration'] = duration_override
    pos, rot, ts = generate_uav_trajectory(traj_type, **traj_kwargs)
    np.random.seed(42)
    meas = generate_synthetic_measurements(x_true.reshape(3, 1), pos, rot, ts, K, R_BC, t_BC, sigma)

    x0 = x_true + np.array([15.0, -10.0, 5.0])
    ekf = BuildingEKF(x0, P0, Q_mat, np.diag([sigma**2, sigma**2]))

    errors, traces = [], []
    for i, m in enumerate(meas):
        if i > 0:
            ekf.predict(m['time'] - meas[i-1]['time'])
        for d in m['detections']:
            if d['building_id'] == 1:
                ekf.update(d['z_noisy'], m['uav_pos'], m['uav_R'], K, R_BC, t_BC)
        errors.append(ekf.position_error(x_true))
        traces.append(ekf.trace_P())
    return errors, traces, ekf


class TestEKFConvergence:
    def test_converges_circular(self):
        x_true = np.array([50.0, 30.0, 0.0])
        errors, _, ekf = _run_ekf(x_true, 'circular',
            dict(center=[50, 30, -80], radius=60, speed=5, dt=0.5, duration=80))
        assert errors[-1] < 2.0, f"Final error {errors[-1]:.2f}m, expected <2m"
        assert ekf.n_updates > 10

    def test_covariance_shrinks(self):
        x_true = np.array([0.0, 0.0, 0.0])
        _, traces, _ = _run_ekf(x_true, 'circular',
            dict(center=[0, 0, -60], radius=40, speed=5, dt=1.0, duration=60),
            Q_mat=np.diag([0.001]*3))
        ratio = traces[-1] / traces[0]
        assert ratio < 0.1, f"Covariance ratio {ratio:.4f}, expected <0.1"

    def test_missing_measurements(self):
        x_true = np.array([30.0, 20.0, 0.0])
        pos, rot, ts = generate_uav_trajectory('circular',
            center=[30, 20, -80], radius=50, speed=5, dt=0.5, duration=100)
        np.random.seed(7)
        meas = generate_synthetic_measurements(x_true.reshape(3,1), pos, rot, ts, K, R_BC, t_BC, SIGMA)

        x0 = x_true + np.array([10.0, -8.0, 3.0])
        ekf = BuildingEKF(x0, P0, Q, R_PIXEL)

        N = len(meas)
        gap_start, gap_end = int(0.35*N), int(0.65*N)
        trace_gap_start = trace_gap_end = None

        for i, m in enumerate(meas):
            if i > 0:
                ekf.predict(m['time'] - meas[i-1]['time'])
            if i == gap_start:
                trace_gap_start = ekf.trace_P()
            if i == gap_end:
                trace_gap_end = ekf.trace_P()
            if gap_start <= i <= gap_end:
                continue
            for d in m['detections']:
                if d['building_id'] == 1:
                    ekf.update(d['z_noisy'], m['uav_pos'], m['uav_R'], K, R_BC, t_BC)

        assert trace_gap_end > trace_gap_start, "Covariance should grow during gap"
        assert ekf.position_error(x_true) < 5.0, "EKF should recover after gap"


class TestDegenerateTrajectory:
    def test_straight_line_poor_observability(self):
        x_true = np.array([100.0, 0.0, 0.0])

        pos_str, rot_str, _ = generate_uav_trajectory('straight',
            start_pos=[-50, 0, -80], direction=[1, 0, 0], speed=5, dt=0.5, duration=15)
        pos_circ, rot_circ, _ = generate_uav_trajectory('circular',
            center=[100, 0, -80], radius=50, speed=5, dt=0.5, duration=60)

        _, _, cond_str, _ = check_observability(
            compute_observability_gramian(x_true, pos_str, rot_str, K, R_BC, t_BC, R_PIXEL)[0])
        _, _, cond_circ, _ = check_observability(
            compute_observability_gramian(x_true, pos_circ, rot_circ, K, R_BC, t_BC, R_PIXEL)[0])

        assert cond_str > 10 * cond_circ, \
            f"Degenerate should be worse: cond_str={cond_str:.0f}, cond_circ={cond_circ:.0f}"


class TestAltitudeController:
    def test_high_uncertainty_descends(self):
        _, h_des, _ = altitude_controller(900.0, 80.0, 0.1)
        assert h_des < 40, f"h_des={h_des:.1f}, expected <40 for high uncertainty"

    def test_low_uncertainty_ascends(self):
        _, h_des, _ = altitude_controller(5.0, 80.0, 0.1)
        assert h_des > 100, f"h_des={h_des:.1f}, expected >100 for low uncertainty"

    def test_rate_limiting(self):
        h_cmd, _, _ = altitude_controller(900.0, 80.0, 1.0, h_dot_max=3.0)
        assert abs(h_cmd - 80.0) <= 3.0 + 1e-10

    def test_altitude_bounds(self):
        h_cmd_low = rate_limiter(0.0, 20.0, 1.0)
        h_cmd_high = rate_limiter(200.0, 120.0, 1.0)
        assert h_cmd_low >= 20.0
        assert h_cmd_high <= 120.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
