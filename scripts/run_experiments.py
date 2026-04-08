"""Run all BLOCS experiments and save results."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import json
from blocs_estimation.ekf_building import BuildingEKF, camera_project
from blocs_estimation.trajectory import generate_uav_trajectory, generate_synthetic_measurements
from blocs_estimation.observability import compute_observability_gramian, check_observability
from blocs_control.altitude_controller import altitude_controller

K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
R_BC = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
t_BC = np.zeros(3)
P0 = np.diag([400.0, 400.0, 100.0])
Q = np.diag([0.01, 0.01, 0.01])
SIGMA = 2.0
R_PIXEL = np.diag([SIGMA**2, SIGMA**2])
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'results')


def building_grid(rows=4, cols=4, spacing=30.0):
    buildings = np.zeros((3, rows * cols))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            buildings[:, idx] = [(c - (cols-1)/2) * spacing, (r - (rows-1)/2) * spacing, 0]
            idx += 1
    return buildings


def back_project_to_ground(z_pixel, uav_pos, uav_R):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    ray_C = np.array([(z_pixel[0]-cx)/fx, (z_pixel[1]-cy)/fy, 1.0])
    R_WC = R_BC @ uav_R
    ray_W = R_WC @ ray_C
    t_WC = R_BC @ uav_pos + t_BC
    if abs(ray_W[2]) < 1e-10:
        return np.array([uav_pos[0], uav_pos[1], 0.0])
    lam = -t_WC[2] / ray_W[2]
    return t_WC + lam * ray_W


def run_ekf_single(buildings, pos, rot, ts, sigma, altitude_adaptive=False):
    """Run EKF on all buildings. Returns final errors [M]."""
    M = buildings.shape[1]
    meas = generate_synthetic_measurements(buildings, pos, rot, ts, K, R_BC, t_BC, sigma)
    errors = np.full(M, np.nan)

    for b in range(M):
        x0 = buildings[:, b] + np.array([15*np.random.randn(), 15*np.random.randn(), 5*np.random.randn()])
        ekf = BuildingEKF(x0, P0, Q, np.diag([sigma**2]*2))
        for i, m in enumerate(meas):
            if i > 0:
                ekf.predict(m['time'] - meas[i-1]['time'])
            for d in m['detections']:
                if d['building_id'] == b + 1:
                    ekf.update(d['z_noisy'], m['uav_pos'], m['uav_R'], K, R_BC, t_BC)
        errors[b] = ekf.position_error(buildings[:, b])
    return errors


# ============ EXPERIMENT 1: BASELINE COMPARISON ============
def experiment_baseline():
    print("=== Experiment 1: Baseline Comparison ===")
    buildings = building_grid()
    M = buildings.shape[1]
    n_seeds = 10
    results = {}

    # Pure odometry
    odo_errors = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        pos, rot, ts = generate_uav_trajectory('lawnmower', altitude=-80, duration=120, dt=0.5)
        meas = generate_synthetic_measurements(buildings, pos, rot, ts, K, R_BC, t_BC, SIGMA)
        errs = np.full(M, np.nan)
        for b in range(M):
            last_z, last_pos, last_R = None, None, None
            for m in meas:
                for d in m['detections']:
                    if d['building_id'] == b + 1:
                        last_z, last_pos, last_R = d['z_noisy'], m['uav_pos'], m['uav_R']
            if last_z is not None:
                x_odo = back_project_to_ground(last_z, last_pos, last_R)
                errs[b] = np.linalg.norm(x_odo - buildings[:, b])
        odo_errors.append(np.nanmean(errs))
    results['odometry'] = {'mean_rmse': float(np.mean(odo_errors)), 'std': float(np.std(odo_errors))}
    print(f"  Pure Odometry: {results['odometry']['mean_rmse']:.2f} +/- {results['odometry']['std']:.2f} m")

    # Fixed-altitude EKF
    for alt in [40, 60, 80, 100]:
        ekf_errors = []
        for seed in range(n_seeds):
            np.random.seed(seed)
            pos, rot, ts = generate_uav_trajectory('lawnmower', altitude=-alt, duration=120, dt=0.5)
            errs = run_ekf_single(buildings, pos, rot, ts, SIGMA)
            ekf_errors.append(np.nanmean(errs))
        key = f'ekf_fixed_{alt}'
        results[key] = {'mean_rmse': float(np.mean(ekf_errors)), 'std': float(np.std(ekf_errors))}
        print(f"  EKF (h={alt}m): {results[key]['mean_rmse']:.2f} +/- {results[key]['std']:.2f} m")

    # Adaptive EKF
    adapt_errors = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        pos, rot, ts = generate_uav_trajectory('lawnmower', altitude=-80, duration=120, dt=0.5)
        M = buildings.shape[1]
        ekfs = {}
        h_current = 80.0
        final_errs = np.full(M, np.nan)

        for i in range(len(ts)):
            dt_step = ts[i] - ts[i-1] if i > 0 else 0
            for b_id in ekfs:
                ekfs[b_id].predict(dt_step)

            if ekfs:
                mean_tr = np.mean([e.trace_P() for e in ekfs.values()])
                h_current, _, _ = altitude_controller(mean_tr, h_current, max(dt_step, 0.01))
                pos[2, i] = -h_current

            for b in range(M):
                z, valid = camera_project(buildings[:, b], pos[:, i], rot[i], K, R_BC, t_BC)
                if valid:
                    z_noisy = z + SIGMA * np.random.randn(2)
                    if b not in ekfs:
                        x0 = buildings[:, b] + np.array([15*np.random.randn(), 15*np.random.randn(), 5*np.random.randn()])
                        ekfs[b] = BuildingEKF(x0, P0, Q, R_PIXEL)
                    ekfs[b].update(z_noisy, pos[:, i], rot[i], K, R_BC, t_BC)

        for b in range(M):
            if b in ekfs:
                final_errs[b] = ekfs[b].position_error(buildings[:, b])
        adapt_errors.append(np.nanmean(final_errs))

    results['adaptive'] = {'mean_rmse': float(np.mean(adapt_errors)), 'std': float(np.std(adapt_errors))}
    print(f"  Adaptive EKF: {results['adaptive']['mean_rmse']:.2f} +/- {results['adaptive']['std']:.2f} m")
    return results


# ============ EXPERIMENT 3: OBSERVABILITY VALIDATION ============
def experiment_observability():
    print("\n=== Experiment 3: Observability Validation ===")
    x_true = np.array([50.0, 30.0, 0.0])

    pos_c, rot_c, ts_c = generate_uav_trajectory('circular', center=[50,30,-80], radius=50, speed=5, dt=0.5, duration=80)
    pos_s, rot_s, ts_s = generate_uav_trajectory('straight', start_pos=[-50,30,-80], direction=[1,0,0], speed=5, dt=0.5, duration=15)

    _, _, cond_c, eig_c = check_observability(compute_observability_gramian(x_true, pos_c, rot_c, K, R_BC, t_BC, R_PIXEL)[0])
    _, _, cond_s, eig_s = check_observability(compute_observability_gramian(x_true, pos_s, rot_s, K, R_BC, t_BC, R_PIXEL)[0])

    # Run EKF on both
    circ_final, str_final = [], []
    for seed in range(10):
        np.random.seed(seed)
        meas_c = generate_synthetic_measurements(x_true.reshape(3,1), pos_c, rot_c, ts_c, K, R_BC, t_BC, SIGMA)
        meas_s = generate_synthetic_measurements(x_true.reshape(3,1), pos_s, rot_s, ts_s, K, R_BC, t_BC, SIGMA)
        x0 = x_true + np.array([15, -10, 5.0])

        for meas_list, result_list in [(meas_c, circ_final), (meas_s, str_final)]:
            ekf = BuildingEKF(x0, P0, Q, R_PIXEL)
            for i, m in enumerate(meas_list):
                if i > 0: ekf.predict(m['time'] - meas_list[i-1]['time'])
                for d in m['detections']:
                    if d['building_id'] == 1:
                        ekf.update(d['z_noisy'], m['uav_pos'], m['uav_R'], K, R_BC, t_BC)
            result_list.append(ekf.position_error(x_true))

    results = {
        'circular': {'cond': float(cond_c), 'eigenvalues': eig_c.tolist(), 'mean_error': float(np.mean(circ_final))},
        'straight': {'cond': float(cond_s), 'eigenvalues': eig_s.tolist(), 'mean_error': float(np.mean(str_final))}
    }
    print(f"  Circular: cond={cond_c:.1f}, mean error={np.mean(circ_final):.2f} m")
    print(f"  Straight: cond={cond_s:.1f}, mean error={np.mean(str_final):.2f} m")
    return results


# ============ EXPERIMENT 4: NOISE SENSITIVITY ============
def experiment_noise():
    print("\n=== Experiment 4: Noise Sensitivity ===")
    x_true = np.array([50.0, 30.0, 0.0])
    pos, rot, ts = generate_uav_trajectory('circular', center=[50,30,-80], radius=50, speed=5, dt=0.5, duration=80)

    sigmas = [1, 2, 3, 5, 7, 10]
    results = {}
    for sigma in sigmas:
        errors = []
        for seed in range(10):
            np.random.seed(seed)
            meas = generate_synthetic_measurements(x_true.reshape(3,1), pos, rot, ts, K, R_BC, t_BC, sigma)
            x0 = x_true + np.array([15, -10, 5.0])
            ekf = BuildingEKF(x0, P0, Q, np.diag([sigma**2]*2))
            for i, m in enumerate(meas):
                if i > 0: ekf.predict(m['time'] - meas[i-1]['time'])
                for d in m['detections']:
                    if d['building_id'] == 1:
                        ekf.update(d['z_noisy'], m['uav_pos'], m['uav_R'], K, R_BC, t_BC)
            errors.append(ekf.position_error(x_true))
        results[sigma] = {'mean': float(np.mean(errors)), 'std': float(np.std(errors))}
        print(f"  sigma={sigma}px: RMSE={results[sigma]['mean']:.2f} +/- {results[sigma]['std']:.2f} m")
    return results


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}
    all_results['baseline'] = experiment_baseline()
    all_results['observability'] = experiment_observability()
    all_results['noise'] = experiment_noise()

    with open(os.path.join(RESULTS_DIR, 'all_experiments.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {RESULTS_DIR}/all_experiments.json")
