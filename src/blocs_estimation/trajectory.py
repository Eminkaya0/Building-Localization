"""UAV trajectory generation utilities."""

import numpy as np
from typing import Tuple, List


def eul2rotm_zyx(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """ZYX Euler angles to rotation matrix (body-to-world)."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])


def generate_uav_trajectory(traj_type: str, **kwargs) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Generate UAV trajectory waypoints.

    Returns: (positions [3xN], rotations [list of 3x3], timestamps [N])
    """
    altitude = kwargs.get('altitude', -80.0)
    speed = kwargs.get('speed', 5.0)
    duration = kwargs.get('duration', 120.0)
    dt = kwargs.get('dt', 0.1)
    timestamps = np.arange(0, duration + dt, dt)
    N = len(timestamps)
    positions = np.zeros((3, N))
    rotations = []

    if traj_type == 'circular':
        center = np.array(kwargs.get('center', [0.0, 0.0, altitude]))
        radius = kwargs.get('radius', 50.0)
        omega = speed / radius
        for i in range(N):
            theta = omega * timestamps[i]
            positions[:, i] = center + np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0])
            rotations.append(eul2rotm_zyx(theta + np.pi / 2))

    elif traj_type == 'straight':
        start_pos = np.array(kwargs.get('start_pos', [-100.0, 0.0, altitude]))
        direction = np.array(kwargs.get('direction', [1.0, 0.0, 0.0]), dtype=float)
        direction /= np.linalg.norm(direction)
        heading = np.arctan2(direction[1], direction[0])
        R = eul2rotm_zyx(heading)
        for i in range(N):
            positions[:, i] = start_pos + speed * timestamps[i] * direction
            rotations.append(R)

    elif traj_type == 'lawnmower':
        w = kwargs.get('width', 100.0)
        h = kwargs.get('height', 100.0)
        spacing = kwargs.get('spacing', 20.0)
        start = np.array(kwargs.get('start_pos', [-w/2, -h/2, altitude]))
        n_lines = int(h / spacing) + 1
        waypoints = []
        for j in range(n_lines):
            y = start[1] + j * spacing
            if j % 2 == 0:
                waypoints.append(np.array([start[0], y, altitude]))
                waypoints.append(np.array([start[0] + w, y, altitude]))
            else:
                waypoints.append(np.array([start[0] + w, y, altitude]))
                waypoints.append(np.array([start[0], y, altitude]))
        waypoints = np.column_stack(waypoints)
        positions, rotations = _interpolate_waypoints(waypoints, speed, timestamps)

    elif traj_type == 'hover':
        hover_pos = np.array(kwargs.get('center', [0.0, 0.0, altitude]))
        for i in range(N):
            positions[:, i] = hover_pos
            rotations.append(np.eye(3))
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    return positions, rotations, timestamps


def _interpolate_waypoints(waypoints: np.ndarray, speed: float, timestamps: np.ndarray):
    n_wp = waypoints.shape[1]
    cum_dist = np.zeros(n_wp)
    for j in range(1, n_wp):
        cum_dist[j] = cum_dist[j-1] + np.linalg.norm(waypoints[:, j] - waypoints[:, j-1])
    total_dist = cum_dist[-1]

    N = len(timestamps)
    positions = np.zeros((3, N))
    rotations = []
    for i in range(N):
        d = min(speed * timestamps[i], total_dist)
        seg = np.searchsorted(cum_dist, d, side='left')
        seg = max(seg, 1)
        seg_start = cum_dist[seg - 1]
        seg_len = cum_dist[seg] - seg_start
        alpha = (d - seg_start) / seg_len if seg_len > 0 else 0.0
        positions[:, i] = waypoints[:, seg-1] + alpha * (waypoints[:, seg] - waypoints[:, seg-1])
        dir_vec = waypoints[:, seg] - waypoints[:, seg-1]
        heading = np.arctan2(dir_vec[1], dir_vec[0])
        rotations.append(eul2rotm_zyx(heading))
    return positions, rotations


def generate_synthetic_measurements(buildings: np.ndarray, positions: np.ndarray,
                                     rotations: list, timestamps: np.ndarray,
                                     K: np.ndarray, R_BC: np.ndarray, t_BC: np.ndarray,
                                     sigma_pixel: float = 2.0, img_size=None) -> list:
    """Generate noisy pixel measurements from a UAV trajectory."""
    from .ekf_building import camera_project
    N = len(timestamps)
    M = buildings.shape[1] if buildings.ndim == 2 else 1
    if buildings.ndim == 1:
        buildings = buildings.reshape(3, 1)

    measurements = []
    for i in range(N):
        dets = []
        for j in range(M):
            z_true, is_valid = camera_project(buildings[:, j], positions[:, i], rotations[i], K, R_BC, t_BC, img_size)
            if is_valid:
                z_noisy = z_true + sigma_pixel * np.random.randn(2)
                dets.append({'building_id': j + 1, 'z_true': z_true, 'z_noisy': z_noisy})
        measurements.append({
            'time': timestamps[i], 'uav_pos': positions[:, i],
            'uav_R': rotations[i], 'detections': dets
        })
    return measurements
