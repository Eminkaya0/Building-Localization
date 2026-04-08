"""Extended Kalman Filter for static building position estimation."""

import numpy as np
from typing import Optional, Tuple


def world_to_camera(x_world: np.ndarray, uav_pos: np.ndarray, uav_R_WB: np.ndarray,
                    R_BC: np.ndarray, t_BC: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform a 3D world point to camera frame coordinates."""
    R_WC = R_BC @ uav_R_WB
    R_CW = R_WC.T
    t_WC = R_BC @ uav_pos + t_BC
    t_CW = -R_CW @ t_WC
    p_C = R_CW @ x_world + t_CW
    return p_C, R_CW, t_CW


def camera_project(x_world: np.ndarray, uav_pos: np.ndarray, uav_R_WB: np.ndarray,
                   K: np.ndarray, R_BC: np.ndarray, t_BC: np.ndarray,
                   img_size: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
    """Project a 3D world point to pixel coordinates via pinhole model."""
    p_C, _, _ = world_to_camera(x_world, uav_pos, uav_R_WB, R_BC, t_BC)
    X_C, Y_C, Z_C = p_C

    if Z_C <= 0:
        return np.array([np.nan, np.nan]), False

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * X_C / Z_C + cx
    v = fy * Y_C / Z_C + cy
    z_pixel = np.array([u, v])

    is_valid = True
    if img_size is not None:
        if u < 0 or u >= img_size[0] or v < 0 or v >= img_size[1]:
            is_valid = False
    return z_pixel, is_valid


def camera_project_jacobian(x_world: np.ndarray, uav_pos: np.ndarray, uav_R_WB: np.ndarray,
                            K: np.ndarray, R_BC: np.ndarray, t_BC: np.ndarray) -> np.ndarray:
    """Compute the 2x3 measurement Jacobian H = dh/dx."""
    p_C, R_CW, _ = world_to_camera(x_world, uav_pos, uav_R_WB, R_BC, t_BC)
    X_C, Y_C, Z_C = p_C
    fx, fy = K[0, 0], K[1, 1]

    J_pi = (1.0 / Z_C) * np.array([
        [fx, 0.0, -fx * X_C / Z_C],
        [0.0, fy, -fy * Y_C / Z_C]
    ])
    return J_pi @ R_CW


class BuildingEKF:
    """EKF for estimating a single static building's 3D position."""

    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray,
                 R_pixel: np.ndarray, gate_threshold: float = 9.21):
        self.x = x0.flatten().copy()
        self.P = P0.copy()
        self.Q = Q.copy()
        self.R_meas = R_pixel.copy()
        self.gate_threshold = gate_threshold
        self.n_updates = 0

    def predict(self, dt: float):
        self.P = self.P + self.Q * dt

    def update(self, z_pixel: np.ndarray, uav_pos: np.ndarray, uav_R: np.ndarray,
               K: np.ndarray, R_BC: np.ndarray, t_BC: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        z_pred, is_valid = camera_project(self.x, uav_pos, uav_R, K, R_BC, t_BC)
        if not is_valid:
            return False, np.full(2, np.nan), np.full((2, 2), np.nan)

        H = camera_project_jacobian(self.x, uav_pos, uav_R, K, R_BC, t_BC)
        innovation = z_pixel.flatten() - z_pred
        S = H @ self.P @ H.T + self.R_meas

        # Mahalanobis gating
        d2 = innovation @ np.linalg.solve(S, innovation)
        if d2 > self.gate_threshold:
            return False, innovation, S

        K_gain = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K_gain @ innovation

        # Joseph form
        I_KH = np.eye(3) - K_gain @ H
        self.P = I_KH @ self.P @ I_KH.T + K_gain @ self.R_meas @ K_gain.T
        self.P = (self.P + self.P.T) / 2.0

        self.n_updates += 1
        return True, innovation, S

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x.copy(), self.P.copy()

    def trace_P(self) -> float:
        return np.trace(self.P)

    def position_error(self, x_true: np.ndarray) -> float:
        return np.linalg.norm(self.x - x_true.flatten())
