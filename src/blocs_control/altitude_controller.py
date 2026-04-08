"""Uncertainty-driven altitude controller."""

import numpy as np


def altitude_controller(mean_trace: float, h_current: float, dt: float,
                        h_min=20.0, h_max=120.0, alpha=3.0, J_ref=900.0,
                        tau=5.0, h_dot_max=3.0) -> tuple:
    """Compute commanded altitude from EKF covariance trace.

    Returns: (h_cmd, h_des, J_bar)
    """
    J_norm = mean_trace / J_ref
    h_des = h_min + (h_max - h_min) * np.exp(-alpha * J_norm)
    h_cmd = rate_limiter(h_des, h_current, dt, h_dot_max, tau, h_min, h_max)
    return h_cmd, h_des, mean_trace


def rate_limiter(h_des: float, h_current: float, dt: float,
                 h_dot_max=3.0, tau=5.0, h_min=20.0, h_max=120.0) -> float:
    """Apply rate limiting and altitude bounds."""
    h_dot = (h_des - h_current) / tau
    h_dot = np.clip(h_dot, -h_dot_max, h_dot_max)
    h_cmd = h_current + h_dot * dt
    return np.clip(h_cmd, h_min, h_max)
