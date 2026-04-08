"""Observability analysis for static building localization."""

import numpy as np
from typing import Tuple
from .ekf_building import camera_project, camera_project_jacobian


def compute_observability_gramian(x_building: np.ndarray, positions: np.ndarray,
                                   rotations: list, K: np.ndarray, R_BC: np.ndarray,
                                   t_BC: np.ndarray, R_meas: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the observability Gramian O = sum(H_i' R^-1 H_i).

    Returns: (O [3x3], eigenvalues [3], eigenvectors [3x3])
    """
    R_inv = np.linalg.inv(R_meas)
    O = np.zeros((3, 3))
    N = positions.shape[1]

    for i in range(N):
        _, is_valid = camera_project(x_building, positions[:, i], rotations[i], K, R_BC, t_BC)
        if not is_valid:
            continue
        H = camera_project_jacobian(x_building, positions[:, i], rotations[i], K, R_BC, t_BC)
        O += H.T @ R_inv @ H

    eigenvalues, eigenvectors = np.linalg.eigh(O)
    idx = np.argsort(eigenvalues)[::-1]
    return O, eigenvalues[idx], eigenvectors[:, idx]


def check_observability(O: np.ndarray, tol: float = 1e-6) -> Tuple[bool, int, float, np.ndarray]:
    """Check observability from Gramian. Returns: (is_observable, rank, cond_num, eigenvalues)."""
    eigenvalues = np.sort(np.linalg.eigvalsh(O))[::-1]
    rank = np.sum(eigenvalues > tol * eigenvalues[0])
    is_observable = rank == 3
    cond_num = eigenvalues[0] / eigenvalues[2] if eigenvalues[2] > 0 else np.inf
    return is_observable, rank, cond_num, eigenvalues
