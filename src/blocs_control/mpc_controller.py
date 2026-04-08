"""Model Predictive Controller for uncertainty-driven altitude optimization.

Predicts covariance evolution over a receding horizon and selects the altitude
sequence that minimizes a weighted sum of estimation uncertainty and control effort.

The key insight from observability analysis: information gain scales as 1/h^2,
so the MPC explicitly models this relationship when predicting future covariance.
"""

import numpy as np
from typing import List, Tuple, Optional


class CovariancePredictionModel:
    """Predicts EKF covariance evolution as a function of altitude.

    At altitude h, the information gain per measurement is proportional to 1/h^2
    (from the Jacobian norm ||H|| ~ f/Z_C where Z_C ~ h for nadir viewing).

    The predicted covariance update at altitude h:
        P_next = P + Q*dt - gamma/h^2 * P  (simplified information reduction)

    where gamma encapsulates focal length and measurement noise.
    """

    def __init__(self, Q: np.ndarray, R_pixel: np.ndarray, f: float = 500.0,
                 n_buildings_fov_ref: int = 16, h_ref: float = 80.0):
        self.Q = Q
        self.R_pixel = R_pixel
        self.f = f
        # Information gain rate at reference altitude (calibrated from EKF runs)
        # At h_ref, with n_buildings in FOV, the trace reduction per second
        self.gamma = f**2 / (R_pixel[0, 0] * h_ref**2)  # baseline info rate
        self.h_ref = h_ref
        self.n_ref = n_buildings_fov_ref

    def predict_trace(self, trace_current: float, h: float, dt: float,
                      n_visible: int = 8) -> float:
        """Predict trace(P) after dt seconds at altitude h.

        Models: d/dt tr(P) = tr(Q) - n_vis * (f^2 / (R * h^2)) * tr(P) / tr(P_scale)
        Simplified to a scalar ODE for computational efficiency in MPC.
        """
        q_rate = np.trace(self.Q)  # process noise rate
        # Information rate scales as n_visible / h^2
        info_rate = n_visible * self.f**2 / (self.R_pixel[0, 0] * h**2)
        # Effective trace reduction (saturates when trace is small)
        trace_reduction = info_rate * min(trace_current, 900.0) / 900.0
        trace_next = trace_current + (q_rate - trace_reduction) * dt
        return max(trace_next, 0.1)  # prevent negative trace

    def predict_n_visible(self, h: float, n_total: int = 16,
                          fov_half_angle: float = 0.57) -> int:
        """Estimate number of buildings visible at altitude h.

        FOV width = 2 * h * tan(fov_half_angle).
        For a grid of buildings with known spacing, count those in FOV.
        """
        fov_width = 2 * h * np.tan(fov_half_angle)
        # Approximate: visible fraction of grid scales with FOV area
        fov_area = fov_width**2
        grid_area = (3 * 30.0)**2  # 4x4 grid, 30m spacing -> 90m x 90m
        fraction = min(fov_area / grid_area, 1.0)
        return max(1, int(n_total * fraction))


class AltitudeMPC:
    """Receding-horizon MPC for UAV altitude optimization.

    Cost function over horizon N:
        J = sum_{k=0}^{N-1} [ w_J * trace(P_k) + w_u * (h_k - h_{k-1})^2
                              + w_coverage * coverage_penalty(h_k) ]
        + w_terminal * trace(P_N)

    Decision variables: altitude sequence {h_0, ..., h_{N-1}}
    Constraints: h_min <= h_k <= h_max, |h_k - h_{k-1}| <= h_dot_max * dt
    """

    def __init__(self, prediction_model: CovariancePredictionModel,
                 horizon: int = 10, dt: float = 2.0,
                 h_min: float = 20.0, h_max: float = 120.0,
                 h_dot_max: float = 3.0,
                 w_trace: float = 1.0, w_control: float = 0.5,
                 w_coverage: float = 0.3, w_terminal: float = 2.0,
                 n_total_buildings: int = 16):
        self.model = prediction_model
        self.N = horizon
        self.dt = dt
        self.h_min = h_min
        self.h_max = h_max
        self.h_dot_max = h_dot_max
        self.dh_max = h_dot_max * dt  # max altitude change per step

        # Cost weights
        self.w_trace = w_trace
        self.w_control = w_control
        self.w_coverage = w_coverage
        self.w_terminal = w_terminal
        self.n_total = n_total_buildings

        # Discretization for optimization
        self.n_candidates = 7  # altitude candidates per step

    def _coverage_penalty(self, h: float) -> float:
        """Penalize low altitude (poor coverage). Normalized to [0, 1]."""
        return ((self.h_max - h) / (self.h_max - self.h_min))**2

    def _evaluate_sequence(self, h_seq: np.ndarray, trace_0: float,
                           h_current: float) -> float:
        """Evaluate total cost of an altitude sequence."""
        cost = 0.0
        trace_k = trace_0
        h_prev = h_current

        for k in range(len(h_seq)):
            h_k = h_seq[k]
            n_vis = self.model.predict_n_visible(h_k, self.n_total)
            trace_k = self.model.predict_trace(trace_k, h_k, self.dt, n_vis)

            # Stage cost
            cost += self.w_trace * trace_k / 900.0  # normalized
            cost += self.w_control * ((h_k - h_prev) / self.dh_max)**2
            cost += self.w_coverage * self._coverage_penalty(h_k)
            h_prev = h_k

        # Terminal cost
        cost += self.w_terminal * trace_k / 900.0
        return cost

    def solve(self, trace_current: float, h_current: float,
              building_traces: Optional[List[float]] = None) -> Tuple[float, float, dict]:
        """Solve MPC to get optimal altitude command.

        Uses a multi-resolution search: coarse grid -> fine refinement.

        Args:
            trace_current: Current mean trace(P) across buildings
            h_current: Current UAV altitude
            building_traces: Optional per-building traces for priority weighting

        Returns:
            h_cmd: Commanded altitude (rate-limited)
            h_des: Desired altitude from MPC (first element of optimal sequence)
            info: Dict with solver details
        """
        # Stage 1: Coarse search over altitude profiles
        # Generate candidate constant-altitude sequences and ramp sequences
        best_cost = np.inf
        best_seq = np.full(self.N, h_current)

        # Feasible altitude range given rate limit
        h_lo = max(self.h_min, h_current - self.dh_max * self.N)
        h_hi = min(self.h_max, h_current + self.dh_max * self.N)

        # Candidate target altitudes
        h_targets = np.linspace(h_lo, h_hi, self.n_candidates)

        for h_target in h_targets:
            # Generate rate-limited ramp to target
            seq = self._generate_ramp(h_current, h_target)
            cost = self._evaluate_sequence(seq, trace_current, h_current)
            if cost < best_cost:
                best_cost = cost
                best_seq = seq.copy()

        # Stage 2: V-shaped profiles (descend then ascend)
        for h_low in np.linspace(h_lo, h_current, 4):
            for t_switch in range(1, min(self.N, 5)):
                seq = self._generate_v_profile(h_current, h_low, t_switch)
                cost = self._evaluate_sequence(seq, trace_current, h_current)
                if cost < best_cost:
                    best_cost = cost
                    best_seq = seq.copy()

        # Stage 3: Fine-tune around best sequence
        for _ in range(3):  # refinement iterations
            for k in range(self.N):
                # Try small perturbations
                for delta in [-self.dh_max * 0.5, 0, self.dh_max * 0.5]:
                    test_seq = best_seq.copy()
                    test_seq[k] = np.clip(test_seq[k] + delta, self.h_min, self.h_max)
                    # Enforce rate limits
                    test_seq = self._enforce_rate_limits(test_seq, h_current)
                    cost = self._evaluate_sequence(test_seq, trace_current, h_current)
                    if cost < best_cost:
                        best_cost = cost
                        best_seq = test_seq.copy()

        # Extract first action
        h_des = best_seq[0]

        # Apply rate limiting for safety
        h_dot = np.clip((h_des - h_current) / max(self.dt, 0.01),
                        -self.h_dot_max, self.h_dot_max)
        h_cmd = np.clip(h_current + h_dot * self.dt, self.h_min, self.h_max)

        info = {
            'optimal_sequence': best_seq.tolist(),
            'cost': best_cost,
            'predicted_trace_final': self._predict_final_trace(best_seq, trace_current),
            'n_visible_current': self.model.predict_n_visible(h_current, self.n_total),
        }

        return h_cmd, h_des, info

    def _generate_ramp(self, h_start: float, h_target: float) -> np.ndarray:
        """Generate a rate-limited ramp from h_start to h_target."""
        seq = np.zeros(self.N)
        h = h_start
        for k in range(self.N):
            delta = np.clip(h_target - h, -self.dh_max, self.dh_max)
            h = np.clip(h + delta, self.h_min, self.h_max)
            seq[k] = h
        return seq

    def _generate_v_profile(self, h_start: float, h_low: float,
                            t_switch: int) -> np.ndarray:
        """Generate descent-then-ascent V-shaped profile."""
        seq = np.zeros(self.N)
        # Phase 1: descend to h_low
        h = h_start
        for k in range(t_switch):
            delta = np.clip(h_low - h, -self.dh_max, self.dh_max)
            h = np.clip(h + delta, self.h_min, self.h_max)
            seq[k] = h
        # Phase 2: ascend to h_max
        for k in range(t_switch, self.N):
            delta = np.clip(self.h_max - h, -self.dh_max, self.dh_max)
            h = np.clip(h + delta, self.h_min, self.h_max)
            seq[k] = h
        return seq

    def _enforce_rate_limits(self, seq: np.ndarray, h_start: float) -> np.ndarray:
        """Enforce rate limits on a sequence (forward pass)."""
        out = seq.copy()
        h_prev = h_start
        for k in range(len(out)):
            delta = np.clip(out[k] - h_prev, -self.dh_max, self.dh_max)
            out[k] = np.clip(h_prev + delta, self.h_min, self.h_max)
            h_prev = out[k]
        return out

    def _predict_final_trace(self, seq: np.ndarray, trace_0: float) -> float:
        """Predict final trace after executing sequence."""
        trace = trace_0
        for h_k in seq:
            n_vis = self.model.predict_n_visible(h_k, self.n_total)
            trace = self.model.predict_trace(trace, h_k, self.dt, n_vis)
        return trace


def mpc_altitude_controller(building_traces: List[float], h_current: float,
                            dt: float, Q: np.ndarray, R_pixel: np.ndarray,
                            horizon: int = 10, mpc_dt: float = 2.0,
                            h_min: float = 20.0, h_max: float = 120.0,
                            h_dot_max: float = 3.0,
                            n_total_buildings: int = 16) -> Tuple[float, float, dict]:
    """Convenience function: create MPC and solve in one call.

    Args:
        building_traces: List of trace(P_i) for each tracked building
        h_current: Current altitude
        dt: Time step for rate limiting
        Q: Process noise matrix
        R_pixel: Measurement noise matrix
        horizon: MPC prediction horizon (steps)
        mpc_dt: MPC time step (seconds)

    Returns:
        h_cmd, h_des, info
    """
    mean_trace = np.mean(building_traces) if building_traces else 900.0

    model = CovariancePredictionModel(Q, R_pixel)
    mpc = AltitudeMPC(model, horizon=horizon, dt=mpc_dt,
                      h_min=h_min, h_max=h_max, h_dot_max=h_dot_max,
                      n_total_buildings=n_total_buildings)

    h_cmd, h_des, info = mpc.solve(mean_trace, h_current, building_traces)

    # Final rate limiting at actual dt
    h_dot = np.clip((h_cmd - h_current) / max(dt, 0.01), -h_dot_max, h_dot_max)
    h_cmd_final = np.clip(h_current + h_dot * dt, h_min, h_max)

    return h_cmd_final, h_des, info
