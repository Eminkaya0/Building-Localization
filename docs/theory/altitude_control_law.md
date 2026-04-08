# Uncertainty-Driven Altitude Control Law

## 1. Motivation

From the EKF derivation, we know that measurement information scales as $1/h^2$ where $h$ is the UAV altitude above the building. Lower altitude gives better localization but reduces coverage (smaller FOV). Higher altitude covers more buildings but with worse per-building accuracy.

The goal is an altitude controller that automatically trades off accuracy and coverage based on the current estimation uncertainty.

## 2. Design Philosophy

We treat altitude control as an **active sensing** problem: the UAV's altitude is a control input that affects the quality of future measurements. The EKF covariance $\mathbf{P}$ serves as a real-time indicator of estimation quality, and the controller drives the UAV to an altitude that balances:

- **Exploitation** (descend to reduce uncertainty of currently tracked buildings)
- **Exploration** (ascend to discover and track new buildings)

## 3. Primary Control Law

### 3.1 Exponential Mapping

$$
h_{\text{des}} = h_{\min} + (h_{\max} - h_{\min}) \cdot \exp(-\alpha \cdot \bar{J})
$$

where:
- $h_{\min}, h_{\max}$: altitude bounds (physical limits of the UAV and mission)
- $\alpha > 0$: sensitivity parameter
- $\bar{J}$: normalized uncertainty metric derived from EKF covariance

### 3.2 Uncertainty Metric

The simplest choice is the **average trace of P** across all tracked buildings:

$$
\bar{J} = \frac{1}{N_b} \sum_{i=1}^{N_b} \text{tr}(\mathbf{P}_i)
$$

where $N_b$ is the number of currently tracked buildings and $\mathbf{P}_i$ is the covariance of building $i$.

**Behavior:**
- When $\bar{J}$ is large (high uncertainty): $\exp(-\alpha \bar{J}) \to 0$, so $h_{\text{des}} \to h_{\min}$ (descend to get better measurements)
- When $\bar{J}$ is small (low uncertainty): $\exp(-\alpha \bar{J}) \to 1$, so $h_{\text{des}} \to h_{\max}$ (ascend to cover more ground)

### 3.3 Alternative Metrics

| Metric | Formula | When to use |
|--------|---------|-------------|
| Average trace | $\frac{1}{N_b} \sum \text{tr}(\mathbf{P}_i)$ | General-purpose, treats all directions equally |
| Maximum trace | $\max_i \text{tr}(\mathbf{P}_i)$ | Conservative, focuses on worst-estimated building |
| Average max eigenvalue | $\frac{1}{N_b} \sum \lambda_{\max}(\mathbf{P}_i)$ | Focuses on the least observable direction |
| Log determinant | $\frac{1}{N_b} \sum \log \det(\mathbf{P}_i)$ | Information-theoretic (D-optimality) |

We start with average trace for simplicity and may switch to max eigenvalue if experiments show depth estimation is the bottleneck (which the observability analysis suggests).

## 4. Tuning the Sensitivity Parameter

### 4.1 Choosing $\alpha$

The parameter $\alpha$ determines how aggressively the controller responds to uncertainty. To tune it, consider the desired behavior at two operating points:

**Operating point 1:** When average trace is at its initial value $J_0$ (freshly initialized buildings), we want the UAV near $h_{\min}$. Choose $h_{\text{des}} \approx h_{\min} + 0.05 (h_{\max} - h_{\min})$:

$$
\exp(-\alpha J_0) = 0.05 \implies \alpha = \frac{\ln(20)}{J_0} \approx \frac{3.0}{J_0}
$$

**Operating point 2:** When average trace is at a "converged" level $J_c$ (well-estimated buildings), we want the UAV near $h_{\max}$. Choose $h_{\text{des}} \approx h_{\min} + 0.95 (h_{\max} - h_{\min})$:

$$
\exp(-\alpha J_c) = 0.95 \implies \alpha = \frac{\ln(1/0.95)}{J_c} \approx \frac{0.05}{J_c}
$$

For consistency: $J_0 / J_c \approx 60$, which means the covariance trace should decrease by about 60x from initialization to convergence. With $P_0 = \text{diag}(400, 400, 100)$, $J_0 = 900$. If converged trace is about $J_c = 15$ (5 m^2 per direction), the ratio is 60. Then:

$$
\alpha \approx \frac{3.0}{900} = 0.0033 \text{ m}^{-2}
$$

### 4.2 Normalization

To make $\alpha$ independent of the number of buildings and initial covariance, we normalize:

$$
\bar{J}_{\text{norm}} = \frac{\bar{J}}{J_{\text{ref}}}
$$

where $J_{\text{ref}} = \text{tr}(\mathbf{P}_0)$ is the initial trace. Then use $\alpha_{\text{norm}} \approx 3.0$ with the normalized metric.

## 5. Rate Limiting

The raw control law can produce abrupt altitude changes, which are undesirable for:
- Physical actuator limits (vertical speed constraints)
- Measurement continuity (sudden altitude changes cause large changes in FOV, potentially losing tracked buildings)
- Safety (rapid descent risks collision)

### 5.1 Altitude Rate Limit

$$
\dot{h}_{\text{cmd}} = \text{clip}\left(\frac{h_{\text{des}} - h_{\text{current}}}{\tau}, -\dot{h}_{\max}, +\dot{h}_{\max}\right)
$$

where:
- $\tau$: time constant for the first-order altitude response (e.g., $\tau = 5$ s)
- $\dot{h}_{\max}$: maximum vertical speed (e.g., 3 m/s for a typical multirotor)

The commanded altitude at each step:

$$
h_{\text{cmd}}(t + \Delta t) = h_{\text{current}}(t) + \dot{h}_{\text{cmd}} \cdot \Delta t
$$

### 5.2 Altitude Bounds Enforcement

Hard clamp the commanded altitude:

$$
h_{\text{cmd}} = \text{clip}(h_{\text{cmd}}, h_{\min}, h_{\max})
$$

Typical values:
- $h_{\min} = 20$ m (safety margin above buildings)
- $h_{\max} = 120$ m (regulatory/communication limits)

## 6. Stability Analysis

### 6.1 Lyapunov Argument (Sketch)

Consider the total uncertainty as a Lyapunov candidate:

$$
V = \sum_{i=1}^{N_b} \text{tr}(\mathbf{P}_i)
$$

The time derivative of $V$ depends on the EKF update rate and the information gain per measurement. From the EKF covariance update:

$$
\dot{V} \approx -\sum_{i=1}^{N_b} \text{tr}(\mathbf{K}_i \mathbf{S}_i \mathbf{K}_i^T) + N_b \cdot \text{tr}(\mathbf{Q})
$$

The first term (information gain from updates) scales as $1/h^2$. The second term (process noise) is constant. The controller drives $h$ lower when $V$ is large, increasing the information gain. As $V$ decreases, $h$ increases, reducing information gain until an equilibrium is reached where:

$$
\frac{1}{h_{\text{eq}}^2} \cdot (\text{information factor}) = \text{tr}(\mathbf{Q})
$$

This equilibrium is stable: perturbations that increase $V$ cause the controller to descend (more info, $V$ decreases), and perturbations that decrease $V$ cause the controller to ascend (less info, $V$ increases back toward equilibrium).

**Note:** This is a qualitative argument. A rigorous proof would require bounding the information gain as a function of $h$ and showing that the closed-loop covariance dynamics have a globally asymptotically stable equilibrium. This is a contribution opportunity for the paper.

### 6.2 Convergence Rate

The EKF convergence rate depends on the eigenvalues of $(\mathbf{I} - \mathbf{K}\mathbf{H})$. At lower altitude, $\mathbf{H}$ has larger entries, $\mathbf{K}$ is larger, and convergence is faster. The adaptive controller accelerates initial convergence by starting low, then gradually ascends as estimates improve.

## 7. Controller Block Diagram

```
                          +-------------+
  EKF Covariances P_i --> | Compute J   | --> J_bar
                          +-------------+
                                |
                                v
                          +-------------+
                     J -> | Control Law | --> h_des
                          | h = f(J)    |
                          +-------------+
                                |
                                v
                          +-------------+
  h_current ------------> | Rate Limit  | --> h_cmd
                          | + Clamp     |
                          +-------------+
                                |
                                v
                          +-------------+
                          | UAV Altitude|
                          | Controller  |
                          +-------------+
```

## 8. Parameter Summary

| Parameter | Symbol | Default | Notes |
|-----------|--------|---------|-------|
| Min altitude | $h_{\min}$ | 20 m | Safety above buildings |
| Max altitude | $h_{\max}$ | 120 m | Regulatory/comm limit |
| Sensitivity | $\alpha_{\text{norm}}$ | 3.0 | After normalization |
| Reference trace | $J_{\text{ref}}$ | tr(P_0) | For normalization |
| Time constant | $\tau$ | 5 s | Altitude response speed |
| Max vertical speed | $\dot{h}_{\max}$ | 3 m/s | Physical limit |
| Process noise | q | 0.01 m^2/s | From EKF design |

## 9. Extensions (Future Work)

### 9.1 Per-Building Altitude Optimization

Instead of using the average uncertainty, optimize the altitude for the building that would benefit most:

$$
h_{\text{des}} = \arg\min_h \sum_{i \in \text{visible}(h)} \text{tr}\left(\mathbf{P}_i^{+}(h)\right)
$$

where $\mathbf{P}_i^{+}(h)$ is the predicted posterior covariance at altitude $h$. This is a 1D optimization that can be solved efficiently.

### 9.2 Multi-Objective Control

Combine altitude control with lateral trajectory optimization to jointly optimize coverage and accuracy. This connects to informative path planning literature and could be a follow-up paper.

### 9.3 Altitude-Dependent Measurement Noise

If $\mathbf{R}(h)$ increases with altitude (worse YOLOv8 accuracy at higher altitudes due to smaller bounding boxes), the controller naturally accounts for this through the EKF covariance, which will grow faster at high altitude.
