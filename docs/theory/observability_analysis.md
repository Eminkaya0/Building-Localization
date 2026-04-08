# Observability Analysis for Static Building Localization

## 1. Problem Setup

We analyze the observability of a static 3D building position $\mathbf{x} = [x_b, y_b, z_b]^T$ observed through a monocular camera mounted on a moving UAV. The key question: **under what UAV trajectories can we uniquely determine the building's 3D position from pixel measurements alone?**

This is a classical structure-from-motion observability problem, but we frame it in the EKF/control-theoretic language to connect directly with our estimation framework.

## 2. System Description

**State:** $\mathbf{x} = [x_b, y_b, z_b]^T \in \mathbb{R}^3$ (static, so $\dot{\mathbf{x}} = 0$)

**Measurement at time $t_i$:**

$$
\mathbf{z}_i = h_i(\mathbf{x}) = \pi(\mathbf{R}_{CW}^{(i)} \mathbf{x} + \mathbf{t}_{CW}^{(i)})
$$

where $\pi$ is the pinhole projection and $(\mathbf{R}_{CW}^{(i)}, \mathbf{t}_{CW}^{(i)})$ is the camera-to-world transform at time $t_i$, which depends on the UAV trajectory.

Each measurement provides 2 scalar equations (u, v) for 3 unknowns. A single measurement constrains the building to lie on a ray in 3D space. We need at least 2 measurements from different viewpoints to triangulate.

## 3. Observability Gramian

For a nonlinear system with no dynamics ($\mathbf{f} = 0$), the local observability Gramian based on $N$ measurements is:

$$
\mathcal{O} = \sum_{i=1}^{N} \mathbf{H}_i^T \mathbf{R}^{-1} \mathbf{H}_i
$$

where $\mathbf{H}_i = \frac{\partial h_i}{\partial \mathbf{x}}$ is the measurement Jacobian at time $t_i$.

The system is **locally observable** at $\mathbf{x}$ if and only if $\mathcal{O}$ is full rank, i.e., $\text{rank}(\mathcal{O}) = 3$.

Since each $\mathbf{H}_i$ is $2 \times 3$, a single measurement gives $\text{rank}(\mathbf{H}_i^T \mathbf{R}^{-1} \mathbf{H}_i) \leq 2$. Therefore **at least 2 measurements from different positions are required**.

## 4. Single Measurement Analysis

From the EKF derivation, the Jacobian at measurement $i$ is:

$$
\mathbf{H}_i = \mathbf{J}_\pi^{(i)} \cdot \mathbf{R}_{CW}^{(i)}
$$

where:

$$
\mathbf{J}_\pi^{(i)} = \frac{1}{Z_C^{(i)}} \begin{bmatrix} f_x & 0 & -f_x \frac{X_C^{(i)}}{Z_C^{(i)}} \\ 0 & f_y & -f_y \frac{Y_C^{(i)}}{Z_C^{(i)}} \end{bmatrix}
$$

The null space of $\mathbf{H}_i$ is 1-dimensional (since rank = 2) and corresponds to the **viewing ray direction** from camera $i$ through the building. Any displacement of $\mathbf{x}$ along this ray produces the same pixel observation.

Specifically, the null vector is:

$$
\mathbf{n}_i = (\mathbf{R}_{CW}^{(i)})^T \cdot \mathbf{p}_C^{(i)} / \|\mathbf{p}_C^{(i)}\|
$$

which is the ray from camera $i$ to the building, expressed in the world frame.

## 5. Two-Measurement Observability

Consider two measurements from UAV positions $\mathbf{p}_1$ and $\mathbf{p}_2$ with corresponding Jacobians $\mathbf{H}_1$ and $\mathbf{H}_2$.

The stacked observability matrix is:

$$
\mathcal{O}_2 = \begin{bmatrix} \mathbf{H}_1 \\ \mathbf{H}_2 \end{bmatrix} \in \mathbb{R}^{4 \times 3}
$$

The system is observable iff $\text{rank}(\mathcal{O}_2) = 3$.

**Condition for observability:** The null spaces of $\mathbf{H}_1$ and $\mathbf{H}_2$ must not overlap. Equivalently, the two viewing rays must not be parallel:

$$
\mathbf{n}_1 \times \mathbf{n}_2 \neq \mathbf{0}
$$

This is the **parallax condition**: the two camera positions must provide different viewing angles to the building.

### 5.1 Geometric Interpretation

Each pixel measurement constrains the building to a ray in 3D. Two non-parallel rays intersect at a unique point (in the noise-free case), determining the building position. If the rays are parallel (no parallax), they don't intersect — the depth along the ray is unobservable.

### 5.2 Quantifying Observability Quality

Even when $\text{rank}(\mathcal{O}_2) = 3$, the conditioning matters. Define the **parallax angle** $\theta$ between the two viewing rays:

$$
\cos\theta = \frac{|\mathbf{n}_1 \cdot \mathbf{n}_2|}{\|\mathbf{n}_1\| \|\mathbf{n}_2\|}
$$

- $\theta \to 0$: Rays nearly parallel, poor conditioning, large depth uncertainty
- $\theta = 90°$: Optimal parallax, best depth estimation
- $\theta \to 180°$: Also good parallax (viewing from opposite sides)

The condition number of $\mathcal{O}$ degrades as $O(1/\sin\theta)$, meaning small parallax angles lead to large estimation errors in the depth direction.

## 6. Degenerate Motions

### 6.1 Motion Along the Camera-Building Line

**Theorem:** If the UAV moves along the line connecting the camera and the building (i.e., purely toward or away from the building along the optical axis), the building position is **not observable**.

**Proof:** Let the building be at $\mathbf{x}_b$ and the UAV move from $\mathbf{p}_1$ to $\mathbf{p}_2 = \mathbf{p}_1 + \lambda (\mathbf{x}_b - \mathbf{p}_1)$ for some scalar $\lambda$.

For a nadir camera looking straight down, this corresponds to pure altitude changes while directly above the building. The viewing ray direction from both positions is identical:

$$
\mathbf{n}_1 = \frac{\mathbf{x}_b - \mathbf{p}_1}{\|\mathbf{x}_b - \mathbf{p}_1\|} = \frac{\mathbf{x}_b - \mathbf{p}_2}{\|\mathbf{x}_b - \mathbf{p}_2\|} = \mathbf{n}_2
$$

Therefore $\mathbf{n}_1 \times \mathbf{n}_2 = \mathbf{0}$, the observability matrix has rank $< 3$, and the system is unobservable. $\square$

**Physical intuition:** Changing altitude while directly above a building changes the size of the bounding box but not its center position in pixels. The pixel center gives bearing information, and if the bearing doesn't change, depth is indeterminate.

### 6.2 Special Case: Pure Altitude Change (Not Directly Above)

If the UAV changes altitude but is not directly above the building, the viewing ray direction changes slightly due to the off-axis viewing angle. This provides *some* parallax, but it is typically very weak. The observability Gramian is technically full-rank but poorly conditioned.

### 6.3 Straight-Line Lateral Motion (Good Case)

If the UAV flies laterally (perpendicular to the viewing direction), the parallax angle grows linearly with the baseline distance. This is the most informative trajectory for depth estimation.

## 7. Observability Gramian for N Measurements

For $N$ measurements taken along a trajectory:

$$
\mathcal{O}_N = \sum_{i=1}^{N} \mathbf{H}_i^T \mathbf{R}^{-1} \mathbf{H}_i \in \mathbb{R}^{3 \times 3}
$$

This is a positive semi-definite matrix. Its eigenvalues $\lambda_1 \geq \lambda_2 \geq \lambda_3 \geq 0$ characterize observability:

- $\lambda_3 > 0$: System is observable
- $\lambda_3 = 0$: System is unobservable, and the eigenvector corresponding to $\lambda_3$ is the unobservable direction
- $\lambda_1 / \lambda_3$: Condition number — indicates how "uniformly" observable the system is across directions

### 7.1 Eigenvalue Interpretation

For a nadir camera at altitude $h$ above a building:

- $\lambda_1, \lambda_2 \propto f^2 / (h^2 \sigma^2)$: These correspond to lateral (x, y) observability and are always well-conditioned from a single measurement
- $\lambda_3$: This corresponds to depth (z) observability and requires parallax from UAV motion

The EKF covariance lower bound (Cramer-Rao bound) is:

$$
\mathbf{P} \succeq \mathcal{O}_N^{-1}
$$

## 8. Trajectory Design Implications

### 8.1 Good Trajectories for Observability

| Trajectory | Parallax | Observability | Notes |
|-----------|----------|---------------|-------|
| Circular orbit around building | Strong, uniform | Excellent | All 3D directions well-observed |
| Lawnmower pattern | Strong lateral | Good | Depth well-observed perpendicular to sweep direction |
| Figure-8 | Strong, varied | Excellent | Good diversity of viewing angles |
| Straight line past building | Moderate | Good | Parallax from lateral motion |

### 8.2 Poor Trajectories

| Trajectory | Parallax | Observability | Notes |
|-----------|----------|---------------|-------|
| Hovering above building | None | Unobservable | No baseline at all |
| Pure altitude change above building | None | Unobservable | Degenerate case from Section 6.1 |
| Moving directly toward building | None | Unobservable | Same ray direction |
| Very short baseline | Weak | Poorly conditioned | Need sufficient distance between views |

### 8.3 Minimum Baseline for Reliable Estimation

For a building at range $d$ from the UAV, with pixel noise $\sigma$ and focal length $f$, the depth estimation error scales as:

$$
\sigma_z \approx \frac{d^2 \sigma}{f \cdot b}
$$

where $b$ is the baseline (distance between measurement positions). To achieve $\sigma_z < \sigma_z^{\max}$:

$$
b > \frac{d^2 \sigma}{f \cdot \sigma_z^{\max}}
$$

For $d = 80$ m, $\sigma = 2$ px, $f = 500$ px, $\sigma_z^{\max} = 5$ m:

$$
b > \frac{80^2 \times 2}{500 \times 5} = 5.12 \text{ m}
$$

So the UAV needs to move at least ~5 m laterally to resolve depth to within 5 m. This informs the minimum flight speed and measurement interval.

## 9. Connection to Altitude Controller

The observability analysis directly informs the altitude control design:

1. **At low altitude ($h$ small):** $\lambda_1, \lambda_2$ are large (good lateral precision), and depth parallax improves because the same lateral UAV displacement subtends a larger angle. But FOV is smaller.

2. **At high altitude ($h$ large):** $\lambda_1, \lambda_2$ are small (poor lateral precision), depth parallax is weaker. But FOV is larger and more buildings are visible.

The adaptive controller in `altitude_control_law.md` balances these effects using the EKF covariance trace as a proxy for overall estimation quality.

## 10. Experimental Validation Plan

To validate this observability analysis, Experiment 3 in the experiment plan will:

1. Run the EKF with a **circular orbit** trajectory (good parallax) and show convergence
2. Run the EKF with a **straight line toward buildings** trajectory (degenerate) and show divergence or non-convergence in the depth direction
3. Plot the eigenvalues of $\mathcal{O}_N$ over time for both trajectories
4. Show that the eigenvector of the smallest eigenvalue aligns with the viewing ray direction in the degenerate case
