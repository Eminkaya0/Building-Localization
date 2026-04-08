# Extended Kalman Filter for Building Position Estimation

## 1. Problem Statement

A UAV equipped with a monocular downward-facing camera flies over a set of static buildings. At each time step, YOLOv8 detects buildings and returns bounding box centers in pixel coordinates. The UAV's pose (position and orientation) is available from onboard odometry. The goal is to estimate each building's 3D position in the world frame using an EKF that fuses sequential bounding box measurements.

## 2. Coordinate Frames

- **World frame (W):** Fixed, NED (North-East-Down) or ENU convention. Origin at a chosen reference point.
- **Body frame (B):** Attached to the UAV center of mass.
- **Camera frame (C):** Attached to the camera. Z-axis points along the optical axis (downward for a nadir camera).

## 3. State Vector

Since buildings are static, the state vector for a single building is simply its 3D position in the world frame:

$$
\mathbf{x} = \begin{bmatrix} x_b \\ y_b \\ z_b \end{bmatrix} \in \mathbb{R}^3
$$

where $(x_b, y_b, z_b)$ is the building's position (e.g., centroid of the rooftop).

## 4. Process Model (Prediction Step)

The building does not move. The process model is the identity:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k
$$

The state transition matrix is:

$$
\mathbf{F} = \mathbf{I}_{3 \times 3}
$$

However, we add a small process noise $\mathbf{Q}$ to account for:
- Numerical stability (prevents covariance from collapsing to zero)
- Model uncertainty (the building centroid estimate may shift slightly as more of the building becomes visible)

**Prediction equations:**

$$
\hat{\mathbf{x}}_{k|k-1} = \hat{\mathbf{x}}_{k-1|k-1}
$$

$$
\mathbf{P}_{k|k-1} = \mathbf{P}_{k-1|k-1} + \mathbf{Q} \cdot \Delta t
$$

We scale Q by $\Delta t$ so that the noise injection rate is consistent regardless of the update frequency. A typical starting value is:

$$
\mathbf{Q} = \text{diag}(q, q, q), \quad q = 0.01 \text{ m}^2/\text{s}
$$

## 5. Measurement Model

### 5.1 Camera Projection

Given the building position $\mathbf{x} = [x_b, y_b, z_b]^T$ in the world frame and the camera's extrinsic parameters, the measurement is the projected pixel coordinates $(u, v)$ of the building centroid.

The full projection pipeline:

**Step 1: World to camera frame transformation**

Let the UAV provide its pose as position $\mathbf{p}_{uav} = [x_u, y_u, z_u]^T$ and rotation matrix $\mathbf{R}_{WB}$ (world-to-body). Let $\mathbf{R}_{BC}$ and $\mathbf{t}_{BC}$ be the fixed body-to-camera extrinsics. Then:

$$
\mathbf{R}_{WC} = \mathbf{R}_{BC} \cdot \mathbf{R}_{WB}
$$

$$
\mathbf{t}_{WC} = \mathbf{R}_{BC} \cdot \mathbf{p}_{uav} + \mathbf{t}_{BC}
$$

The building position in camera frame:

$$
\mathbf{p}_C = \mathbf{R}_{WC}^T \cdot (\mathbf{x} - \mathbf{t}_{WC})
$$

More precisely, if we define $\mathbf{R}_{CW} = \mathbf{R}_{WC}^T$ and $\mathbf{t}_{CW} = -\mathbf{R}_{CW} \cdot \mathbf{t}_{WC}$:

$$
\mathbf{p}_C = \mathbf{R}_{CW} \cdot \mathbf{x} + \mathbf{t}_{CW}
$$

Let $\mathbf{p}_C = [X_C, Y_C, Z_C]^T$.

**Step 2: Perspective projection**

Using the pinhole camera model with intrinsic matrix:

$$
\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

The projected pixel coordinates are:

$$
u = f_x \frac{X_C}{Z_C} + c_x
$$

$$
v = f_y \frac{Y_C}{Z_C} + c_y
$$

### 5.2 Nonlinear Measurement Function

Combining Steps 1 and 2, the measurement function is:

$$
\mathbf{z} = h(\mathbf{x}) = \begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} f_x \frac{X_C}{Z_C} + c_x \\ f_y \frac{Y_C}{Z_C} + c_y \end{bmatrix}
$$

where $\mathbf{p}_C = [X_C, Y_C, Z_C]^T = \mathbf{R}_{CW} \cdot \mathbf{x} + \mathbf{t}_{CW}$ and $\mathbf{R}_{CW}, \mathbf{t}_{CW}$ are computed from the current UAV pose (treated as known/given, not estimated).

## 6. Measurement Jacobian

We need $\mathbf{H} = \frac{\partial h}{\partial \mathbf{x}}$, a $2 \times 3$ matrix.

### 6.1 Chain Rule Decomposition

$$
\mathbf{H} = \frac{\partial h}{\partial \mathbf{x}} = \frac{\partial h}{\partial \mathbf{p}_C} \cdot \frac{\partial \mathbf{p}_C}{\partial \mathbf{x}}
$$

**Term 1:** $\frac{\partial \mathbf{p}_C}{\partial \mathbf{x}}$

Since $\mathbf{p}_C = \mathbf{R}_{CW} \cdot \mathbf{x} + \mathbf{t}_{CW}$:

$$
\frac{\partial \mathbf{p}_C}{\partial \mathbf{x}} = \mathbf{R}_{CW}
$$

This is a $3 \times 3$ matrix (the rotation from world to camera frame).

**Term 2:** $\frac{\partial h}{\partial \mathbf{p}_C}$

Let $\mathbf{p}_C = [X_C, Y_C, Z_C]^T$. Then:

$$
\frac{\partial u}{\partial X_C} = \frac{f_x}{Z_C}, \quad
\frac{\partial u}{\partial Y_C} = 0, \quad
\frac{\partial u}{\partial Z_C} = -\frac{f_x X_C}{Z_C^2}
$$

$$
\frac{\partial v}{\partial X_C} = 0, \quad
\frac{\partial v}{\partial Y_C} = \frac{f_y}{Z_C}, \quad
\frac{\partial v}{\partial Z_C} = -\frac{f_y Y_C}{Z_C^2}
$$

Therefore:

$$
\frac{\partial h}{\partial \mathbf{p}_C} = \frac{1}{Z_C} \begin{bmatrix} f_x & 0 & -f_x \frac{X_C}{Z_C} \\ 0 & f_y & -f_y \frac{Y_C}{Z_C} \end{bmatrix}
$$

### 6.2 Full Jacobian

$$
\mathbf{H} = \frac{1}{Z_C} \begin{bmatrix} f_x & 0 & -f_x \frac{X_C}{Z_C} \\ 0 & f_y & -f_y \frac{Y_C}{Z_C} \end{bmatrix} \cdot \mathbf{R}_{CW}
$$

**Key insight:** The Jacobian scales as $1/Z_C$ — when the camera is far from the building (large $Z_C$, i.e., high altitude), the Jacobian entries are small. This means each measurement provides less information at high altitude, which directly motivates the adaptive altitude controller.

### 6.3 Compact Notation

Define the projection Jacobian:

$$
\mathbf{J}_\pi = \frac{1}{Z_C} \begin{bmatrix} f_x & 0 & -f_x \frac{X_C}{Z_C} \\ 0 & f_y & -f_y \frac{Y_C}{Z_C} \end{bmatrix}
$$

Then:

$$
\mathbf{H} = \mathbf{J}_\pi \cdot \mathbf{R}_{CW}
$$

## 7. Measurement Noise Covariance

The measurement noise comes from:
- YOLOv8 bounding box center localization error
- Pixel discretization

We model this as zero-mean Gaussian noise in pixel coordinates:

$$
\mathbf{R} = \begin{bmatrix} \sigma_u^2 & 0 \\ 0 & \sigma_v^2 \end{bmatrix}
$$

Starting values: $\sigma_u = \sigma_v = 2$ pixels. This can be refined by analyzing YOLOv8 detection accuracy on the specific building classes.

**Note on altitude dependence:** At higher altitudes, bounding boxes are smaller and detection noise may increase. A more refined model could make R altitude-dependent:

$$
\sigma_u(h) = \sigma_{u,0} \cdot \left(\frac{h}{h_{\text{ref}}}\right)^\beta
$$

where $\beta > 0$ models the degradation. For the initial implementation, we use constant R.

## 8. Initial State and Covariance

### 8.1 Initial State Estimate

When a building is first detected, we initialize its 3D position using a single-frame inverse projection (the QUASAR approach). Given:
- Pixel coordinates $(u_0, v_0)$ from the first detection
- UAV pose at detection time
- An assumed building height $z_{b,\text{assumed}}$ (e.g., ground level, $z_b = 0$)

We back-project the pixel ray and intersect it with the assumed height plane:

$$
\hat{x}_{b,0} = x_u + (z_{b,\text{assumed}} - z_u) \cdot \frac{(u_0 - c_x)}{f_x} \cdot \frac{1}{\cos(\theta)}
$$

(Simplified for a nadir camera; the actual computation uses the full rotation.)

### 8.2 Initial Covariance

Since the initial estimate is based on a single measurement and an assumed height, uncertainty is high:

$$
\mathbf{P}_0 = \text{diag}(\sigma_{x,0}^2, \sigma_{y,0}^2, \sigma_{z,0}^2)
$$

Typical values:
- $\sigma_{x,0} = \sigma_{y,0} = 20$ m (large lateral uncertainty from single-view projection)
- $\sigma_{z,0} = 10$ m (building height is partially constrained by prior knowledge)

## 9. EKF Update Equations

When a measurement $\mathbf{z}_k = [u_k, v_k]^T$ is received:

**Innovation (measurement residual):**

$$
\mathbf{y}_k = \mathbf{z}_k - h(\hat{\mathbf{x}}_{k|k-1})
$$

**Innovation covariance:**

$$
\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^T + \mathbf{R}
$$

**Kalman gain:**

$$
\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^T \mathbf{S}_k^{-1}
$$

**State update:**

$$
\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \mathbf{y}_k
$$

**Covariance update (Joseph form for numerical stability):**

$$
\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^T + \mathbf{K}_k \mathbf{R} \mathbf{K}_k^T
$$

The Joseph form is preferred over the standard $\mathbf{P} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}$ because it guarantees positive semi-definiteness even with numerical errors.

## 10. Measurement Gating

To reject outlier detections (misidentified buildings, false positives), we use the Mahalanobis distance:

$$
d_M^2 = \mathbf{y}_k^T \mathbf{S}_k^{-1} \mathbf{y}_k
$$

The measurement is accepted only if:

$$
d_M^2 \leq \chi^2_{2, \alpha}
$$

where $\chi^2_{2, \alpha}$ is the chi-squared threshold with 2 degrees of freedom (since $\mathbf{z} \in \mathbb{R}^2$). For $\alpha = 0.99$:

$$
\chi^2_{2, 0.99} = 9.21
$$

This means we reject measurements that fall outside the 99% confidence ellipse of the predicted measurement distribution.

## 11. Handling Multiple Buildings

Each detected building gets its own EKF instance. The system maintains a dictionary:

```
buildings = {
    building_id_1: BuildingEKF(...),
    building_id_2: BuildingEKF(...),
    ...
}
```

**Data association:** For now, we assume YOLOv8 provides consistent tracking IDs (or use simple nearest-neighbor matching in pixel space). A more robust approach would use the Mahalanobis distance for association, but this is deferred to Phase 3.

**Track management:**
- **Initialization:** New EKF when a detection doesn't match any existing track
- **Missing measurements:** If a building leaves the FOV, the predict step still runs (covariance grows), but no update is performed
- **Track deletion:** Remove tracks whose covariance exceeds a maximum threshold (building is "lost")

## 12. Summary of EKF Parameters

| Parameter | Symbol | Default Value | Notes |
|-----------|--------|---------------|-------|
| Process noise | $q$ | 0.01 m^2/s | Diagonal elements of Q |
| Pixel noise (u) | $\sigma_u$ | 2 px | YOLOv8 bbox center accuracy |
| Pixel noise (v) | $\sigma_v$ | 2 px | YOLOv8 bbox center accuracy |
| Initial xy uncertainty | $\sigma_{x,0}, \sigma_{y,0}$ | 20 m | Single-view back-projection error |
| Initial z uncertainty | $\sigma_{z,0}$ | 10 m | Building height prior |
| Gating threshold | $\chi^2_{2,0.99}$ | 9.21 | 99% confidence, 2 DOF |

## 13. Information Content vs. Altitude

The Fisher Information Matrix (FIM) for a single measurement is:

$$
\mathbf{I}_k = \mathbf{H}_k^T \mathbf{R}^{-1} \mathbf{H}_k
$$

Since $\mathbf{H}_k \propto 1/Z_C$ and $Z_C \approx h$ (UAV altitude above building) for a nadir camera:

$$
\mathbf{I}_k \propto \frac{1}{h^2}
$$

This shows that **information per measurement scales inversely with altitude squared**. Halving the altitude quadruples the information content. This quantitative relationship is the theoretical foundation for the adaptive altitude controller designed in `altitude_control_law.md`.

However, higher altitude means larger FOV and more buildings visible simultaneously, so there is a fundamental **information-per-building vs. coverage trade-off** that the controller must balance.
