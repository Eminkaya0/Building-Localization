# blocs_estimation

EKF implementation for building position estimation.

## Responsibilities

- BuildingEKF class: predict/update cycle for a single building
- Multi-building tracking with dictionary of EKF instances
- Measurement gating (Mahalanobis distance)
- Observability metric computation

## Design

The EKF is implemented in **pure Python** with no ROS dependencies, enabling standalone unit testing. ROS integration is handled by a separate node wrapper.
