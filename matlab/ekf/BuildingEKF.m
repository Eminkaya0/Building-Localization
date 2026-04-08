classdef BuildingEKF < handle
% BUILDINGEKF Extended Kalman Filter for static building position estimation.
%
%   Estimates [x_b, y_b, z_b]^T of a static building in the world frame
%   using sequential monocular camera measurements (bounding box centers).
%
%   Usage:
%       ekf = BuildingEKF(x0, P0, Q, R_pixel);
%       ekf.predict(dt);
%       accepted = ekf.update(z_pixel, uav_pos, uav_R, K, R_BC, t_BC);
%       [x_hat, P] = ekf.get_state();
%
%   See also: MultiBuildingTracker, camera_project, camera_project_jacobian

    properties
        x       % [3x1] State estimate (building position in world frame)
        P       % [3x3] State covariance matrix
        Q       % [3x3] Process noise covariance (per second)
        R_meas  % [2x2] Measurement noise covariance (pixel^2)
        gate_threshold  % Mahalanobis distance squared threshold for gating
        n_updates       % Number of successful updates
    end

    methods
        function obj = BuildingEKF(x0, P0, Q, R_pixel, gate_threshold)
        % BUILDINGEKF Constructor.
        %
        %   ekf = BuildingEKF(x0, P0, Q, R_pixel, gate_threshold)
        %
        %   Inputs:
        %       x0             - [3x1] Initial state estimate
        %       P0             - [3x3] Initial covariance
        %       Q              - [3x3] Process noise covariance (per second)
        %       R_pixel        - [2x2] Measurement noise covariance (pixel^2)
        %       gate_threshold - Mahalanobis distance^2 threshold (default: 9.21)
            obj.x = x0(:);
            obj.P = P0;
            obj.Q = Q;
            obj.R_meas = R_pixel;
            if nargin < 5
                obj.gate_threshold = 9.21;  % chi2(2, 0.99)
            else
                obj.gate_threshold = gate_threshold;
            end
            obj.n_updates = 0;
        end

        function predict(obj, dt)
        % PREDICT EKF prediction step (static building model).
        %
        %   ekf.predict(dt)
        %
        %   Since the building is static, the state doesn't change.
        %   Only the covariance grows by Q*dt.
            obj.P = obj.P + obj.Q * dt;
        end

        function [accepted, innovation, S] = update(obj, z_pixel, uav_pos, uav_R, K, R_BC, t_BC)
        % UPDATE EKF measurement update step.
        %
        %   [accepted, innovation, S] = ekf.update(z_pixel, uav_pos, uav_R, K, R_BC, t_BC)
        %
        %   Inputs:
        %       z_pixel - [2x1] Measured pixel coordinates [u; v]
        %       uav_pos - [3x1] UAV position in world frame
        %       uav_R   - [3x3] UAV rotation (body-to-world)
        %       K       - [3x3] Camera intrinsic matrix
        %       R_BC    - [3x3] Body-to-camera rotation
        %       t_BC    - [3x1] Body-to-camera translation
        %
        %   Outputs:
        %       accepted   - logical, true if measurement passed gating
        %       innovation - [2x1] Measurement residual (y = z - h(x))
        %       S          - [2x2] Innovation covariance

            % Predicted measurement
            [z_pred, is_valid] = camera_project(obj.x, uav_pos, uav_R, K, R_BC, t_BC);

            if ~is_valid
                accepted = false;
                innovation = [NaN; NaN];
                S = NaN(2);
                return;
            end

            % Measurement Jacobian
            H = camera_project_jacobian(obj.x, uav_pos, uav_R, K, R_BC, t_BC);

            % Innovation
            innovation = z_pixel(:) - z_pred;

            % Innovation covariance
            S = H * obj.P * H' + obj.R_meas;

            % Mahalanobis gating
            d2 = innovation' * (S \ innovation);
            if d2 > obj.gate_threshold
                accepted = false;
                return;
            end

            % Kalman gain
            K_gain = obj.P * H' / S;

            % State update
            obj.x = obj.x + K_gain * innovation;

            % Covariance update (Joseph form for numerical stability)
            I_KH = eye(3) - K_gain * H;
            obj.P = I_KH * obj.P * I_KH' + K_gain * obj.R_meas * K_gain';

            % Ensure symmetry
            obj.P = (obj.P + obj.P') / 2;

            accepted = true;
            obj.n_updates = obj.n_updates + 1;
        end

        function [x_hat, P_out] = get_state(obj)
        % GET_STATE Return current state estimate and covariance.
            x_hat = obj.x;
            P_out = obj.P;
        end

        function tr = trace_P(obj)
        % TRACE_P Return trace of covariance matrix.
            tr = trace(obj.P);
        end

        function d = position_error(obj, x_true)
        % POSITION_ERROR Euclidean distance between estimate and truth.
            d = norm(obj.x - x_true(:));
        end
    end
end
