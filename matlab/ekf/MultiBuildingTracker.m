classdef MultiBuildingTracker < handle
% MULTIBUILDINGTRACKER Track multiple buildings using independent EKF instances.
%
%   Maintains a dictionary of BuildingEKF objects, one per tracked building.
%   Uses nearest-neighbor data association in pixel space.
%
%   Usage:
%       tracker = MultiBuildingTracker(Q, R_pixel, P0_default, gate_threshold);
%       tracker.predict_all(dt);
%       tracker.update_all(detections, uav_pos, uav_R, K, R_BC, t_BC);
%       [states, covs] = tracker.get_all_states();
%
%   See also: BuildingEKF

    properties
        filters         % containers.Map of building_id -> BuildingEKF
        Q               % [3x3] Process noise covariance
        R_meas          % [2x2] Measurement noise covariance
        P0_default      % [3x3] Default initial covariance
        gate_threshold  % Mahalanobis gating threshold
        max_covariance  % Maximum trace(P) before track deletion
    end

    methods
        function obj = MultiBuildingTracker(Q, R_pixel, P0_default, gate_threshold, max_covariance)
        % Constructor.
        %
        %   tracker = MultiBuildingTracker(Q, R_pixel, P0_default, gate_threshold, max_covariance)
            obj.filters = containers.Map('KeyType', 'int32', 'ValueType', 'any');
            obj.Q = Q;
            obj.R_meas = R_pixel;
            obj.P0_default = P0_default;
            if nargin < 4; gate_threshold = 9.21; end
            if nargin < 5; max_covariance = 1e6; end
            obj.gate_threshold = gate_threshold;
            obj.max_covariance = max_covariance;
        end

        function predict_all(obj, dt)
        % PREDICT_ALL Run prediction step for all tracked buildings.
            keys = obj.filters.keys();
            for i = 1:length(keys)
                obj.filters(keys{i}).predict(dt);
            end
        end

        function update_all(obj, detections, uav_pos, uav_R, K, R_BC, t_BC)
        % UPDATE_ALL Process all detections and update corresponding EKFs.
        %
        %   detections - struct array with fields:
        %       .building_id - Integer ID (used as tracker key)
        %       .z_noisy     - [2x1] Noisy pixel measurement
        %
        %   If a building_id is new, a new EKF is initialized using back-projection.
            for d = 1:length(detections)
                det = detections(d);
                bid = int32(det.building_id);

                if ~obj.filters.isKey(bid)
                    % Initialize new track via back-projection
                    x0 = obj.back_project_init(det.z_noisy, uav_pos, uav_R, K, R_BC, t_BC);
                    ekf = BuildingEKF(x0, obj.P0_default, obj.Q, obj.R_meas, obj.gate_threshold);
                    obj.filters(bid) = ekf;
                end

                obj.filters(bid).update(det.z_noisy, uav_pos, uav_R, K, R_BC, t_BC);
            end

            % Prune tracks with excessive covariance
            obj.prune_tracks();
        end

        function [ids, states, covs] = get_all_states(obj)
        % GET_ALL_STATES Return all tracked building states.
        %
        %   [ids, states, covs] = tracker.get_all_states()
        %
        %   Outputs:
        %       ids    - [1xN] Building IDs
        %       states - [3xN] State estimates
        %       covs   - {1xN} Cell array of 3x3 covariance matrices
            keys = obj.filters.keys();
            N = length(keys);
            ids = zeros(1, N);
            states = zeros(3, N);
            covs = cell(1, N);

            for i = 1:N
                ids(i) = keys{i};
                [states(:, i), covs{i}] = obj.filters(keys{i}).get_state();
            end
        end

        function J = mean_trace(obj)
        % MEAN_TRACE Average covariance trace across all tracked buildings.
            keys = obj.filters.keys();
            N = length(keys);
            if N == 0
                J = Inf;
                return;
            end
            total = 0;
            for i = 1:N
                total = total + obj.filters(keys{i}).trace_P();
            end
            J = total / N;
        end

        function N = num_tracks(obj)
        % NUM_TRACKS Number of currently tracked buildings.
            N = obj.filters.Count;
        end
    end

    methods (Access = private)
        function x0 = back_project_init(~, z_pixel, uav_pos, uav_R, K, R_BC, t_BC)
        % BACK_PROJECT_INIT Initialize building position by back-projecting pixel
        %   to ground plane (z_building = 0 assumed).

            fx = K(1,1); fy = K(2,2);
            cx = K(1,3); cy = K(2,3);

            % Pixel to normalized camera coordinates
            x_norm = (z_pixel(1) - cx) / fx;
            y_norm = (z_pixel(2) - cy) / fy;
            ray_C = [x_norm; y_norm; 1];

            % Ray in world frame
            R_WC = R_BC * uav_R;
            ray_W = R_WC * ray_C;

            % Camera position in world frame
            t_WC = R_BC * uav_pos + t_BC;

            % Intersect with z=0 plane (ground level)
            % t_WC + lambda * ray_W => z = 0
            if abs(ray_W(3)) < 1e-10
                % Ray parallel to ground, use UAV position projection
                x0 = [uav_pos(1); uav_pos(2); 0];
            else
                lambda = -t_WC(3) / ray_W(3);
                x0 = t_WC + lambda * ray_W;
            end
        end

        function prune_tracks(obj)
        % PRUNE_TRACKS Remove tracks with excessively large covariance.
            keys = obj.filters.keys();
            for i = 1:length(keys)
                if obj.filters(keys{i}).trace_P() > obj.max_covariance
                    obj.filters.remove(keys{i});
                end
            end
        end
    end
end
