function results = run_baseline_comparison()
% RUN_BASELINE_COMPARISON Compare pure odometry, fixed-altitude EKF, and adaptive EKF.
%
%   Experiment 1: Baseline comparison across methods.
%   - Pure odometry (single-frame inverse projection, QUASAR-style)
%   - Fixed-altitude EKF at 40m, 60m, 80m, 100m
%   - Adaptive EKF (proposed)
%
%   Runs 10 seeds each, lawnmower trajectory, 4x4 grid of buildings.

    fprintf('=== Experiment 1: Baseline Comparison ===\n');

    addpath('../utils', '../ekf', '../control', '../analysis');

    % Camera parameters
    K = [500 0 320; 0 500 240; 0 0 1];
    R_BC = [1 0 0; 0 -1 0; 0 0 -1];
    t_BC = [0; 0; 0];
    sigma_pixel = 2;
    R_pixel = diag([sigma_pixel^2, sigma_pixel^2]);

    % 4x4 grid of buildings
    buildings = generate_building_grid(4, 4, 30);  % 30m spacing
    M = size(buildings, 2);

    % EKF parameters
    P0 = diag([400, 400, 100]);
    Q  = diag([0.01, 0.01, 0.01]);

    % Methods to compare
    fixed_altitudes = [40, 60, 80, 100];
    n_seeds = 10;
    duration = 120;
    dt = 0.5;

    % Results storage
    results = struct();
    methods_list = {};

    % --- Pure Odometry Baseline ---
    fprintf('  Running: Pure Odometry...\n');
    odo_errors = zeros(M, n_seeds);
    for seed = 1:n_seeds
        rng(seed);
        params.altitude = -80;
        params.duration = duration;
        params.dt = dt;
        [pos, rot, ts] = generate_uav_trajectory('lawnmower', params);
        meas = generate_synthetic_measurements(buildings, pos, rot, ts, K, R_BC, t_BC, sigma_pixel);

        % Pure odometry: use last available measurement's back-projection
        for b = 1:M
            last_pixel = NaN(2,1);
            last_uav_pos = NaN(3,1);
            last_uav_R = eye(3);
            for i = 1:length(meas)
                for d = 1:length(meas(i).detections)
                    if meas(i).detections(d).building_id == b
                        last_pixel = meas(i).detections(d).z_noisy;
                        last_uav_pos = meas(i).uav_pos;
                        last_uav_R = meas(i).uav_R;
                    end
                end
            end
            if ~any(isnan(last_pixel))
                x_odo = back_project_to_ground(last_pixel, last_uav_pos, last_uav_R, K, R_BC, t_BC);
                odo_errors(b, seed) = norm(x_odo - buildings(:, b));
            else
                odo_errors(b, seed) = NaN;
            end
        end
    end
    results.odometry.errors = odo_errors;
    results.odometry.mean_rmse = nanmean(sqrt(nanmean(odo_errors.^2, 1)));
    methods_list{end+1} = 'Pure Odometry';
    fprintf('    Mean RMSE: %.2f m\n', results.odometry.mean_rmse);

    % --- Fixed-Altitude EKF ---
    for ai = 1:length(fixed_altitudes)
        alt = fixed_altitudes(ai);
        method_name = sprintf('EKF (h=%dm)', alt);
        fprintf('  Running: %s...\n', method_name);

        ekf_errors = zeros(M, n_seeds);
        for seed = 1:n_seeds
            rng(seed);
            params.altitude = -alt;
            params.duration = duration;
            params.dt = dt;
            [pos, rot, ts] = generate_uav_trajectory('lawnmower', params);
            meas = generate_synthetic_measurements(buildings, pos, rot, ts, K, R_BC, t_BC, sigma_pixel);

            % Run independent EKF per building
            for b = 1:M
                x0 = buildings(:, b) + [15*randn; 15*randn; 5*randn];
                ekf = BuildingEKF(x0, P0, Q, R_pixel);

                for i = 1:length(meas)
                    if i > 1
                        ekf.predict(meas(i).time - meas(i-1).time);
                    end
                    for d = 1:length(meas(i).detections)
                        if meas(i).detections(d).building_id == b
                            ekf.update(meas(i).detections(d).z_noisy, ...
                                       meas(i).uav_pos, meas(i).uav_R, K, R_BC, t_BC);
                        end
                    end
                end
                ekf_errors(b, seed) = ekf.position_error(buildings(:, b));
            end
        end

        field_name = sprintf('ekf_fixed_%d', alt);
        results.(field_name).errors = ekf_errors;
        results.(field_name).mean_rmse = mean(sqrt(mean(ekf_errors.^2, 1)));
        methods_list{end+1} = method_name; %#ok<AGROW>
        fprintf('    Mean RMSE: %.2f m\n', results.(field_name).mean_rmse);
    end

    % --- Adaptive-Altitude EKF (Proposed) ---
    fprintf('  Running: Adaptive EKF...\n');
    ctrl_params.h_min = 20;
    ctrl_params.h_max = 120;
    ctrl_params.alpha = 3.0;
    ctrl_params.J_ref = trace(P0);
    ctrl_params.tau = 5;
    ctrl_params.h_dot_max = 3;

    adaptive_errors = zeros(M, n_seeds);
    for seed = 1:n_seeds
        rng(seed);

        % Start at high altitude
        h_current = 80;
        params.altitude = -h_current;
        params.duration = duration;
        params.dt = dt;
        [pos, rot, ts] = generate_uav_trajectory('lawnmower', params);

        % Initialize tracker
        tracker = MultiBuildingTracker(Q, R_pixel, P0);

        % Simulate with adaptive altitude
        for i = 1:length(ts)
            if i > 1
                dt_step = ts(i) - ts(i-1);
                tracker.predict_all(dt_step);

                % Update altitude based on covariance
                [h_current, ~, ~] = altitude_controller(tracker, h_current, dt_step, ctrl_params);

                % Adjust UAV position altitude
                pos(3, i) = -h_current;
            end

            % Generate measurements at current altitude
            [z_pix, is_valid] = deal(cell(1, M));
            for b = 1:M
                [z_pix{b}, is_valid{b}] = camera_project(buildings(:, b), pos(:,i), rot{i}, K, R_BC, t_BC);
            end

            % Create detections
            dets = [];
            for b = 1:M
                if is_valid{b}
                    det.building_id = b;
                    det.z_noisy = z_pix{b} + sigma_pixel * randn(2, 1);
                    dets = [dets, det]; %#ok<AGROW>
                end
            end

            tracker.update_all(dets, pos(:, i), rot{i}, K, R_BC, t_BC);
        end

        [ids, states, ~] = tracker.get_all_states();
        for b = 1:M
            idx = find(ids == b, 1);
            if ~isempty(idx)
                adaptive_errors(b, seed) = norm(states(:, idx) - buildings(:, b));
            else
                adaptive_errors(b, seed) = NaN;
            end
        end
    end

    results.adaptive.errors = adaptive_errors;
    results.adaptive.mean_rmse = nanmean(sqrt(nanmean(adaptive_errors.^2, 1)));
    methods_list{end+1} = 'Adaptive EKF';
    fprintf('    Mean RMSE: %.2f m\n', results.adaptive.mean_rmse);

    % --- Print Summary ---
    fprintf('\n--- Summary ---\n');
    fprintf('%-20s  Mean RMSE (m)\n', 'Method');
    fprintf('%-20s  %.2f\n', 'Pure Odometry', results.odometry.mean_rmse);
    for ai = 1:length(fixed_altitudes)
        field_name = sprintf('ekf_fixed_%d', fixed_altitudes(ai));
        fprintf('%-20s  %.2f\n', sprintf('EKF (h=%dm)', fixed_altitudes(ai)), ...
                results.(field_name).mean_rmse);
    end
    fprintf('%-20s  %.2f\n', 'Adaptive EKF', results.adaptive.mean_rmse);

    % Save results
    save('../../data/results/baseline_comparison.mat', 'results', 'methods_list');
    fprintf('\nResults saved to data/results/baseline_comparison.mat\n');
end

%% Helper functions

function buildings = generate_building_grid(rows, cols, spacing)
% Generate a grid of building positions at z=0
    buildings = zeros(3, rows * cols);
    idx = 0;
    for r = 1:rows
        for c = 1:cols
            idx = idx + 1;
            buildings(:, idx) = [(c - (cols+1)/2) * spacing;
                                 (r - (rows+1)/2) * spacing;
                                 0];
        end
    end
end

function x0 = back_project_to_ground(z_pixel, uav_pos, uav_R, K, R_BC, t_BC)
% Back-project pixel to ground plane (z=0)
    fx = K(1,1); fy = K(2,2);
    cx = K(1,3); cy = K(2,3);

    x_norm = (z_pixel(1) - cx) / fx;
    y_norm = (z_pixel(2) - cy) / fy;
    ray_C = [x_norm; y_norm; 1];

    R_WC = R_BC * uav_R;
    ray_W = R_WC * ray_C;
    t_WC = R_BC * uav_pos + t_BC;

    if abs(ray_W(3)) < 1e-10
        x0 = [uav_pos(1); uav_pos(2); 0];
    else
        lambda = -t_WC(3) / ray_W(3);
        x0 = t_WC + lambda * ray_W;
    end
end
