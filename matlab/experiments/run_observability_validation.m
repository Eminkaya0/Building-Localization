function results = run_observability_validation()
% RUN_OBSERVABILITY_VALIDATION Validate observability analysis with EKF.
%
%   Experiment 3: Compare circular orbit (good parallax) vs straight-line
%   toward buildings (degenerate). Demonstrates that observability theory
%   predicts EKF convergence behavior.

    fprintf('=== Experiment 3: Observability Validation ===\n');

    addpath('../utils', '../ekf', '../analysis');

    x_true = [50; 30; 0];
    K = [500 0 320; 0 500 240; 0 0 1];
    R_BC = [1 0 0; 0 -1 0; 0 0 -1];
    t_BC = [0; 0; 0];
    sigma_pixel = 2;
    R_pixel = diag([sigma_pixel^2, sigma_pixel^2]);
    P0 = diag([400, 400, 100]);
    Q  = diag([0.01, 0.01, 0.01]);

    duration = 80;
    dt = 0.5;

    % --- Trajectory A: Circular orbit (good) ---
    params_circ.center = [50; 30; -80];
    params_circ.radius = 50;
    params_circ.speed = 5;
    params_circ.duration = duration;
    params_circ.dt = dt;
    [pos_circ, rot_circ, ts_circ] = generate_uav_trajectory('circular', params_circ);

    % --- Trajectory B: Straight line toward building (degenerate) ---
    params_str.start_pos = [-50; 30; -80];
    params_str.direction = [1; 0; 0];
    params_str.speed = 5;
    params_str.duration = duration;
    params_str.dt = dt;
    [pos_str, rot_str, ts_str] = generate_uav_trajectory('straight', params_str);

    % Observability Gramian analysis
    [O_circ, eig_circ, evec_circ] = compute_observability_gramian(x_true, pos_circ, rot_circ, K, R_BC, t_BC, R_pixel);
    [O_str, eig_str, evec_str]    = compute_observability_gramian(x_true, pos_str, rot_str, K, R_BC, t_BC, R_pixel);

    [obs_circ, ~, cond_circ] = check_observability(O_circ);
    [obs_str, ~, cond_str]   = check_observability(O_str);

    fprintf('  Circular:  observable=%d, cond=%.1f\n', obs_circ, cond_circ);
    fprintf('  Straight:  observable=%d, cond=%.1f\n', obs_str, cond_str);

    % Run EKF on both trajectories
    n_seeds = 10;
    errors_circ = zeros(n_seeds, length(ts_circ));
    errors_str  = zeros(n_seeds, length(ts_str));
    traces_circ = zeros(n_seeds, length(ts_circ));
    traces_str  = zeros(n_seeds, length(ts_str));

    for seed = 1:n_seeds
        rng(seed);
        meas_circ = generate_synthetic_measurements(x_true, pos_circ, rot_circ, ts_circ, K, R_BC, t_BC, sigma_pixel);
        meas_str  = generate_synthetic_measurements(x_true, pos_str, rot_str, ts_str, K, R_BC, t_BC, sigma_pixel);

        x0 = x_true + [15; -10; 5];

        % Circular
        ekf_c = BuildingEKF(x0, P0, Q, R_pixel);
        for i = 1:length(meas_circ)
            if i > 1; ekf_c.predict(meas_circ(i).time - meas_circ(i-1).time); end
            for d = 1:length(meas_circ(i).detections)
                if meas_circ(i).detections(d).building_id == 1
                    ekf_c.update(meas_circ(i).detections(d).z_noisy, meas_circ(i).uav_pos, meas_circ(i).uav_R, K, R_BC, t_BC);
                end
            end
            errors_circ(seed, i) = ekf_c.position_error(x_true);
            traces_circ(seed, i) = ekf_c.trace_P();
        end

        % Straight
        ekf_s = BuildingEKF(x0, P0, Q, R_pixel);
        for i = 1:length(meas_str)
            if i > 1; ekf_s.predict(meas_str(i).time - meas_str(i-1).time); end
            for d = 1:length(meas_str(i).detections)
                if meas_str(i).detections(d).building_id == 1
                    ekf_s.update(meas_str(i).detections(d).z_noisy, meas_str(i).uav_pos, meas_str(i).uav_R, K, R_BC, t_BC);
                end
            end
            errors_str(seed, i) = ekf_s.position_error(x_true);
            traces_str(seed, i) = ekf_s.trace_P();
        end
    end

    % Store results
    results.circular.errors = errors_circ;
    results.circular.traces = traces_circ;
    results.circular.obs_gramian = O_circ;
    results.circular.eigenvalues = eig_circ;
    results.circular.cond = cond_circ;

    results.straight.errors = errors_str;
    results.straight.traces = traces_str;
    results.straight.obs_gramian = O_str;
    results.straight.eigenvalues = eig_str;
    results.straight.cond = cond_str;

    results.timestamps = ts_circ;

    % Summary
    fprintf('  Circular final error (mean): %.2f m\n', mean(errors_circ(:, end)));
    fprintf('  Straight final error (mean): %.2f m\n', mean(errors_str(:, end)));

    % Plot
    figure('Name', 'Observability Validation');

    subplot(2,1,1);
    hold on;
    plot(ts_circ, mean(errors_circ, 1), 'b-', 'LineWidth', 2);
    plot(ts_str, mean(errors_str, 1), 'r-', 'LineWidth', 2);
    xlabel('Time (s)'); ylabel('Position Error (m)');
    title('EKF Convergence: Circular vs Straight Trajectory');
    legend('Circular (good parallax)', 'Straight (degenerate)');
    grid on;

    subplot(2,1,2);
    hold on;
    plot(ts_circ, mean(traces_circ, 1), 'b-', 'LineWidth', 2);
    plot(ts_str, mean(traces_str, 1), 'r-', 'LineWidth', 2);
    xlabel('Time (s)'); ylabel('trace(P)');
    title('Covariance Trace Over Time');
    legend('Circular', 'Straight');
    grid on;

    save('../../data/results/observability_validation.mat', 'results');
    fprintf('\nResults saved to data/results/observability_validation.mat\n');
end
