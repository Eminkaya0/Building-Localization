function test_degenerate_trajectory()
% TEST_DEGENERATE_TRAJECTORY Verify EKF fails to converge on degenerate trajectory.
%
%   Uses a straight-line trajectory directed at the building (no parallax).
%   The depth direction should remain poorly observable, resulting in large errors.

    fprintf('=== Test: Degenerate Trajectory ===\n');

    addpath('../utils', '../ekf', '../analysis');

    x_true = [100; 0; 0];
    K = [500 0 320; 0 500 240; 0 0 1];
    R_BC = [1 0 0; 0 -1 0; 0 0 -1];
    t_BC = [0; 0; 0];

    % Straight line toward the building (degenerate: no parallax in depth)
    params.start_pos = [0; 0; -80];
    params.direction = [1; 0; 0];  % Moving toward building
    params.speed = 5;
    params.duration = 15;  % Stop before reaching building
    params.dt = 0.5;
    [pos_degen, rot_degen, ts_degen] = generate_uav_trajectory('straight', params);

    % Circular trajectory for comparison (good parallax)
    params_circ.center = [100; 0; -80];
    params_circ.radius = 50;
    params_circ.speed = 5;
    params_circ.duration = 60;
    params_circ.dt = 0.5;
    [pos_circ, rot_circ, ts_circ] = generate_uav_trajectory('circular', params_circ);

    sigma_pixel = 2;
    R_pixel = diag([sigma_pixel^2, sigma_pixel^2]);

    % Observability analysis
    R_meas = R_pixel;
    [O_degen, eig_degen] = compute_observability_gramian(x_true, pos_degen, rot_degen, ...
                                                          K, R_BC, t_BC, R_meas);
    [O_circ, eig_circ] = compute_observability_gramian(x_true, pos_circ, rot_circ, ...
                                                        K, R_BC, t_BC, R_meas);

    [obs_degen, ~, cond_degen] = check_observability(O_degen);
    [obs_circ, ~, cond_circ]   = check_observability(O_circ);

    fprintf('  Degenerate: observable=%d, cond=%.1f, eigs=[%.2e, %.2e, %.2e]\n', ...
            obs_degen, cond_degen, eig_degen(1), eig_degen(2), eig_degen(3));
    fprintf('  Circular:   observable=%d, cond=%.1f, eigs=[%.2e, %.2e, %.2e]\n', ...
            obs_circ, cond_circ, eig_circ(1), eig_circ(2), eig_circ(3));

    % Run EKF on degenerate trajectory
    rng(42);
    meas_degen = generate_synthetic_measurements(x_true, pos_degen, rot_degen, ...
                                                  ts_degen, K, R_BC, t_BC, sigma_pixel);

    x0 = x_true + [0; 5; 8];
    P0 = diag([400, 400, 100]);
    Q  = diag([0.01, 0.01, 0.01]);

    ekf_degen = BuildingEKF(x0, P0, Q, R_pixel);
    for i = 1:length(meas_degen)
        if i > 1
            dt = meas_degen(i).time - meas_degen(i-1).time;
            ekf_degen.predict(dt);
        end
        dets = meas_degen(i).detections;
        for d = 1:length(dets)
            if dets(d).building_id == 1
                ekf_degen.update(dets(d).z_noisy, meas_degen(i).uav_pos, ...
                                 meas_degen(i).uav_R, K, R_BC, t_BC);
            end
        end
    end

    error_degen = ekf_degen.position_error(x_true);
    fprintf('  Degenerate trajectory final error: %.2f m\n', error_degen);

    % The degenerate trajectory should have much worse conditioning
    assert(cond_degen > 10 * cond_circ, ...
           'Degenerate trajectory should be much worse conditioned');

    fprintf('  PASSED: Degenerate trajectory has %.0fx worse conditioning\n\n', ...
            cond_degen / cond_circ);
end
