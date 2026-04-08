function test_ekf_convergence()
% TEST_EKF_CONVERGENCE Verify EKF converges to true building position.
%
%   Uses a circular UAV trajectory around a known building position.
%   Expects RMSE < 2m after sufficient measurements.

    fprintf('=== Test: EKF Convergence ===\n');

    % Add paths
    addpath('../utils', '../ekf');

    % Ground truth building position
    x_true = [50; 30; 0];

    % Camera parameters (typical downward-facing camera)
    K = [500 0 320; 0 500 240; 0 0 1];  % fx=fy=500, cx=320, cy=240
    R_BC = [1 0 0; 0 -1 0; 0 0 -1];     % Camera looking down (Z-down)
    t_BC = [0; 0; 0];

    % Generate circular trajectory around building
    params.center = [50; 30; -80];  % Center above building, NED (negative Z = up)
    params.radius = 60;
    params.speed = 5;
    params.duration = 80;
    params.dt = 0.5;
    [positions, rotations, timestamps] = generate_uav_trajectory('circular', params);

    % Generate synthetic measurements
    sigma_pixel = 2;
    rng(42);  % Reproducibility
    measurements = generate_synthetic_measurements(x_true, positions, rotations, ...
                                                    timestamps, K, R_BC, t_BC, sigma_pixel);

    % Initialize EKF with poor initial estimate
    x0 = x_true + [15; -10; 5];  % 20m off
    P0 = diag([400, 400, 100]);
    Q  = diag([0.01, 0.01, 0.01]);
    R_pixel = diag([sigma_pixel^2, sigma_pixel^2]);

    ekf = BuildingEKF(x0, P0, Q, R_pixel);

    % Run EKF
    errors = [];
    traces = [];
    for i = 1:length(measurements)
        if i > 1
            dt = measurements(i).time - measurements(i-1).time;
            ekf.predict(dt);
        end

        dets = measurements(i).detections;
        for d = 1:length(dets)
            if dets(d).building_id == 1
                ekf.update(dets(d).z_noisy, measurements(i).uav_pos, ...
                           measurements(i).uav_R, K, R_BC, t_BC);
            end
        end

        errors = [errors, ekf.position_error(x_true)]; %#ok<AGROW>
        traces = [traces, ekf.trace_P()]; %#ok<AGROW>
    end

    % Check convergence
    final_error = errors(end);
    fprintf('  Initial error: %.2f m\n', errors(1));
    fprintf('  Final error:   %.2f m\n', final_error);
    fprintf('  Final trace(P): %.2f\n', traces(end));
    fprintf('  Number of updates: %d\n', ekf.n_updates);

    assert(final_error < 2.0, 'EKF did not converge: final error = %.2f m', final_error);
    fprintf('  PASSED: Final error < 2m\n\n');
end
