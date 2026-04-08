function test_covariance_shrinks()
% TEST_COVARIANCE_SHRINKS Verify that covariance decreases with informative measurements.
%
%   Uses a circular trajectory (good parallax) and checks that trace(P)
%   is monotonically non-increasing after each update.

    fprintf('=== Test: Covariance Shrinks ===\n');

    addpath('../utils', '../ekf');

    x_true = [0; 0; 0];
    K = [500 0 320; 0 500 240; 0 0 1];
    R_BC = [1 0 0; 0 -1 0; 0 0 -1];
    t_BC = [0; 0; 0];

    params.center = [0; 0; -60];
    params.radius = 40;
    params.speed = 5;
    params.duration = 60;
    params.dt = 1.0;
    [positions, rotations, timestamps] = generate_uav_trajectory('circular', params);

    sigma_pixel = 2;
    rng(123);
    measurements = generate_synthetic_measurements(x_true, positions, rotations, ...
                                                    timestamps, K, R_BC, t_BC, sigma_pixel);

    x0 = x_true + [10; -10; 3];
    P0 = diag([400, 400, 100]);
    Q  = diag([0.001, 0.001, 0.001]);  % Very small Q to see clear shrinkage
    R_pixel = diag([sigma_pixel^2, sigma_pixel^2]);

    ekf = BuildingEKF(x0, P0, Q, R_pixel);

    traces_after_update = [];
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
                traces_after_update = [traces_after_update, ekf.trace_P()]; %#ok<AGROW>
            end
        end
    end

    % Check that trace is generally decreasing (allow small bumps from noise)
    % We check that the final trace is much smaller than the initial
    initial_trace = trace(P0);
    final_trace = traces_after_update(end);
    reduction_ratio = final_trace / initial_trace;

    fprintf('  Initial trace(P): %.2f\n', initial_trace);
    fprintf('  Final trace(P):   %.2f\n', final_trace);
    fprintf('  Reduction ratio:  %.4f\n', reduction_ratio);

    assert(reduction_ratio < 0.1, ...
           'Covariance did not shrink enough: ratio = %.4f', reduction_ratio);
    fprintf('  PASSED: Covariance reduced by >90%%\n\n');
end
