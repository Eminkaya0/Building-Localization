function test_missing_measurements()
% TEST_MISSING_MEASUREMENTS Verify EKF handles measurement gaps correctly.
%
%   Simulates periods where the building is not visible (e.g., out of FOV).
%   Covariance should grow during gaps and shrink when measurements resume.

    fprintf('=== Test: Missing Measurements ===\n');

    addpath('../utils', '../ekf');

    x_true = [30; 20; 0];
    K = [500 0 320; 0 500 240; 0 0 1];
    R_BC = [1 0 0; 0 -1 0; 0 0 -1];
    t_BC = [0; 0; 0];

    params.center = [30; 20; -80];
    params.radius = 50;
    params.speed = 5;
    params.duration = 100;
    params.dt = 0.5;
    [positions, rotations, timestamps] = generate_uav_trajectory('circular', params);

    sigma_pixel = 2;
    rng(7);
    measurements = generate_synthetic_measurements(x_true, positions, rotations, ...
                                                    timestamps, K, R_BC, t_BC, sigma_pixel);

    x0 = x_true + [10; -8; 3];
    P0 = diag([400, 400, 100]);
    Q  = diag([0.01, 0.01, 0.01]);
    R_pixel = diag([sigma_pixel^2, sigma_pixel^2]);

    ekf = BuildingEKF(x0, P0, Q, R_pixel);

    % Run EKF but skip measurements in the middle 30% of the trajectory
    N = length(measurements);
    gap_start = round(0.35 * N);
    gap_end   = round(0.65 * N);

    trace_before_gap = NaN;
    trace_at_gap_start = NaN;
    trace_at_gap_end = NaN;

    for i = 1:N
        if i > 1
            dt = measurements(i).time - measurements(i-1).time;
            ekf.predict(dt);
        end

        % Skip updates during gap
        if i >= gap_start && i <= gap_end
            if i == gap_start
                trace_at_gap_start = ekf.trace_P();
            end
            if i == gap_end
                trace_at_gap_end = ekf.trace_P();
            end
            continue;
        end

        if i == gap_start - 1
            trace_before_gap = ekf.trace_P();
        end

        dets = measurements(i).detections;
        for d = 1:length(dets)
            if dets(d).building_id == 1
                ekf.update(dets(d).z_noisy, measurements(i).uav_pos, ...
                           measurements(i).uav_R, K, R_BC, t_BC);
            end
        end
    end

    final_error = ekf.position_error(x_true);

    fprintf('  Trace before gap: %.2f\n', trace_before_gap);
    fprintf('  Trace at gap start: %.2f\n', trace_at_gap_start);
    fprintf('  Trace at gap end:   %.2f\n', trace_at_gap_end);
    fprintf('  Final error: %.2f m\n', final_error);

    % Covariance should grow during gap
    assert(trace_at_gap_end > trace_at_gap_start, ...
           'Covariance should grow during measurement gap');

    % EKF should still converge after gap
    assert(final_error < 5.0, ...
           'EKF did not recover after gap: error = %.2f m', final_error);

    fprintf('  PASSED: Covariance grew during gap, EKF recovered after\n\n');
end
