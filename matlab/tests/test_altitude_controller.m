function test_altitude_controller()
% TEST_ALTITUDE_CONTROLLER Verify altitude controller behavior.
%
%   Checks:
%   1. High uncertainty -> controller commands low altitude
%   2. Low uncertainty -> controller commands high altitude
%   3. Rate limiting works correctly
%   4. Altitude stays within bounds

    fprintf('=== Test: Altitude Controller ===\n');

    addpath('../control');

    params.h_min = 20;
    params.h_max = 120;
    params.alpha = 3.0;
    params.J_ref = 900;
    params.tau = 5;
    params.h_dot_max = 3;

    % Test 1: High uncertainty -> low altitude
    J_high = 900;  % Initial trace(P0) = 900
    h_current = 80;
    [~, h_des_high] = altitude_controller(J_high, h_current, 0.1, params);

    fprintf('  High uncertainty (J=%.0f): h_des = %.1f m\n', J_high, h_des_high);
    assert(h_des_high < 40, 'High uncertainty should command low altitude');

    % Test 2: Low uncertainty -> high altitude
    J_low = 5;  % Well-converged
    [~, h_des_low] = altitude_controller(J_low, h_current, 0.1, params);

    fprintf('  Low uncertainty (J=%.0f):  h_des = %.1f m\n', J_low, h_des_low);
    assert(h_des_low > 100, 'Low uncertainty should command high altitude');

    % Test 3: Rate limiting
    h_current = 80;
    dt = 1.0;
    [h_cmd, h_des] = altitude_controller(J_high, h_current, dt, params);

    max_change = params.h_dot_max * dt;
    actual_change = abs(h_cmd - h_current);

    fprintf('  Rate limit test: h_des=%.1f, h_cmd=%.1f, change=%.2f (max=%.1f)\n', ...
            h_des, h_cmd, actual_change, max_change);
    assert(actual_change <= max_change + 1e-10, 'Rate limit violated');

    % Test 4: Altitude bounds
    [h_cmd_low] = altitude_controller(1e6, params.h_min, 1.0, params);
    [h_cmd_high] = altitude_controller(0.001, params.h_max, 1.0, params);

    fprintf('  Bounds test: h_cmd_low=%.1f (min=%.0f), h_cmd_high=%.1f (max=%.0f)\n', ...
            h_cmd_low, params.h_min, h_cmd_high, params.h_max);
    assert(h_cmd_low >= params.h_min, 'Below minimum altitude');
    assert(h_cmd_high <= params.h_max, 'Above maximum altitude');

    fprintf('  PASSED: All altitude controller tests\n\n');
end
