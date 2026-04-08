function run_all_tests()
% RUN_ALL_TESTS Execute all unit tests for the BLOCS project.
%
%   Usage: run_all_tests()
%   Run from the matlab/tests/ directory.

    fprintf('============================================\n');
    fprintf('  BLOCS Unit Tests\n');
    fprintf('============================================\n\n');

    % Add all paths
    addpath('../utils', '../ekf', '../control', '../analysis');

    tests = {'test_ekf_convergence', ...
             'test_covariance_shrinks', ...
             'test_missing_measurements', ...
             'test_degenerate_trajectory', ...
             'test_altitude_controller'};

    n_passed = 0;
    n_failed = 0;
    failed_tests = {};

    for i = 1:length(tests)
        try
            feval(tests{i});
            n_passed = n_passed + 1;
        catch ME
            n_failed = n_failed + 1;
            failed_tests{end+1} = tests{i}; %#ok<AGROW>
            fprintf('  FAILED: %s\n', ME.message);
            fprintf('          in %s (line %d)\n\n', ME.stack(1).name, ME.stack(1).line);
        end
    end

    fprintf('============================================\n');
    fprintf('  Results: %d passed, %d failed\n', n_passed, n_failed);
    if n_failed > 0
        fprintf('  Failed tests:\n');
        for i = 1:length(failed_tests)
            fprintf('    - %s\n', failed_tests{i});
        end
    end
    fprintf('============================================\n');
end
