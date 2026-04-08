function results = run_noise_sensitivity()
% RUN_NOISE_SENSITIVITY Evaluate EKF robustness to measurement noise.
%
%   Experiment 4: Vary pixel noise sigma from 1 to 10 and measure RMSE.

    fprintf('=== Experiment 4: Noise Sensitivity ===\n');

    addpath('../utils', '../ekf');

    x_true = [50; 30; 0];
    K = [500 0 320; 0 500 240; 0 0 1];
    R_BC = [1 0 0; 0 -1 0; 0 0 -1];
    t_BC = [0; 0; 0];

    P0 = diag([400, 400, 100]);
    Q  = diag([0.01, 0.01, 0.01]);

    params.center = [50; 30; -80];
    params.radius = 50;
    params.speed = 5;
    params.duration = 80;
    params.dt = 0.5;
    [positions, rotations, timestamps] = generate_uav_trajectory('circular', params);

    sigma_values = [1, 2, 3, 5, 7, 10];
    n_seeds = 10;
    n_sigma = length(sigma_values);

    final_rmse = zeros(n_sigma, n_seeds);

    for si = 1:n_sigma
        sigma = sigma_values(si);
        R_pixel = diag([sigma^2, sigma^2]);
        fprintf('  sigma = %d px: ', sigma);

        for seed = 1:n_seeds
            rng(seed);
            meas = generate_synthetic_measurements(x_true, positions, rotations, ...
                                                    timestamps, K, R_BC, t_BC, sigma);

            x0 = x_true + [15; -10; 5];
            ekf = BuildingEKF(x0, P0, Q, R_pixel);

            for i = 1:length(meas)
                if i > 1; ekf.predict(meas(i).time - meas(i-1).time); end
                for d = 1:length(meas(i).detections)
                    if meas(i).detections(d).building_id == 1
                        ekf.update(meas(i).detections(d).z_noisy, meas(i).uav_pos, ...
                                   meas(i).uav_R, K, R_BC, t_BC);
                    end
                end
            end

            final_rmse(si, seed) = ekf.position_error(x_true);
        end

        fprintf('mean RMSE = %.2f m (std = %.2f)\n', mean(final_rmse(si,:)), std(final_rmse(si,:)));
    end

    results.sigma_values = sigma_values;
    results.final_rmse = final_rmse;
    results.mean_rmse = mean(final_rmse, 2);
    results.std_rmse = std(final_rmse, 0, 2);

    % Plot
    figure('Name', 'Noise Sensitivity');
    errorbar(sigma_values, results.mean_rmse, results.std_rmse, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Pixel Noise \sigma (px)');
    ylabel('Final RMSE (m)');
    title('EKF Localization Error vs. Measurement Noise');
    grid on;

    save('../../data/results/noise_sensitivity.mat', 'results');
    fprintf('\nResults saved to data/results/noise_sensitivity.mat\n');
end
