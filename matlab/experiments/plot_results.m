function plot_results()
% PLOT_RESULTS Generate publication-quality figures from experiment results.
%
%   Loads results from data/results/ and creates figures suitable for
%   IEEE conference/journal submission.

    fprintf('=== Generating Publication Figures ===\n');

    results_dir = '../../data/results';

    % Set default figure properties for publication
    set(0, 'DefaultAxesFontSize', 12);
    set(0, 'DefaultTextFontSize', 12);
    set(0, 'DefaultLineLineWidth', 1.5);

    % --- Figure 1: Baseline Comparison Bar Chart ---
    if exist(fullfile(results_dir, 'baseline_comparison.mat'), 'file')
        load(fullfile(results_dir, 'baseline_comparison.mat'), 'results', 'methods_list');

        rmse_values = [results.odometry.mean_rmse, ...
                       results.ekf_fixed_40.mean_rmse, ...
                       results.ekf_fixed_60.mean_rmse, ...
                       results.ekf_fixed_80.mean_rmse, ...
                       results.ekf_fixed_100.mean_rmse, ...
                       results.adaptive.mean_rmse];

        fig1 = figure('Name', 'Baseline Comparison', 'Position', [100 100 800 400]);
        b = bar(rmse_values);
        b.FaceColor = 'flat';
        b.CData(end, :) = [0.2 0.6 0.2];  % Green for proposed method
        set(gca, 'XTickLabel', methods_list, 'XTickLabelRotation', 30);
        ylabel('Mean RMSE (m)');
        title('Building Localization: Method Comparison');
        grid on;

        saveas(fig1, fullfile(results_dir, 'fig_baseline_comparison.png'));
        fprintf('  Saved: fig_baseline_comparison.png\n');
    end

    % --- Figure 2: Observability Validation ---
    if exist(fullfile(results_dir, 'observability_validation.mat'), 'file')
        load(fullfile(results_dir, 'observability_validation.mat'), 'results');
        ts = results.timestamps;

        fig2 = figure('Name', 'Observability Validation', 'Position', [100 100 800 600]);

        subplot(2,1,1);
        hold on;
        fill([ts, fliplr(ts)], ...
             [mean(results.circular.errors,1) + std(results.circular.errors,0,1), ...
              fliplr(mean(results.circular.errors,1) - std(results.circular.errors,0,1))], ...
             [0.7 0.7 1], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
        fill([ts, fliplr(ts)], ...
             [mean(results.straight.errors,1) + std(results.straight.errors,0,1), ...
              fliplr(mean(results.straight.errors,1) - std(results.straight.errors,0,1))], ...
             [1 0.7 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
        plot(ts, mean(results.circular.errors,1), 'b-', 'LineWidth', 2);
        plot(ts, mean(results.straight.errors,1), 'r-', 'LineWidth', 2);
        xlabel('Time (s)'); ylabel('Position Error (m)');
        title('EKF Convergence by Trajectory Type');
        legend('', '', 'Circular orbit', 'Straight line (degenerate)', 'Location', 'northeast');
        grid on; hold off;

        subplot(2,1,2);
        hold on;
        plot(ts, mean(results.circular.traces,1), 'b-', 'LineWidth', 2);
        plot(ts, mean(results.straight.traces,1), 'r-', 'LineWidth', 2);
        xlabel('Time (s)'); ylabel('trace(P)');
        title('Covariance Trace Over Time');
        legend('Circular orbit', 'Straight line', 'Location', 'northeast');
        set(gca, 'YScale', 'log');
        grid on; hold off;

        saveas(fig2, fullfile(results_dir, 'fig_observability_validation.png'));
        fprintf('  Saved: fig_observability_validation.png\n');
    end

    % --- Figure 3: Noise Sensitivity ---
    if exist(fullfile(results_dir, 'noise_sensitivity.mat'), 'file')
        load(fullfile(results_dir, 'noise_sensitivity.mat'), 'results');

        fig3 = figure('Name', 'Noise Sensitivity', 'Position', [100 100 600 400]);
        errorbar(results.sigma_values, results.mean_rmse, results.std_rmse, ...
                 'bo-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
        xlabel('Pixel Noise \sigma (px)');
        ylabel('Final RMSE (m)');
        title('Localization Accuracy vs. Measurement Noise');
        grid on;

        saveas(fig3, fullfile(results_dir, 'fig_noise_sensitivity.png'));
        fprintf('  Saved: fig_noise_sensitivity.png\n');
    end

    fprintf('Done.\n');
end
