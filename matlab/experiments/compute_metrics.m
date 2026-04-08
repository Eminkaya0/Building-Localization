function metrics = compute_metrics(errors_over_time, timestamps)
% COMPUTE_METRICS Compute localization performance metrics.
%
%   metrics = compute_metrics(errors_over_time, timestamps)
%
%   Inputs:
%       errors_over_time - [MxN] Euclidean errors for M buildings over N timesteps
%       timestamps       - [1xN] Time stamps
%
%   Outputs:
%       metrics - struct with fields:
%           .rmse_per_building   - [Mx1] RMSE per building
%           .mean_rmse           - Scalar mean RMSE across buildings
%           .final_errors        - [Mx1] Final error per building
%           .cep50               - Circular Error Probable (50th percentile of final errors)
%           .cep90               - 90th percentile of final errors
%           .convergence_time    - [Mx1] Time to reach error < threshold (per building)
%           .convergence_thresh  - Threshold used for convergence

    convergence_thresh = 5.0;  % meters

    [M, N] = size(errors_over_time);

    % RMSE per building (over time)
    rmse_per_building = sqrt(mean(errors_over_time.^2, 2));

    % Mean RMSE
    mean_rmse = mean(rmse_per_building);

    % Final errors
    final_errors = errors_over_time(:, end);

    % CEP (Circular Error Probable)
    cep50 = prctile(final_errors, 50);
    cep90 = prctile(final_errors, 90);

    % Convergence time per building
    convergence_time = NaN(M, 1);
    for i = 1:M
        idx = find(errors_over_time(i, :) < convergence_thresh, 1, 'first');
        if ~isempty(idx)
            convergence_time(i) = timestamps(idx);
        end
    end

    metrics.rmse_per_building  = rmse_per_building;
    metrics.mean_rmse          = mean_rmse;
    metrics.final_errors       = final_errors;
    metrics.cep50              = cep50;
    metrics.cep90              = cep90;
    metrics.convergence_time   = convergence_time;
    metrics.convergence_thresh = convergence_thresh;
end
