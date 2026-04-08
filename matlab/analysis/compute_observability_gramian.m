function [O, eigenvalues, eigenvectors] = compute_observability_gramian(x_building, positions, rotations, K, R_BC, t_BC, R_meas)
% COMPUTE_OBSERVABILITY_GRAMIAN Compute the observability Gramian for a static building.
%
%   [O, eigenvalues, eigenvectors] = compute_observability_gramian(x_building, ...
%       positions, rotations, K, R_BC, t_BC, R_meas)
%
%   The observability Gramian is: O = sum_i H_i' * R^{-1} * H_i
%   The system is locally observable iff rank(O) = 3.
%
%   Inputs:
%       x_building - [3x1] Building position in world frame
%       positions  - [3xN] UAV positions along trajectory
%       rotations  - {1xN} Cell array of body-to-world rotations
%       K          - [3x3] Camera intrinsic matrix
%       R_BC       - [3x3] Body-to-camera rotation
%       t_BC       - [3x1] Body-to-camera translation
%       R_meas     - [2x2] Measurement noise covariance
%
%   Outputs:
%       O            - [3x3] Observability Gramian
%       eigenvalues  - [3x1] Eigenvalues (sorted descending)
%       eigenvectors - [3x3] Corresponding eigenvectors (columns)
%
%   See also: check_observability, camera_project_jacobian

    N = size(positions, 2);
    R_inv = inv(R_meas);
    O = zeros(3, 3);

    for i = 1:N
        % Check if building is visible from this position
        [~, is_valid] = camera_project(x_building, positions(:, i), ...
                                        rotations{i}, K, R_BC, t_BC);
        if ~is_valid
            continue;
        end

        H_i = camera_project_jacobian(x_building, positions(:, i), ...
                                       rotations{i}, K, R_BC, t_BC);
        O = O + H_i' * R_inv * H_i;
    end

    % Eigendecomposition
    [V, D] = eig(O);
    eigenvalues = diag(D);

    % Sort descending
    [eigenvalues, idx] = sort(eigenvalues, 'descend');
    eigenvectors = V(:, idx);
end
