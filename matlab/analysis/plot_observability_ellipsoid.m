function fig = plot_observability_ellipsoid(x_building, P, positions, fig_title)
% PLOT_OBSERVABILITY_ELLIPSOID Visualize the uncertainty ellipsoid and UAV trajectory.
%
%   fig = plot_observability_ellipsoid(x_building, P, positions, fig_title)
%
%   Inputs:
%       x_building - [3x1] Building position (or estimate)
%       P          - [3x3] Covariance matrix
%       positions  - [3xN] UAV trajectory positions (optional)
%       fig_title  - Figure title string (optional)
%
%   Outputs:
%       fig - Figure handle

    if nargin < 3; positions = []; end
    if nargin < 4; fig_title = 'Uncertainty Ellipsoid'; end

    fig = figure('Name', fig_title);
    hold on; grid on;

    % Generate ellipsoid surface (3-sigma)
    n_sigma = 3;
    [V, D] = eig(P);
    radii = n_sigma * sqrt(diag(D));

    [X, Y, Z] = ellipsoid(0, 0, 0, radii(1), radii(2), radii(3), 30);

    % Rotate ellipsoid
    for i = 1:numel(X)
        p = V * [X(i); Y(i); Z(i)] + x_building;
        X(i) = p(1); Y(i) = p(2); Z(i) = p(3);
    end

    surf(X, Y, Z, 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'FaceColor', [0.8 0.2 0.2]);

    % Plot building position
    plot3(x_building(1), x_building(2), x_building(3), 'r*', 'MarkerSize', 15, 'LineWidth', 2);

    % Plot UAV trajectory if provided
    if ~isempty(positions)
        plot3(positions(1,:), positions(2,:), positions(3,:), 'b-', 'LineWidth', 1.5);
        plot3(positions(1,1), positions(2,1), positions(3,1), 'go', 'MarkerSize', 10, 'LineWidth', 2);
        plot3(positions(1,end), positions(2,end), positions(3,end), 'rs', 'MarkerSize', 10, 'LineWidth', 2);
    end

    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title(fig_title);
    legend('3\sigma ellipsoid', 'Building', 'UAV trajectory', 'Start', 'End', ...
           'Location', 'best');
    axis equal;
    view(3);
    hold off;
end
