function [positions, rotations, timestamps] = generate_uav_trajectory(type, params)
% GENERATE_UAV_TRAJECTORY Generate UAV trajectory waypoints.
%
%   [positions, rotations, timestamps] = generate_uav_trajectory(type, params)
%
%   Inputs:
%       type   - string: 'circular', 'lawnmower', 'straight', 'hover'
%       params - struct with trajectory parameters:
%           .center    - [3x1] Center point of trajectory (default: [0;0;-80])
%           .radius    - Radius for circular trajectory (default: 50 m)
%           .altitude  - Flight altitude (negative Z in NED) (default: -80)
%           .speed     - UAV speed in m/s (default: 5)
%           .duration  - Total flight time in seconds (default: 120)
%           .dt        - Time step in seconds (default: 0.1)
%           .width     - Lawnmower pattern width (default: 100 m)
%           .height    - Lawnmower pattern height (default: 100 m)
%           .spacing   - Lawnmower line spacing (default: 20 m)
%           .direction - [3x1] Direction for straight-line trajectory
%           .start_pos - [3x1] Starting position for straight-line
%
%   Outputs:
%       positions  - [3xN] UAV positions in world frame (NED)
%       rotations  - {1xN} Cell array of 3x3 rotation matrices (body-to-world)
%       timestamps - [1xN] Time stamps in seconds

    if nargin < 2; params = struct(); end

    % Default parameters
    altitude = get_param(params, 'altitude', -80);
    speed    = get_param(params, 'speed', 5);
    duration = get_param(params, 'duration', 120);
    dt       = get_param(params, 'dt', 0.1);

    timestamps = 0:dt:duration;
    N = length(timestamps);

    positions = zeros(3, N);
    rotations = cell(1, N);

    switch lower(type)
        case 'circular'
            center = get_param(params, 'center', [0; 0; altitude]);
            radius = get_param(params, 'radius', 50);
            omega  = speed / radius;  % angular velocity

            for i = 1:N
                t = timestamps(i);
                theta = omega * t;
                positions(:, i) = center + [radius * cos(theta);
                                             radius * sin(theta);
                                             0];
                % Body X-axis tangent to circle (heading along trajectory)
                heading = theta + pi/2;
                rotations{i} = eul2rotm_zyx(heading, 0, 0);
            end

        case 'lawnmower'
            w       = get_param(params, 'width', 100);
            h       = get_param(params, 'height', 100);
            spacing = get_param(params, 'spacing', 20);
            start   = get_param(params, 'start_pos', [-w/2; -h/2; altitude]);

            % Generate lawnmower waypoints
            n_lines = floor(h / spacing) + 1;
            waypoints = [];
            for j = 0:n_lines-1
                y = start(2) + j * spacing;
                if mod(j, 2) == 0
                    waypoints = [waypoints, [start(1); y; altitude], [start(1) + w; y; altitude]]; %#ok<AGROW>
                else
                    waypoints = [waypoints, [start(1) + w; y; altitude], [start(1); y; altitude]]; %#ok<AGROW>
                end
            end

            % Interpolate along waypoints at constant speed
            [positions, rotations] = interpolate_waypoints(waypoints, speed, timestamps);

        case 'straight'
            start_pos = get_param(params, 'start_pos', [-100; 0; altitude]);
            direction = get_param(params, 'direction', [1; 0; 0]);
            direction = direction / norm(direction);

            for i = 1:N
                t = timestamps(i);
                positions(:, i) = start_pos + speed * t * direction;
                heading = atan2(direction(2), direction(1));
                rotations{i} = eul2rotm_zyx(heading, 0, 0);
            end

        case 'hover'
            hover_pos = get_param(params, 'center', [0; 0; altitude]);
            for i = 1:N
                positions(:, i) = hover_pos;
                rotations{i} = eye(3);
            end

        otherwise
            error('Unknown trajectory type: %s', type);
    end
end

%% Helper functions

function val = get_param(params, name, default)
    if isfield(params, name)
        val = params.(name);
    else
        val = default;
    end
end

function R = eul2rotm_zyx(yaw, pitch, roll)
% Simple ZYX Euler angle to rotation matrix (body-to-world)
    cy = cos(yaw);   sy = sin(yaw);
    cp = cos(pitch); sp = sin(pitch);
    cr = cos(roll);  sr = sin(roll);

    R = [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr;
         sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr;
         -sp,   cp*sr,            cp*cr];
end

function [positions, rotations] = interpolate_waypoints(waypoints, speed, timestamps)
% Interpolate position along piecewise-linear waypoint path at constant speed
    N = length(timestamps);
    positions = zeros(3, N);
    rotations = cell(1, N);

    % Compute cumulative distances along waypoints
    n_wp = size(waypoints, 2);
    cum_dist = zeros(1, n_wp);
    for j = 2:n_wp
        cum_dist(j) = cum_dist(j-1) + norm(waypoints(:,j) - waypoints(:,j-1));
    end
    total_dist = cum_dist(end);

    for i = 1:N
        d = speed * timestamps(i);
        d = min(d, total_dist);  % clamp to path length

        % Find segment
        seg = find(cum_dist >= d, 1, 'first');
        if seg <= 1
            seg = 2;
        end

        % Interpolate within segment
        seg_start = cum_dist(seg - 1);
        seg_len   = cum_dist(seg) - seg_start;
        if seg_len > 0
            alpha = (d - seg_start) / seg_len;
        else
            alpha = 0;
        end

        positions(:, i) = waypoints(:, seg-1) + alpha * (waypoints(:, seg) - waypoints(:, seg-1));

        % Heading along segment direction
        dir = waypoints(:, seg) - waypoints(:, seg-1);
        heading = atan2(dir(2), dir(1));
        rotations{i} = eul2rotm_zyx(heading, 0, 0);
    end
end
