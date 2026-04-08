function h_cmd = rate_limiter(h_des, h_current, dt, h_dot_max, tau, h_min, h_max)
% RATE_LIMITER Apply rate limiting and altitude bounds to a desired altitude.
%
%   h_cmd = rate_limiter(h_des, h_current, dt, h_dot_max, tau, h_min, h_max)
%
%   Inputs:
%       h_des     - Desired altitude from control law
%       h_current - Current altitude
%       dt        - Time step in seconds
%       h_dot_max - Maximum vertical speed in m/s (default: 3)
%       tau       - Time constant in seconds (default: 5)
%       h_min     - Minimum altitude (default: 20)
%       h_max     - Maximum altitude (default: 120)
%
%   Outputs:
%       h_cmd - Rate-limited and clamped commanded altitude

    if nargin < 4; h_dot_max = 3; end
    if nargin < 5; tau = 5; end
    if nargin < 6; h_min = 20; end
    if nargin < 7; h_max = 120; end

    % First-order response with rate limiting
    h_dot = (h_des - h_current) / tau;

    % Clamp rate
    h_dot = max(-h_dot_max, min(h_dot_max, h_dot));

    % Apply rate
    h_cmd = h_current + h_dot * dt;

    % Enforce altitude bounds
    h_cmd = max(h_min, min(h_max, h_cmd));
end
