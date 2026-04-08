function [h_cmd, h_des, J_bar] = altitude_controller(tracker, h_current, dt, params)
% ALTITUDE_CONTROLLER Uncertainty-driven altitude controller.
%
%   [h_cmd, h_des, J_bar] = altitude_controller(tracker, h_current, dt, params)
%
%   Computes desired altitude based on EKF covariance. When uncertainty is high,
%   descends to get better measurements. When uncertainty is low, ascends for coverage.
%
%   Control law:
%       h_des = h_min + (h_max - h_min) * exp(-alpha_norm * J_bar / J_ref)
%
%   Inputs:
%       tracker   - MultiBuildingTracker or scalar (mean covariance trace)
%       h_current - Current altitude (positive value, converted internally)
%       dt        - Time step in seconds
%       params    - struct with fields:
%           .h_min     - Minimum altitude in m (default: 20)
%           .h_max     - Maximum altitude in m (default: 120)
%           .alpha     - Normalized sensitivity parameter (default: 3.0)
%           .J_ref     - Reference trace for normalization (default: 900)
%           .tau       - Time constant for altitude response (default: 5 s)
%           .h_dot_max - Maximum vertical speed in m/s (default: 3)
%
%   Outputs:
%       h_cmd - Commanded altitude (rate-limited, clamped)
%       h_des - Raw desired altitude from control law (before rate limiting)
%       J_bar - Current mean covariance trace
%
%   See also: rate_limiter, MultiBuildingTracker

    if nargin < 4; params = struct(); end

    % Default parameters
    h_min     = get_field(params, 'h_min', 20);
    h_max     = get_field(params, 'h_max', 120);
    alpha     = get_field(params, 'alpha', 3.0);
    J_ref     = get_field(params, 'J_ref', 900);
    tau       = get_field(params, 'tau', 5);
    h_dot_max = get_field(params, 'h_dot_max', 3);

    % Get mean covariance trace
    if isa(tracker, 'MultiBuildingTracker')
        J_bar = tracker.mean_trace();
    elseif isa(tracker, 'BuildingEKF')
        J_bar = tracker.trace_P();
    else
        J_bar = tracker;  % Allow passing scalar directly
    end

    % Control law: exponential mapping
    J_norm = J_bar / J_ref;
    h_des = h_min + (h_max - h_min) * exp(-alpha * J_norm);

    % Rate limiting
    h_cmd = rate_limiter(h_des, h_current, dt, h_dot_max, tau, h_min, h_max);
end

function val = get_field(s, name, default)
    if isfield(s, name)
        val = s.(name);
    else
        val = default;
    end
end
