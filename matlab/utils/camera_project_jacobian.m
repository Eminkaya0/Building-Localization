function H = camera_project_jacobian(x_world, uav_pos, uav_R_WB, K, R_BC, t_BC)
% CAMERA_PROJECT_JACOBIAN Compute the measurement Jacobian H = dh/dx.
%
%   H = camera_project_jacobian(x_world, uav_pos, uav_R_WB, K, R_BC, t_BC)
%
%   Computes the 2x3 Jacobian of the pinhole projection with respect to
%   the 3D building position in world frame. Uses the chain rule:
%       H = J_pi * R_CW
%   where J_pi is the projection Jacobian and R_CW is the world-to-camera rotation.
%
%   Inputs:
%       x_world  - [3x1] Building position in world frame
%       uav_pos  - [3x1] UAV position in world frame
%       uav_R_WB - [3x3] Rotation from body to world frame
%       K        - [3x3] Camera intrinsic matrix
%       R_BC     - [3x3] Body-to-camera rotation
%       t_BC     - [3x1] Body-to-camera translation
%
%   Outputs:
%       H - [2x3] Measurement Jacobian dh/dx
%
%   See also: camera_project, world_to_camera

    % Get camera frame coordinates and rotation
    [p_C, R_CW, ~] = world_to_camera(x_world, uav_pos, uav_R_WB, R_BC, t_BC);

    X_C = p_C(1);
    Y_C = p_C(2);
    Z_C = p_C(3);

    fx = K(1,1);
    fy = K(2,2);

    % Projection Jacobian J_pi (2x3)
    % dh/dp_C = (1/Z_C) * [fx  0  -fx*X_C/Z_C;
    %                        0  fy  -fy*Y_C/Z_C]
    J_pi = (1 / Z_C) * [fx,  0, -fx * X_C / Z_C;
                          0, fy, -fy * Y_C / Z_C];

    % Full Jacobian: H = J_pi * R_CW
    H = J_pi * R_CW;
end
