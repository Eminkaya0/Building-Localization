function [p_C, R_CW, t_CW] = world_to_camera(x_world, uav_pos, uav_R_WB, R_BC, t_BC)
% WORLD_TO_CAMERA Transform a 3D world point to camera frame coordinates.
%
%   [p_C, R_CW, t_CW] = world_to_camera(x_world, uav_pos, uav_R_WB, R_BC, t_BC)
%
%   Inputs:
%       x_world  - [3x1] Building position in world frame
%       uav_pos  - [3x1] UAV position in world frame
%       uav_R_WB - [3x3] Rotation from body to world frame
%       R_BC     - [3x3] Rotation from body to camera frame (extrinsic calibration)
%       t_BC     - [3x1] Translation from body to camera frame (extrinsic calibration)
%
%   Outputs:
%       p_C  - [3x1] Point in camera frame [X_C; Y_C; Z_C]
%       R_CW - [3x3] Rotation from world to camera frame
%       t_CW - [3x1] Translation component of world-to-camera transform
%
%   The camera frame convention: Z-axis along optical axis (downward for nadir).
%
%   See also: camera_project, camera_project_jacobian

    % World-to-camera rotation: R_CW = (R_WC)^T where R_WC = R_BC * R_WB
    R_WC = R_BC * uav_R_WB;
    R_CW = R_WC';

    % World-to-camera translation
    t_WC = R_BC * uav_pos + t_BC;
    t_CW = -R_CW * t_WC;

    % Transform world point to camera frame
    p_C = R_CW * x_world + t_CW;
end
