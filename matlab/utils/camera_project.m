function [z_pixel, is_valid] = camera_project(x_world, uav_pos, uav_R_WB, K, R_BC, t_BC, img_size)
% CAMERA_PROJECT Project a 3D world point to pixel coordinates via pinhole model.
%
%   [z_pixel, is_valid] = camera_project(x_world, uav_pos, uav_R_WB, K, R_BC, t_BC, img_size)
%
%   Inputs:
%       x_world  - [3x1] Building position in world frame
%       uav_pos  - [3x1] UAV position in world frame
%       uav_R_WB - [3x3] Rotation from body to world frame
%       K        - [3x3] Camera intrinsic matrix [fx 0 cx; 0 fy cy; 0 0 1]
%       R_BC     - [3x3] Body-to-camera rotation (extrinsic calibration)
%       t_BC     - [3x1] Body-to-camera translation (extrinsic calibration)
%       img_size - [2x1] Image size [width; height] in pixels (optional)
%
%   Outputs:
%       z_pixel  - [2x1] Pixel coordinates [u; v]
%       is_valid - logical, true if point is in front of camera (and in FOV if img_size given)
%
%   See also: world_to_camera, camera_project_jacobian

    % Transform to camera frame
    p_C = world_to_camera(x_world, uav_pos, uav_R_WB, R_BC, t_BC);

    X_C = p_C(1);
    Y_C = p_C(2);
    Z_C = p_C(3);

    % Check if point is in front of camera
    if Z_C <= 0
        z_pixel = [NaN; NaN];
        is_valid = false;
        return;
    end

    % Pinhole projection
    fx = K(1,1);
    fy = K(2,2);
    cx = K(1,3);
    cy = K(2,3);

    u = fx * (X_C / Z_C) + cx;
    v = fy * (Y_C / Z_C) + cy;

    z_pixel = [u; v];

    % Check FOV if image size is provided
    is_valid = true;
    if nargin >= 7 && ~isempty(img_size)
        if u < 0 || u >= img_size(1) || v < 0 || v >= img_size(2)
            is_valid = false;
        end
    end
end
