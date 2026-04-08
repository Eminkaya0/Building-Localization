function measurements = generate_synthetic_measurements(buildings, positions, rotations, timestamps, K, R_BC, t_BC, sigma_pixel, img_size)
% GENERATE_SYNTHETIC_MEASUREMENTS Generate noisy pixel measurements from a UAV trajectory.
%
%   measurements = generate_synthetic_measurements(buildings, positions, rotations, ...
%                       timestamps, K, R_BC, t_BC, sigma_pixel, img_size)
%
%   Inputs:
%       buildings    - [3xM] Ground truth building positions in world frame
%       positions    - [3xN] UAV positions along trajectory
%       rotations    - {1xN} Cell array of body-to-world rotation matrices
%       timestamps   - [1xN] Time stamps
%       K            - [3x3] Camera intrinsic matrix
%       R_BC         - [3x3] Body-to-camera rotation
%       t_BC         - [3x1] Body-to-camera translation
%       sigma_pixel  - Pixel noise standard deviation (default: 2)
%       img_size     - [2x1] Image size [width; height] (optional, for FOV check)
%
%   Outputs:
%       measurements - struct array with fields:
%           .time      - Timestamp
%           .uav_pos   - [3x1] UAV position
%           .uav_R     - [3x3] UAV rotation (body-to-world)
%           .detections - struct array per visible building:
%               .building_id  - Index into buildings array
%               .z_true       - [2x1] True pixel coordinates
%               .z_noisy      - [2x1] Noisy pixel coordinates
%               .is_valid     - Whether detection is within FOV

    if nargin < 8 || isempty(sigma_pixel)
        sigma_pixel = 2;
    end
    if nargin < 9
        img_size = [];
    end

    N = length(timestamps);
    M = size(buildings, 2);

    measurements = struct('time', cell(1, N), 'uav_pos', cell(1, N), ...
                          'uav_R', cell(1, N), 'detections', cell(1, N));

    for i = 1:N
        measurements(i).time    = timestamps(i);
        measurements(i).uav_pos = positions(:, i);
        measurements(i).uav_R   = rotations{i};

        dets = [];
        for j = 1:M
            [z_true, is_valid] = camera_project(buildings(:, j), positions(:, i), ...
                                                 rotations{i}, K, R_BC, t_BC, img_size);

            if is_valid
                noise = sigma_pixel * randn(2, 1);
                z_noisy = z_true + noise;

                det.building_id = j;
                det.z_true      = z_true;
                det.z_noisy     = z_noisy;
                det.is_valid    = true;
                dets = [dets, det]; %#ok<AGROW>
            end
        end

        measurements(i).detections = dets;
    end
end
