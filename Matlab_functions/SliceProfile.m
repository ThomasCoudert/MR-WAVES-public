function [profile] = SliceProfile(FAtrain, sliceThickness, samplingFactor)

%% Parameters
% slice thickness in mm
% FA in degrees
% samplingFactor : define the number of point in the RF profile (different sampling)

%% Get slice profile
fullProfile = getSliceProfile(FAtrain, sliceThickness);
profile = fullProfile(:, 1:samplingFactor:end);

end
