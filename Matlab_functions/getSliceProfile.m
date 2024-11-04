function [profile] = getSliceProfile(FAtrain, slice_thickness)

if nargin < 2
    slice_thickness = 3;
end

%% Calculate dictionary of slice profiles for various flip angles

%% Load RF pulse information
rfvar = sg_300_100_0;
gamma = 4258;                   % gyoromagnetic ratio for protons, Hz/Gauss

%% Define dictionary properties
%flip_angles = unique(FAtrain);           % degrees - defines dictionary entries
flip_angles = FAtrain;
n_dict = numel(flip_angles);    % size of dictionary
T1 = Inf;                       % do not consider relaxation (short pulse)
T2 = Inf;
n_pos = 201;                    % number of spatial samples
extent = 4;                     % extent of sampling (multiple of slice thickness)
rf_dur = 0.0015;                 % RF pulse duration, seconds

%% Calculate RF and gradient envelopes with correct scale
% We will have numel(am_shape) RF and gradient samples for the RF pulse
% with slice select gradient, followed by a single sample where rf=0 with
% the refocusing gradient (we are neglecting relaxation).
am_shape = rfvar.am_shape;
am_shape = am_shape / (mean(am_shape) * rf_dur); % Normalize to integral=1
% scale envelope to produce a nominal 1° flip angle
rf_per_degree = am_shape * (1/360) / gamma;  % Gauss
rf_per_degree = [rf_per_degree; 0]; % add null sample at the end for refoc

dt = rf_dur / rfvar.am_samples; % duration of each RF/grad sample (seconds)

% calculate amplitude of slice select gradient
g_s = rfvar.bw_ex / (rf_dur*gamma*slice_thickness); % Gauss/mm
g_s = 10 * g_s;                 % Gauss/cm

% calculate positions to space them symetrically. Center of slice: pos=0
x = slice_thickness * extent * ((1:n_pos)-(n_pos+1)/2)/n_pos;  % mm

% initialize dictionary and also a matrix with magnetizations for debugging
dict_fa = zeros(n_dict, n_pos);
dict_ph = zeros(n_dict, n_pos);
Mxyz = zeros(n_dict,n_pos,3);

for i_dict = 1:n_dict
    this_flip = flip_angles(i_dict);% degrees
    rf = rf_per_degree * this_flip; % Scale RF pulse in amplitude
    
    % calculate reference position of the pulse (from the start) for refocusing
    ref = rf_dur * (rfvar.sym - rfvar.ref_ex*(this_flip/90)^2); % seconds
    refoc_dur = rf_dur - ref;       % time after the reference position
    g_refoc = -g_s * refoc_dur / dt;% g_refoc*dt = -g_s*refoc_dur (seconds)
    grad = [g_s*ones(rfvar.am_samples,1); g_refoc];
    
    % Apply RF pulse along the y-axis (imaginary B1). In a right-handed
    % coordinate system this produces real-valued transverse magnetization.
    [Mx,My,Mz] = sliceprofile(1i*rf,grad,dt,T1,T2,x);
    Mxyz(i_dict,:,:) = cat(3, Mx', My', Mz');
    
    % calculate actual flip angles from the resulting magnetization
    fa = atan2(sqrt(Mx.^2+My.^2), Mz)'; % rad
    % calculate actual RF phases from the resulting magnetization
    ph = atan2(My, Mx)';        % rad
    
    %     % calculate complex afi signals across the slice
    %     afi_signals(i_dict,:,1) = exp(1i*ph).*sin(fa).*(1-e2+(1-e1)*e2*cos(fa))./(1-e1*e2*cos(fa).^2);
    %     afi_signals(i_dict,:,2) = exp(1i*ph).*sin(fa).*(1-e1+(1-e2)*e1*cos(fa))./(1-e1*e2*cos(fa).^2);
    
    dict_fa(i_dict,:) = fa * 180/pi;% degrees
    dict_ph(i_dict,:) = ph * 180/pi;% degrees
end


profile = dict_fa;
end
