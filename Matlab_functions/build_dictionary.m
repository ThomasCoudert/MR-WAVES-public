%% Calculate dictionary of slice profiles for various flip angles

%% Load RF pulse information
rfvar = sg_300_100_0;
gamma = 4258;                   % gyoromagnetic ratio for protons, Hz/Gauss

%% Define dictionary properties
nominal_flip = 45;              % degrees - required to calculate isodelay
flip_angles = 1:1000;           % degrees - defines dictionary entries
n_dict = numel(flip_angles);    % size of dictionary
T1 = Inf;                       % do not consider relaxation (short pulse)
T2 = Inf;
slice_thickness = 4;            % mm
n_pos = 201;                    % number of spatial samples
extent = 4;                     % extent of sampling (multiple of slice thickness)
rf_dur = 0.0015;                 % RF pulse duration, seconds

% %% Set AFI acquisition parameters to verify AFI signals
% tr1 = 0.02;                     % seconds
% n   = 5;
% t1  = 0.5;                      % seconds
% tr2 = tr1*n;
% e1  = exp(-tr1/t1);
% e2  = exp(-tr2/t1);
% cfa = cos(flip_angles*pi/180);
% sfa = sin(flip_angles*pi/180);
% afi_theory(:,1) = sfa.*(1-e2+(1-e1)*e2*cfa)./(1-e1*e2*cfa.^2); 
% afi_theory(:,2) = sfa.*(1-e1+(1-e2)*e1*cfa)./(1-e1*e2*cfa.^2); 

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
% calculate reference position of the pulse (from the start) for refocusing
ref = rf_dur * (rfvar.sym - rfvar.ref_ex*(nominal_flip/90)^2); % seconds
refoc_dur = rf_dur - ref;       % time after the reference position
g_refoc = -g_s * refoc_dur / dt;% g_refoc*dt = -g_s*refoc_dur (seconds)
grad = [g_s*ones(rfvar.am_samples,1); g_refoc];

% calculate positions to space them symetrically. Center of slice: pos=0
x = slice_thickness * extent * ((1:n_pos)-(n_pos+1)/2)/n_pos;  % mm

% initialize dictionary and also a matrix with magnetizations for debugging
dict_fa = zeros(n_dict, n_pos);
dict_ph = zeros(n_dict, n_pos);
Mxyz = zeros(n_dict,n_pos,3);
afi_signals = zeros(n_dict, n_pos, 2);

t_start = tic;
for i_dict = 1:n_dict
    this_flip = flip_angles(i_dict);% degrees
    rf = rf_per_degree * this_flip; % Scale RF pulse in amplitude
    
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
t_end = toc(t_start);
fprintf('Duration of simulation: %f s\n', t_end);

%%
% average magnetization across simulation domain
% scale with extent, since only 1/extent is inside the slice
mean_Mxyz = squeeze(mean(Mxyz,2)) * extent;
% figure; 
% subplot(2,1,1);
% plot(flip_angles, mean_Mxyz(:,1), 'b'); 
% hold on; 
% plot(flip_angles, sin(flip_angles*pi/180),'r'); 
% ylabel('integral(M_x)')
% xlabel('Flip angle [deg]')
% legend('Simulated','sin(alpha)');
% title('Average magnetization across slice')
% grid on
% subplot(2,1,2);
% plot(flip_angles, squeeze(mean(Mxyz(:,:,3),2)), 'b'); 
% hold on; 
% plot(flip_angles, 1+(cos(flip_angles*pi/180)-1)/extent,'r'); 
% ylabel('integral(M_z)')
% xlabel('Flip angle [deg]')
% legend('Simulated','1 + (cos(alpha)-1)/extent');
% grid on

%%
% % average AFI signals across simulation domain
% % scale with extent, since only 1/extent is inside the slice
% afi_sim = squeeze(mean(afi_signals,2)) * extent;
% figure; 
% subplot(2,1,1);
% plot(flip_angles, real(afi_sim(:,1)), 'b'); 
% hold on; 
% plot(flip_angles, afi_theory(:,1),'r'); 
% plot(flip_angles, imag(afi_sim(:,1)), 'g');  % should be zero, plot to check
% ylabel('AFI S_1')
% xlabel('Flip angle [deg]')
% legend('Simulated','Theory');
% title('AFI signals')
% grid on
% subplot(2,1,2);
% plot(flip_angles, real(afi_sim(:,2)), 'b'); 
% hold on; 
% plot(flip_angles, afi_theory(:,2),'r'); 
% plot(flip_angles, imag(afi_sim(:,2)), 'g');  % should be zero, plot to check
% ylabel('AFI S_2')
% xlabel('Flip angle [deg]')
% grid on

%%
plot_flip_idx = 45;
figure;
subplot(2,1,1);
plot(x,Mxyz(plot_flip_idx,:,1),'b');
hold on;
plot(x,Mxyz(plot_flip_idx,:,2),'r');
plot(x,Mxyz(plot_flip_idx,:,3),'g');
grid on
legend('M_x','M_y','M_z')
xlabel('Position [mm]')
ylabel('Magnetization')
title(sprintf('Flip angle: %d°',flip_angles(plot_flip_idx)))
subplot(2,1,2);
plot(x, dict_fa(plot_flip_idx,:),'b');
hold on
plot(x, dict_ph(plot_flip_idx,:),'r');
legend('Flip angle','B_1 phase')
xlabel('Position [mm]')
ylabel('Flip angle / phase [deg]')
grid on


%%
% plot_flip_idx = 90;
% fh = figure;
% pos = get(fh, 'Position');
% pos(3:4) = [1120 840];
% set(fh, 'Position', pos);
% ah = subplot(2,1,1);
% plot(x,Mxyz(plot_flip_idx,:,1),'b', 'LineWidth', 2);
% hold on;
% plot(x,Mxyz(plot_flip_idx,:,2),'r', 'LineWidth', 2);
% plot(x,Mxyz(plot_flip_idx,:,3),'g', 'LineWidth', 2);
% grid on
% legend('M_x','M_y','M_z')
% xlabel('Position [mm]')
% ylabel('Magnetization')
% title(sprintf('Flip angle: %d°',flip_angles(plot_flip_idx)))
% set(ah, 'FontSize', 16)
% ah = subplot(2,1,2);
% plot(x, real(afi_signals(plot_flip_idx,:,1)),'b', 'LineWidth', 2);
% hold on
% plot(x, real(afi_signals(plot_flip_idx,:,2)),'r', 'LineWidth', 2);
% afi_1_ideal = zeros(size(x));
% afi_1_ideal(abs(x)<=1) = real(afi_signals(plot_flip_idx,x==0,1));
% afi_2_ideal = zeros(size(x));
% afi_2_ideal(abs(x)<=1) = real(afi_signals(plot_flip_idx,x==0,2));
% plot(x, afi_1_ideal,'b:', 'LineWidth', 2);
% plot(x, afi_2_ideal,'r:', 'LineWidth', 2);
% legend('real(AFI S1) - Bloch','real(AFI S2) - Bloch','real(AFI S1) - theor.','real(AFI S2) - theor.')
% xlabel('Position [mm]')
% ylabel('AFI signal intensity (a.u.)')
% grid on
% set(ah, 'FontSize', 16)
