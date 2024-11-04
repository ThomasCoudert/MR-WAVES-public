% matlab function to call in python with :

% import matlab.engine
% eng = matlab.engine.start_matlab()
% eng.call_in_python()

%%
function dictionary=BlochOverlay_python(nP, FA ,Phi ,TR ,TE, T1list, T2list, B1list, dflist, ...
    spoil, invPulse, nMultiEchoes)

% Overlay function for Bloch computation, using functions written by B. Hargreaves
% This function uses a parfor loop from Matlab Parallel Computing Toolbox

% This is the call-in python adapted function of BlochOverlay
% -------------------------------------------------------------------------
% - nP : nb pulses
% - FA : rad
% - Phi : rad
% - TR : s
% - TE : s
% - T1list : s
% - T2list : s
% - df : Hz
% - gradAmp: spoiling gradient amplitude (T/m)
% - gradDur: spoiling gradient duration (s)
% - fov = field of view (m), used to compute isochromat positions
%
% - dictionary = matrix nSignals * nPulses, complex double
% - tF = computation time measured by toc function
% -------------------------------------------------------------------------
switch spoil
    case 1
        spoilType = 'FID';
    case 0
        spoilType = 'bSSFP';
    otherwise
        error('FID or bSSFP type');
end

fov = 256e-6; % m

% WIP - gradients
gradDur = 0.01 *  ones(1, nP);% s
gradAmp = 0.3 * ones(1,nP); % T/m
Shim = 0;
dictionary = zeros(numel(T1list), nP);

if nMultiEchoes > 1
	idxOn = zeros(1, nP);
	switch spoilType
		case 'FID'
	        	idxOn(nMultiEchoes:nMultiEchoes:end) = 1;
	        	gradAmp = gradAmp .* idxOn;
		case 'Echo'
	        	idxOn(1:nMultiEchoes:end) = 1;
			gradAmp = gradAmp .* idxOn;
	end
end

% parfor if nSignals > nWorkers
% ps = parallel.Settings;
if numel(T1list) > 6 %ps.SchedulerComponents.NumWorkers
    if isempty(gcp)
        delete(gcp('nocreate'))
        parpool;
    end
    t = tic;
%     parfor_progress(numel(T1list));
    parfor i = 1:numel(T1list)
        [dictionary(i,:)] = SimuBloch(B1list(i)*FA, Phi, TR, TE, T1list(i), T2list(i), dflist(i), gradDur, gradAmp, Shim, spoilType, invPulse, fov);
%         parfor_progress;
    end
%     parfor_progress(0);
    tF = round(toc(t)); %#ok<*NASGU>
%    fprintf('\n'); fprintf('\n');
else
    % Serial for otherwise
    t = tic;
    for i = 1:numel(T1list)
        [dictionary(i,:)] = SimuBloch(B1list(i)*FA, Phi, TR, TE, T1list(i), T2list(i), dflist(i), gradDur, gradAmp, Shim, spoilType, invPulse, fov);
    end
    tF = round(toc(t));
end

end
