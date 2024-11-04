function [] = compress_svd(Parameters, Sequence, MRSignals, dico_path)
%% Compress MRF dictionary before saving them in a .mat struct
% Also saving Vk and Vkcpx, which are the right singular vector of the SVD
% decomposition, allowing retrieval of complex and module signal in the
% origin space.
% Msvd = abs(Dico.MRSignals{1,1}*Dico.Vk')*abs(Dico.Vk) ||| for module in SVD space
% Zorigin = Dico.MRSignals{1,1} * Dico.Vk';
% Morigin = abs(Dico.MRSignals{1,1} * Dico.Vk');

% Method from [1] D. F. McGivney et al., “SVD Compression for Magnetic Resonance Fingerprinting in the Time Domain,” IEEE Trans. Med. Imaging, vol. 33, no. 12, pp. 2311–2322, Dec. 2014, doi: 10.1109/TMI.2014.2337321.

%% SVD complex
MR_signals = MRSignals;

[~, ~, V] = svd(MR_signals, 'econ');
% V contains the right singular vectors (columns)
% Reduce the rank to a desired value
desired_rank = 10;
V_reduced = V(:, 1:desired_rank);

MR_signals_SVD = MR_signals * V_reduced;
Dico1.MRSignals{1,1} = MR_signals_SVD;
Dico1.Vkcpx = V_reduced;

%% SVD module
MR_signals = abs(MRSignals);

[~, ~, Vabs] = svd(MR_signals, 'econ');
% V contains the right singular vectors (columns)
% Reduce the rank to a desired value
desired_rank = 10;
V_reducedabs = Vabs(:, 1:desired_rank);

% MR_signals_SVD = MR_signals * V_reducedabs;
% Dico1.MRSignals{1,1} = MR_signals_SVD;
Dico1.Vk = V_reducedabs;

%%
Dico1.Parameters.Labels = {'T1', 'T2', 'df', 'gamma', 'B1rel', 'Ttwo_star'};
T2_values = 1./((1./Parameters(:, 2)) + pi.*Parameters(:, 4));
Dico1.Parameters.Par = cat(2, Parameters, T2_values);

%sequence parameters: 
Sequence = Sequence';
Dico1.Tacq = Sequence(2,:);
Dico1.Sequence.TR_train = Sequence(3,:);
Dico1.Sequence.FA_train = Sequence(1,:);
Dico1.Sequence.phi = Sequence(4,:);
Dico1.path = dico_path;
Dico = Dico1;
save(Dico1.path, 'Dico', '-v7.3' )

end