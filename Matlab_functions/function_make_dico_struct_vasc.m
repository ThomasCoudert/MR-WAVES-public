function [] = function_make_dico_struct_vasc(h5path, dico_path)

Parameters = h5read(h5path, '/Parameters');
Sequence = h5read(h5path, '/Sequence');
MRSignals = h5read(h5path, '/MRSignals');
VoxelList = h5read(h5path, '/VoxelList');

% Mag = ((MRSignals.r).^2 + (MRSignals.i).^2).^0.5;
Mag = MRSignals.r + 1i*MRSignals.i ;

%%
% clear Dico
Dico1.MRSignals = {squeeze(Mag)'};
Dico1.Parameters.Labels = {'T1', 'T2', 'df', 'B1rel', 'SO2', 'Vf', 'R'};
Dico1.Parameters.Par = Parameters';
Dico1.Parameters.Par(:, 4) = []; % remove gamma column
Dico1.VoxelList = cellstr(VoxelList);

%sequence parameters: 
Dico1.Tacq = Sequence(2,:);
Dico1.Sequence.TR_train = Sequence(3,:);
Dico1.Sequence.FA_train = Sequence(1,:);
Dico1.Sequence.phi = Sequence(4,:);
Dico1.path = dico_path;
Dico = Dico1;
save(Dico1.path, 'Dico', '-v7.3' )

end
