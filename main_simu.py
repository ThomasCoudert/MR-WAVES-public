from function_utils import *
import mat73
import json
import os
import argparse
import shutil
import termplotlib as tpl


eng = matlab.engine.start_matlab()
s = eng.genpath('./Matlab_functions')
eng.addpath(s, nargout=0)

if not os.path.ismount('/data_network/summer_projects/couderth'):
    raise ValueError('Summer not mounted')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for MRF simulations")

    parser.add_argument("-json", "--json_path", type=str,
                        help="Path_to_config_json_file")

    parser.add_argument("-input_seq", "--input_seq", type=str,
                        help="Path_to_input_seq_param_json")

    args = parser.parse_args()

################### CONFIG ###################
simus_infos = json.load(open(args.json_path))
print(simus_infos)
input_seq = simus_infos['input_seq']

if input_seq:
    print("Loading external json sequence parameters file")
    seqparam = json.load(open(simus_infos['input_seq_path']))
    n_echoes = seqparam["n_echoes"]
    n_pulses = seqparam["n_pulses"]
    N = seqparam["N"]
    FA = seqparam["FA"]
    FA_train = np.array(seqparam["FA_train"])
    TR_train = np.array(seqparam["TR_train"])* 1e3
    TE_train = np.array(seqparam["TE_train"]) * 1e3
    TReff = seqparam["TReff"]
    phase = seqparam["phase"]
    phase_incr = seqparam["phase_incr"]
    if "phi_train" in seqparam:
        phi_train = np.array(seqparam["phi_train"])
    spoil = seqparam["spoil"]
    inv_pulse = seqparam["invPulse"]
else:
    n_echoes = simus_infos['n_echoes']
    n_pulses = simus_infos['n_pulses']
    FA_train_noME = eval(simus_infos['FA_train'])
    TRtrain = simus_infos['TRtrain']  # ms
    TEtrain = simus_infos['TEtrain']  # ms
    phase = simus_infos['phase']
    phase_incr = simus_infos['phase_incr']  # in degrees
    inv_pulse = simus_infos['inv_pulse']  # 0 or 1
    spoil = simus_infos['spoil']

inv_time = simus_infos['inv_time']  # ms
variableFA = simus_infos['variableFA']
name_seq = simus_infos['name_seq']
T1range = eval(simus_infos['T1range']) # ms
T2range = eval(simus_infos['T2range']) # ms
B1range = eval(simus_infos['B1range'])
SVD = simus_infos['SVD']
rank = simus_infos['SVD_rank']
commentaries = simus_infos['commentaries']
output_path = simus_infos['output_path']
gamma_list = simus_infos['gammalist']
param = simus_infos['param']
vasc = simus_infos['vasc']
grid = simus_infos['grid']
slice_profile = simus_infos['slice_profile']
distrib = simus_infos['distrib']
folder = simus_infos['from_folder']
distrib_path = simus_infos['distrib_path']
which_distrib = simus_infos['which_distrib']
bloch_based_dico = simus_infos['bloch_based_dico']


############################# SEQUENCE PARAMETERS #############################
N = n_echoes * n_pulses

#jiang = sio.loadmat('seqJiang2015.mat')


# FA = 45  # deg
# FA_train_noME = np.ones((1, n_pulses)) * FA

# FA_train_noME = np.ones((1, n_pulses)) * 20
# FA_train_ramp = np.linspace(20, 70, 35).reshape((1, 35))
# FA_train_noME[:, 35:70] = FA_train_ramp

# optim FA train
# FA_points = [53., 21., 4., 46., 26., 9., 27., 15., 28., 57., 48., 58., 34.,
#              58., 51.]
# FA_train_noME = Interpol_FA_trains_for_opti(FA_points, len(FA_points))
# GOMEZ (260 pulses)
# FA_train_noME = jiang['Sequence'][0][0][-1][0:N, 0]*(180/pi)

# # alternating FAs
# FA1 = 60  # deg
# FA2 = 30
# FA_train_noME = np.ones((1, n_pulses)) * FA1
# FA_train_noME[:, ::2] = FA2

if not input_seq:
    FA_train = np.zeros((1, N))
    FA_train[:, ::n_echoes] = FA_train_noME
    TR_train = eval(TRtrain)  # ms
    TE_train = eval(TEtrain)  # ms

# echo_times : [4,10,16]ms et TR=21ms
# TE_train = np.array([np.tile([4, 4, 4], n_pulses)])  # ms
# TR_train = np.array([np.tile([6, 6, 9], n_pulses)])  # ms



# TR_train = np.zeros((1, N))
# TE_train = np.zeros((1, N))
# TR_train[0, :] = jiang['Sequence'][0][0][0][0:N, 0] #ms
# TE_train[0, :] = jiang['Sequence'][0][0][1][0:N, 0] #ms

# TE_train = np.array([[4 + i % 3 for i in range(N)]])
# import pandas as pd
# csv_file_path = 'CSV_sequences/20231005_bssfpTE49.csv'
# column_name = 'Column_2'  # Change this to the name of the desired column
# df = pd.read_csv(csv_file_path, header=None, names=[f"Column_{i}" for i in range(7)])
# column_values = df[column_name].values
# TEs = [float(x) for x in column_values[12:]]
# TE_train= np.array(TEs[1:])
# TE_train = TE_train.reshape(1, -1)
# print(TE_train)

# echo_times : [5,10,15]ms et TR=21ms
# TE_train = np.array([np.tile([5, 5, 4], n_pulses)])  # ms
# TR_train = np.array([np.tile([5, 6, 10], n_pulses)])  # ms

# echo_times : [5,10,15]ms et TR=20ms
# TE_train = np.array([np.tile([5, 5, 4], n_pulses)])  # ms
# TR_train = np.array([np.tile([5, 6, 9], n_pulses)])  # ms
if not input_seq or "phi_train" not in (seqparam := json.load(open(simus_infos['input_seq_path']))):
    if phase == 'alt':
        # bssp 0-180° alternance
        phi_bssfp = np.ones((1, round(N / n_echoes))) * 180
        phi_bssfp[:, ::2] = 0  # alternating phase
        phi_cycl = phi_bssfp
        phi_train = np.zeros((1, N))
        phi_train[:, ::n_echoes] = phi_cycl
    elif phase == 'quad':
        # quadratic phase increment
        vecteur_unite = np.arange(start=0, stop=round(N / n_echoes), step=1)
        phi_cycl = 0.5 * (vecteur_unite ** 2 + vecteur_unite + 2) * phase_incr
        phi_cycl = phase_incr * 0.5 * (vecteur_unite ** 2)
        phi_train = np.zeros((1, N))
        phi_train[:, ::n_echoes] = phi_cycl
    elif phase == 'quadalt':
        # quad phase & alternance 0-180
        vecteur_unite = np.arange(start=0, stop=round(N / n_echoes), step=1)
        # phi_cycl = 0.5 * (vecteur_unite ** 2 + vecteur_unite + 2) * phase_incr + 180 * vecteur_unite
        phi_cycl = phase_incr * 0.5 * (vecteur_unite ** 2) + 180 * vecteur_unite
        phi_train = np.zeros((1, N))
        phi_train[:, ::n_echoes] = phi_cycl
    elif phase == 'alt_then_quad':
        # quad phase in the steady state only (after 40 pulses) & alternance 0-180
        vecteur_unite = np.arange(start=0, stop=round(N / n_echoes - 40), step=1)
        phi_cycl = 0.5 * (vecteur_unite ** 2 + vecteur_unite + 2) * phase_incr + 180 * vecteur_unite
        phi_cycl = phase_incr * 0.5 * (vecteur_unite ** 2) + 180 * vecteur_unite
        phi_bssfp = np.ones((1, round(40))) * 180
        phi_bssfp[:, ::2] = 0  # alternating phase in 40 pulses of transient state
        phi_train = np.zeros((1, N))
        phi_train[:, 0:40 * n_echoes:n_echoes] = phi_bssfp
        phi_train[:, 40 * n_echoes:N:n_echoes] = phi_cycl
    elif phase == 'zero':
        # no phase
        phi_train = np.zeros((1, N))

if not input_seq:
    if spoil == 'FISP':
        spoil = 1
    elif spoil == 'bSSFP':
        spoil = 0
    else:
        raise ValueError('Wrong spoil type in argument')
############################# DICO PARAMETERS #############################


directory_save_path = output_path + \
                      "Dico" + name_seq + phase

if not os.path.exists(directory_save_path):
    # Create the folder if it doesn't exist
    os.makedirs(directory_save_path)
    print(f"Folder '{directory_save_path}' created successfully.")
else:
    print(f"Folder '{directory_save_path}' already exists.")

json_filename = os.path.basename(args.json_path)
destination_file_path = os.path.join(directory_save_path + '/', json_filename)
shutil.copyfile(args.json_path, destination_file_path)


gammalist = eval(gamma_list)


if param == 'expanded':
    # dfrange = [-100, 99, 100]  # Hz
    # dfrange = [-50, 49, 100]  # Hz
    # dfrange = [-200, 198, 200]
    dfrange = [-200,199,400]
elif param == 'classic':
    if vasc:
        dfrange = [-50, 49, 100]  # Hz
    else:
        dfrange = [0, 0, 1]  # Hz
else:
    raise ValueError('Wrong param type in argument')
######## SET THE GRID MODEL #########
grid = 'regular'  # 'sobol' for sobol distrib of T1,T2,B1, 'regular' for regular griding

######### REGULAR GRID ##########
if grid == 'regular':
    # T1values = np.concatenate((np.linspace(205, 245, 20), np.linspace(395, 435, 20), np.linspace(690,730,20), np.linspace(1110,1150,20),np.linspace(1280,1320,20), np.linspace(1600,1640,20), np.linspace(2060,2100,20), np.linspace(2360,2400,20), np.linspace(2510, 2550,20)), axis=0) * 1e-3  # s
    T1values = np.linspace(T1range[0], T1range[1], T1range[2]) * 1e-3  # sec
    #T2values = np.concatenate(
    #    (np.linspace(10, 200, int(0.75 * T2range[-1])), np.linspace(200, 600, int(0.25 * T2range[-1]))),
    #    axis=0) * 1e-3  # s
    T2values = np.linspace(T2range[0], T2range[1], T2range[2]) * 1e-3  # sec
    B1values = np.linspace(B1range[0], B1range[1], B1range[2])
    if param == 'expanded':
        print("Loading distribution from 3D vessels geometries...")
        distrib = loadmat(distrib_path)
        big_structure = distrib['bigStructure']
        fn = list(big_structure.keys())
        dfvalues = big_structure[fn[0]]['histo']['values']
        # dfvalues = np.linspace(dfrange[0], dfrange[1], dfrange[2])
    else:
        dfvalues = np.round(np.linspace(dfrange[0], dfrange[1], dfrange[2]))

elif grid == 'sobol':
    ######### SOBOL GRID ###########
    l_bounds = [T1range[0], T2range[0], B1range[0]]
    u_bounds = [T1range[1], T2range[1], B1range[1]]
    sampler = qmc.Sobol(3, scramble=False)
    num_samples = 3000  # You can change this number as needed
    T1values = []
    T2values = []
    B1values = []
    for _ in range(num_samples):
        # Generate a random sample within the specified ranges
        sample = sampler.random()
        point = qmc.scale(sample, l_bounds, u_bounds)

        # Check if the condition is met (T1 > T2)
        if point[0][0] > point[0][1]:
            T1values.append(point[0][0])
            T2values.append(point[0][1])
            B1values.append(point[0][2])
    T1values = np.array(T1values) * 1e-3
    T2values = np.array(T2values) * 1e-3
    B1values = np.array(B1values)
    dfvalues = np.round(np.linspace(dfrange[0], dfrange[1], dfrange[2]))
    dfvalues = np.tile(dfvalues, B1values.shape)
    T1values = np.repeat(T1values, dfrange[2])
    T2values = np.repeat(T2values, dfrange[2])
    B1values = np.repeat(B1values, dfrange[2])

elif grid == 'custom':
    T1values = np.array([800, 1000, 1300, 2000]) * 1e-3
    T2values = np.array([80, 110, 200]) * 1e-3
    B1values = np.array([0.7, 1, 1.3])
    dfvalues = np.round(np.linspace(dfrange[0], dfrange[1], dfrange[2]))
#################################### distribs ####################################
if distrib and folder:
    if distrib_path and (which_distrib is None):
        parser.error("--distrib_path requires --distrib_path.")

    if which_distrib is not None:
        try:
            indices = slice(*map(int, which_distrib.split(':')))
        except ValueError:
            print("Invalid slice format. Please use 'start:stop:step' format.")

    if bloch_based_dico is not None:
        if not os.path.exists(bloch_based_dico):
            parser.error('Unexistent based bloch dico at path:', bloch_based_dico)

    if distrib_path is not None:
        print(which_distrib)
        distribs_names = os.listdir(distrib_path)
        distribs_names = distribs_names[indices]
        n_distrib = len(distribs_names)
        print("Using distributions :", n_distrib)

if not folder:
    n_distrib=20000
#################################### ####################################
ndf = dfvalues.shape[0]
nB1 = B1values.shape[0]
nT1 = T1values.shape[0]
nT2 = T2values.shape[0]

if param == 'classic':
    if grid == 'regular':
        print("Generation of {} signals".format(ndf * nT1 * nT2 * nB1))
    elif grid == 'sobol':
        print("Generation of {} signals".format(ndf * nT1))

elif param == 'expanded':
    if grid == 'regular':
        if vasc:
            print(
                "Generation of {} signals first, expanded vascular parameters into {} signals".format(
                    ndf * nT1 * nT2 * nB1,
                    ndf * nT1 * nT2 * nB1 *
                    n_distrib))
        else:
            print("Generation of {} signals first, expanded T2star into {} signals".format(ndf * nT1 * nT2 * nB1,
                                                                                           ndf * nT1 * nT2 * nB1 *
                                                                                           gammalist.shape[0]))
    elif grid == 'sobol':
        if vasc:
            print(
                "Generation of {} signals first, expanded vascular parameters into {} signals".format(ndf * nT1,
                                                                                                      ndf * nT1 *
                                                                                                      n_distrib))
        else:
            print("Generation of {} signals first, expanded T2star into {} signals".format(ndf * nT1,
                                                                                           ndf * nT1 *
                                                                                           gammalist.shape[0]))
if not os.path.exists(directory_save_path):
    os.makedirs(directory_save_path)
    print('Output directory created at:', directory_save_path)
else:
    print("File already exists on summer and will be overwritted.\n", directory_save_path)

############################## SIMULATIONS #################################


tic = time.time()

if slice_profile:
    if not variableFA:
        FA_distrib = np.array(
            eng.SliceProfile(matlab.double(FA_train_noME.tolist()), matlab.double([[3]]), matlab.double([[1]])))
        FA_distrib.sort()
        # print(FA_distrib.shape)
        # print(np.unique(FA_distrib)[0])
        unique_profile, count = np.unique(FA_distrib, return_counts=True)
        for FA in unique_profile:
            print('FA=', FA)
            print('{} / {}'.format(np.where(unique_profile == FA)[0][0], unique_profile.shape[0]))
            FA_train = np.zeros((1, N))
            FA_train[:, ::n_echoes] = FA
            Data_based, Mag_fa_based = dico_based(T1values, T2values, B1values, dfvalues, FA_train, TR_train, TE_train,
                                                  phi_train, n_echoes, spoil, vasc, inv_pulse, inv_time, grid, eng)
            # print('Based dico generated :', Data_based['parameters'].shape[0], 'signals')
            if param == 'classic':
                Data = Data_based
                Mag_fa = Mag_fa_based
            elif param == 'expanded':
                # print('Generation of expanded dico of', Data_based['parameters'].shape[0] * gammalist.shape[0], 'signals', )
                Data, Mag_fa = generate_gamma_dico_par(Data_based, Mag_fa_based, gammalist)
            if FA == unique_profile[0]:
                Mag = Mag_fa * count[np.where(unique_profile == FA)[0][0]]
            else:
                Mag = Mag + Mag_fa * count[np.where(unique_profile == FA)[0][0]]
        Mag = Mag / len(FA_distrib.T)
        print('Final number of signal after df0 values truncature :{}'.format(Data['parameters'].shape[0]))

    elif variableFA:
        simu_to_do = np.array(
            eng.SliceProfile(matlab.double(FA_train_noME.tolist()), matlab.double([[3]]),
                             matlab.double([[1]])))  # slice thickness, sampling factor
        unique_profiles = np.unique(simu_to_do, axis=1)
        i = 1
        total_c = 0
        for fatrain in unique_profiles.T:
            print('{} / {}'.format(i, unique_profiles.shape[1]))
            count = np.count_nonzero(np.all(simu_to_do == fatrain.reshape((n_pulses, 1)),
                                            axis=0))  # number of occurence of this fatrain in the list of fatrain to simulate
            FA_train = np.zeros((1, N))
            FA_train[:, ::n_echoes] = fatrain
            Data_based, Mag_fa_based = dico_based(T1values, T2values, B1values, dfvalues, FA_train, TR_train, TE_train,
                                                  phi_train, n_echoes, spoil, vasc, inv_pulse, inv_time, grid, eng)
            # print('Based dico generated :', Data_based['parameters'].shape[0], 'signals')
            if param == 'classic':
                Data = Data_based
                Mag_fa = Mag_fa_based
            elif param == 'expanded':
                # print('Generation of expanded dico of', Data_based['parameters'].shape[0] * gammalist.shape[0], 'signals', )
                Data, Mag_fa = generate_gamma_dico_par(Data_based, Mag_fa_based, gammalist)
            if np.array_equal(fatrain, unique_profiles.T[0]):
                Mag = Mag_fa * count
            else:
                Mag = Mag + Mag_fa * count
            i += 1
            total_c += count
        Mag = Mag / total_c
        print('Final number of signal after df0 values truncature :{}'.format(Data['parameters'].shape[0]))
    else:
        raise ValueError('Missing FA train variation argument')

elif not slice_profile:
    if bloch_based_dico is not None:
        dico = mat73.loadmat(bloch_based_dico)
        Mag = dico['Dico']['MRSignals'][0]
        Mag_based = Mag[:, :, np.newaxis]
        Param = dico['Dico']['Parameters']['Par']
        Sequence = np.zeros((N, 4))
        Sequence[:, 0] = dico['Dico']['Sequence']['FA_train']
        Sequence[:, 1] = dico['Dico']['Tacq']
        Sequence[:, 2] = dico['Dico']['Sequence']['TR_train']
        Sequence[:, 3] = dico['Dico']['Sequence']['phi']
        Data_based = {'parameters': Param, 'sequence': Sequence}
        print('Based dico loaded :', Data_based['parameters'].shape[0], 'signals')
    else:
        Data_based, Mag_based = dico_based(T1values, T2values, B1values, dfvalues, FA_train, TR_train, TE_train,
                                           phi_train, n_echoes, spoil, vasc, inv_pulse, inv_time, grid, eng)
        print('Based dico generated :', Data_based['parameters'].shape[0], 'signals')
        #print(repr(Mag_based[20,:,0]))
    if param == 'classic':
        Data = Data_based
        Mag = Mag_based
    elif param == 'expanded':
        if vasc:
            if not folder:
                n_distrib=20000
            print('Generation of expanded dico of', Data_based['parameters'].shape[0] * n_distrib, 'signals', )
            Data, Mag = generate_vasc_dico_par_2(Data_based, Mag_based,
                                               distrib)
        else:
            print('Generation of expanded dico of', Data_based['parameters'].shape[0] * gammalist.shape[0], 'signals', )
            Data, Mag = generate_gamma_dico_par(Data_based, Mag_based, gammalist)
        print('Final number of signal after df0 values truncature :{}'.format(Data['parameters'].shape[0]))

else:
    raise ValueError('Missing slice profile argument')

################################ LOGS ################################
txt_path = directory_save_path + "/Dico.txt"
fd = open(txt_path, 'w')

print('Based dico generated :', Data_based['parameters'].shape[0], 'signals', file=fd)
print('Final number of signal after df0 values truncature :{}'.format(Data['parameters'].shape[0]), file=fd)
print('Ranges:', file=fd)
print('nT1=', nT1, 'nT2=', nT2, 'nB1=', nB1, 'ndf=', ndf, file=fd)
print('T1 from', np.min(Data['parameters'][:, 0]), 'to', np.max(Data['parameters'][:, 0]), 's', file=fd)
print('T2 from', np.min(Data['parameters'][:, 1]), 'to', np.max(Data['parameters'][:, 1]), 's', file=fd)
print('df from', np.min(Data['parameters'][:, 2]), 'to', np.max(Data['parameters'][:, 2]), 'Hz', file=fd)
print('B1 from', np.min(Data['parameters'][:, 4]), 'to', np.max(Data['parameters'][:, 4]), 'Hz', file=fd)
if vasc and param == 'expanded':
    print(n_distrib, ' 3D voxels distributions were used from :', distrib_path, 'at position', which_distrib,
          file=fd)
    print(n_distrib, ' 3D voxels distributions were used from :', distrib_path, 'at position', which_distrib)

print('Ranges:')
print('nT1=', nT1, 'nT2=', nT2, 'nB1=', nB1, 'ndf=', ndf)
print('T1 from', np.min(Data['parameters'][:, 0]), 'to', np.max(Data['parameters'][:, 0]), 's')
print('T2 from', np.min(Data['parameters'][:, 1]), 'to', np.max(Data['parameters'][:, 1]), 's')
print('df from', np.min(Data['parameters'][:, 2]), 'to', np.max(Data['parameters'][:, 2]), 'Hz')
print('B1 from', np.min(Data['parameters'][:, 4]), 'to', np.max(Data['parameters'][:, 4]), 'Hz')
print('Commentaries:', commentaries, file=fd)
Dico = {'Parameters': Data['parameters'], 'Sequence': Data['sequence'], 'VoxelList': Data['voxellist'], 'MRSignals': Mag}
toc = time.time()
print('Generation time:', toc - tic, 'seconds', file=fd)
print('Generation time:', toc - tic, 'seconds')

fd.close()

# ################################ SAVING IN .h5 AND .mat ################################
print('Saving in hdf5 format ...')
fname = directory_save_path + "/DICO.h5"
#if os.path.exists(fname):
#    os.remove(fname)
f = h5py.File(fname, 'w')
for data_name in Dico:
    f.create_dataset(data_name, data=Dico[data_name])
f.close()
#
print('Converting .hdf5 file in .mat ...')
mat_path = directory_save_path + "/DICO.mat"
if vasc:
    eng.function_make_dico_struct_vasc(fname, mat_path, nargout=0)
else:
    eng.function_make_dico_struct(fname, mat_path, nargout=0)
eng.quit()

################################ compress dictionary ################################
if SVD:
    print('Compressing and saving in .mat...')
    parameters = matlab.double(Data['parameters'].tolist(), size=Data['parameters'].shape)
    sequence = matlab.double(Data['sequence'].tolist(), size=Data['sequence'].shape)
    mag = matlab.double(Mag.tolist(), size=Mag.shape, is_complex=True)
    mat_path = directory_save_path + "/DICO.mat"
    eng.compress_svd(parameters, sequence, mag, mat_path, nargout=0)
    eng.compress_svd_fromH5(fname, mat_path, matlab.double([[rank]]), nargout = 0)
