# MR-WAVES
MR Water-diffusion And Vascular Effects Simulations tools. Can be used for Magnetic Resonance Fingerprinting.

Developed at the Grenoble Institute Neurosciences (GIN) in the Neuroimaging Team.

by : Thomas Coudert, Maitê Marçal, Aurélien Delphin, Antoine Barrier, Benjamin Lemasson and Thomas Christen.


# PRE/POST contrast agent MRvF example:
## Intra-voxel magnetic field distributions

Histograms of frequency distributions associated to microvascular voxels need to be stored in a .mat table.

## Microvascular Bloch Simulations
```
python main_simu.py -json config.json
```
==> where the config json file specifies the json defining the MR sequence (here GESFIDE example) as well as the structure containing distribution histograms computed just before.

In this example, the sequence parameter are read from the `GESFIDE_MRvF` json file. 

==> run one simu for pre-CA signal, one for post-CA if needed.

## ADD DIFFUSION
use the RNN, in a local notebook, that will compute the diffusion into the dictionary
```
inference_dico_Distrib.ipynb
```
This should be done for pre and post parts of the dictionary if needed. It use a RNN model trained for pre and post contrast parts of the signal. 

## ADD T2
Simulations have been made at fixed T2=100ms in this example. Various T2 values can be included in the dictionary by simply multiplying signals by appropriate exponential functions for the GESFIDE example. 

Otherwise, varying T2 can be added in the `config.json` file in the first simulations part.


