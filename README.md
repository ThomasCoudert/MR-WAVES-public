# MR-WAVES
MR Water-diffusion And Vascular Effects Simulations tools. Can be used for Magnetic Resonance Fingerprinting.

# Intra-voxel magnetic field distributions
### OLD VERSION USING MRVOX: 
Choose the folder on summer that contain all .mat file of pre-computed distributions (actually done by MRVox)
```
/data_network/summer/projects/Current/2021_MRF_TC/distributions_2024
```
Create a simplified table (matlab structure) with all the distributions histograms (pretty fast if Geometry aren't stored, slow otherwise):
```
choose_distrib_from_folder.m
```
==> run it one time for pre and one time for post contrast, it will create a .mat table with all the distributions (geo1, geo2, ...) that will be used for simulations.

### NEW VERSION IN PYTHON: 

Run `InitSimu3D.ipynb` to generate directly the .mat table based on the voxel (magnetic field computations are done in Python)

# Simulations
Simulations (pratt): 
```
python main_simu.py -json config.json
```
==> where the config json file specifies the json defining the GESFIDE as well as the structure containing distribution histograms computed just before
==> run one simu for pre-CA signal, one for post-CA

# ADD DIFFUSION
use the RNN, in a local notebook, that will compute the diffusion into the dictionary
```
inference_dico_Distrib.ipynb
```
In MATLAB:
```
fromRNNtoMP3.m
``` 
will concatenate pre/post dictionary and format for MP3 and other matlab functions

Then `add_T2.m` add an exponential of choosen T2 values over the signals.

# No diffusion:

Computed dictionaries PRE+POST need to be concatenated and processed before matching:
```
fromDistribtoMP3.m
```
==> compute a single dictionary in the appropriate format


Then : `add_T2.m` to directly use the dico and match after in MP3.


And lets go for matching in MP3!

