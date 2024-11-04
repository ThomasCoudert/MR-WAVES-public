###Install dependencies:
This requires a Python version compatible with the installed Matlab version.

Activate the virtual environment for this project. For example if using virtualenv:
```shell script
workon t2star_mapping
```

If creating a new virtual environment, decide which Python version to use. Here we use python3.7 (the last version
that is compatible with Matlab R2021a)
```shell script
mkvirtualenv -p python3.7 t2star_mapping
```

Install dependencies. Not sure the "matlab_kernel" package is required
```shell script
pip install joblib matplotlib numpy scipy
pip install matlab_kernel
```

Set the correct target directory for installation of the Python Matlab libraries
```shell script
install_dir=$VIRTUAL_ENV
```

Deactivate the environment (the Matlab installation script does not work if this is active)
```shell script
deactivate
```

Go to Matlab engine folder. For example, if Matlab is on your system path (warning: this calls Matlab and is sloooowww):
```shell script
cd `matlab -batch "disp(matlabroot)"`/extern/engines/python
```

Install the Matlab environment for Python by launching the installer (make sure to use the correct Python version)
```shell script
build_dir=`mktemp -d` && \
python3.7 setup.py build --build-base=${build_dir} install --prefix=${install_dir} && \
rm -rf ${build_dir}
```
