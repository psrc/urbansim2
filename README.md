# PSRC UrbanSim

[![Travis-CI Build Status] (https://travis-ci.org/psrc/urbansim2.svg?branch=master)] (https://travis-ci.org/psrc/urbansim2)
[![Coverage Status] (https://coveralls.io/repos/github/psrc/urbansim2/badge.svg?branch=master)] (https://coveralls.io/github/psrc/urbansim2?branch=master)

This is an urbansim-2 implementation of the PSRC land use model. It is a python package that contains PSRC-specfic modifications to the urbansim package developed by UrbanSim Inc. (former Synthicity).


##Installation
###Setup:

1. Install Anaconda Python: http://continuum.io/downloads
2. Clone this repository into a directory called ``psrc_urbansim``.
3. Install the UDST packages ``urbansim``, ``urbansim_defaults`` and ``orca`` by cloning them from [UDST GitHub](https://github.com/UDST). 
4. Set the environmental variable PYTHONPATH to point to those directories, as well as this repository, ``psrc_urbansim``.
5. Create a base year dataset as an hdf5 file by running the script ``data/conversion/cache_to_hdf5.py`` (see [more info](https://github.com/psrc/urbansim2/tree/master/data/conversion)). Move the resulting file into ``psrc_urbansim/data``.
6. Set the environment variable ``DATA_HOME`` to the directory with your base year dataset (minus the ``data`` subdirectory), i.e. to the absolute path of ``psrc_urbansim``. The code will look for the data file in $DATA_HOME/data.
7. Put the name of the data file into ``psrc_urbansim/configs/settings.yaml`` (node ``store``).
8. There might be a few changes to the ``urbansim_defaults`` package that were submitted as pull requests but were not accepted yet. To keep the repository in sync with those changes, navigate to ``urbansim_defaults`` and do
  
  ```
  git remote add psrcedits https://github.com/hanase/urbansim_defaults.git
  git pull psrcedits master
  ``` 

### Setup Note for modelsrv3

The packages are already installed there as well as the baseyear data is available. To set the environment variables in step 4 and 6, open a terminal, go into ``d:/synthicity`` and run ``setpath.bat``. Alternatively, in a Git Bash window do

```
cd /d/synthicity
source setpath.sh
```

In both cases, it changes the environment only for the session in this terminal or bash window.

### Code Update

The code is evolving fast, so update it regularly.

```
cd urbansim
git pull 
cd ../orca
git pull
cd ../urbansim_defaults
git pull
git pull psrcedits master
```

## Using UrbanSim-2

The code is under construction. Currently, only a prototype of the model system is implemented. One can estimate the real estate price model, household location choice model and job location choice model,  using currently only placeholder variables. A simulation script is also available. 

### Estimation

To estimate, go to the ``psrc_urbansim`` directory and do

```
python estimate.py
```
 
The configuration of the REPM can be found in ``configs/repm.yaml`` where the outputs are also written into. New variables can be defined in ``psrc_urbansim/variables.py``.

### Simulation

To simulate, from the ``psrc_urbansim`` directory run

```
python simulate.py
```

The script contains the name of the output file.

