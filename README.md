# PSRC UrbanSim

[![Travis-CI Build Status] (https://travis-ci.org/psrc/urbansim2.svg?branch=master)] (https://travis-ci.org/psrc/urbansim2)
[![Coverage Status](https://coveralls.io/repos/github/psrc/urbansim2/badge.svg?branch=master)](https://coveralls.io/github/psrc/urbansim2?branch=master)

This is an urbansim-2 implementation of the PSRC land use model. It is a python package that contains PSRC-specfic modifications to the urbansim package developed by Synthicity.

It's under construction. Currently, only a simple version of the real estate price model (REPM) is implemented.

###Setup:

1. Install Anaconda Python: http://continuum.io/downloads
2. Clone this repository into a directory called ``psrc_urbansim``.
3. Install the [UDST packages](https://github.com/UDST) ``urbansim``, ``urbansim_defaults`` and ``orca``. 
4. Set the environmental variable PYTHONPATH to point to those directories, as well as this repository, ``psrc_urbansim``.
5. Create a base year dataset as an hdf5 file by running the script ``data/conversion/cache_to_hdf5.py`` (see [more info](https://github.com/psrc/urbansim2/tree/master/data/conversion)). Move the resulting file into ``psrc_urbansim/data``.
6. Set the environment variable ``DATA_HOME`` to the directory with your base year dataset (minus the ``data`` subdirectory), i.e. to the absolute path of ``psrc_urbansim``. The code will look for the data file in $DATA_HOME/data.
7. Put the name of the data file into ``psrc_urbansim/psrc_urbansim/assumptions.py``.

### Setup Note for modelsrv3

The packages are already installed there as well as the baseyear data is available. To set the environment variables in step 4 and 6, open a terminal, go into ``d:/synthicity`` and run ``setpath.bat``. Alternatively, in a Git Bash window do

```
cd /d/synthicity
source setpath.sh
```

In both cases, it changes the environment only for the session in this terminal or bash window.

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

