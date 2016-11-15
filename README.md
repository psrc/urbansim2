# PSRC UrbanSim

[![Travis-CI Build Status] (https://travis-ci.org/psrc/urbansim2.svg?branch=master)] (https://travis-ci.org/psrc/urbansim2)
[![Coverage Status] (https://coveralls.io/repos/github/psrc/urbansim2/badge.svg?branch=master)] (https://coveralls.io/github/psrc/urbansim2?branch=master)

This is an urbansim-2 implementation of the PSRC land use model. It is a python package that contains PSRC-specfic modifications to the urbansim package developed by UrbanSim Inc. (former Synthicity).


##Installation
###Setup:
In the examples below it will be assumed that the base directory for the installation is ``d:/udst``.


1. Install [Anaconda Python](http://continuum.io/downloads). By default it will be installed in a different directory than existing Python, so there is no danger in messing up the current Python installation.
2. Clone this repository into a directory called ``psrc_urbansim``:
   
   ```
   cd /d/udst
   git clone https://github.com/psrc/urbansim2.git psrc_urbansim
   ```
   
3. Install the UDST packages ``urbansim``, ``urbansim_defaults``, ``orca`` and ``pandana`` by cloning them from [UDST GitHub](https://github.com/UDST):

   ```
   cd /d/udst
   git clone https://github.com/UDST/urbansim.git urbansim
   git clone https://github.com/UDST/urbansim_defaults.git urbansim_defaults
   git clone https://github.com/UDST/orca.git orca
   git clone https://github.com/UDST/pandana.git pandana
   ```
   
4. Set the environment variable PYTHONPATH to point to those directories, as well as this repository, ``psrc_urbansim``. If you plan to switch between Opus and UrbanSim-2, put these settings into a  file that can be executed prior to working in the UrbanSim-2 environment. E.g. create a file ``setpath.bat`` with 

   ```
   SET PYTHONPATH=D:/udst/psrc_urbansim;D:/udst/urbansim;D:/udst/urbansim_defaults;D:/udst/orca;D:/udst/pandana
   ```
   
5. Set the PATH variable to point to the Anaconda directory. E.g. add this line to the ``setpath.bat`` file:
   
   ```
   SET PATH=c:/Anaconda;c:/Anaconda/Scripts;%PATH%
   ```
    
6. Create a base year dataset as an hdf5 file by running the script [``psrc_urbansim/data/conversion/cache_to_hdf5.py``](https://github.com/psrc/urbansim2/tree/master/data/conversion/cache_to_hdf5.py) (see [more info](https://github.com/psrc/urbansim2/tree/master/data/conversion)). Move the resulting file into ``psrc_urbansim/data``.
7. Set the environment variable ``DATA_HOME`` to the directory with your base year dataset (minus the ``data`` subdirectory). The code will look for the data file in $DATA_HOME/data. E.g. add this line to ``setpath.bat``:
 
   ```
   SET DATA_HOME=D:/udst/psrc_urbansim
   ```

8. Put the name of the data file into ``psrc_urbansim/configs/settings.yaml`` (node ``store``).
9. There might be a few changes to the ``urbansim_defaults`` package that were submitted as pull requests to UDST but were not accepted yet. To keep the repository in sync with those changes do
  
  ```
  cd /d/udst/psrc_urbansim
  git remote add psrcedits https://github.com/hanase/urbansim_defaults.git
  git pull psrcedits master
  ``` 

### Setup Note for modelsrv3

The packages are already installed there as well as the baseyear data is available. To set the environment variables in step 4, 5 and 7, open a terminal, go into ``d:/synthicity`` and run ``setpath.bat``. Alternatively, in a Git Bash window do

```
cd /d/synthicity
source setpath.sh
```

In both cases, it changes the environment only for the session in this terminal or bash window.

### Code Update

The code is evolving fast, so update it regularly.

```
cd /d/udst/urbansim
git pull 
cd ../orca
git pull
cd ../pandana
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

