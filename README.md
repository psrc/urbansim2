# PSRC UrbanSim

[![Travis-CI Build Status](https://travis-ci.org/psrc/urbansim2.svg?branch=master)](https://travis-ci.org/psrc/urbansim2)
[![Coverage Status](https://coveralls.io/repos/github/psrc/urbansim2/badge.svg?branch=master)](https://coveralls.io/github/psrc/urbansim2?branch=master)

This is an urbansim-2 implementation of the PSRC land use model. It is a python package that contains PSRC-specfic modifications to the urbansim package developed by UrbanSim Inc. (former Synthicity). 


## Installation

### Setup:

In the examples below it will be assumed that the base directory for the installation is ``d:/udst``.


1. Install [Anaconda Python](http://continuum.io/downloads), the latest of the 3.* series. By default it will be installed in a different directory than existing Python, so there is no danger in messing up the current Python installation. Alternatively, use a virtual environment specific for urbansim2. In a command prompt, start a new virtual environment called "urbansim2" as follows:

```
conda create -n urbansim2 python=3.9 anaconda
```
Activate this environment every time you restart the prompt and want to work with urbansim2 by entering the following (for Windows prompts):

```
activate urbansim2
```

To achieve the same thing in a bash shell, type `source activate urbansim2` or `conda activate urbansim2`. To deactivate type `conda deactivate`.

In addition to Anaconda Python, three other packages (zbox, prettytable and pylogit) are needed. Install using the following pip commands:
   
   ```
   pip install zbox
   pip install prettytable
   pip install pylogit
   ```
   
2. Clone this repository into a directory called ``psrc_urbansim``:
   
   ```
   cd /d/udst
   git clone https://github.com/psrc/urbansim2.git psrc_urbansim
   ```
   To use the python3-specific branch, switch to `dev_python3`:
   
   ```
   cd psrc_urbansim
   git switch dev_python3
   cd ..
   ```
    
   
3. Install various UDST packages by cloning them from [UDST GitHub](https://github.com/UDST):

   ```
   cd /d/udst
   git clone https://github.com/UDST/urbansim.git urbansim
   git clone https://github.com/UDST/urbansim_defaults.git urbansim_defaults
   git clone https://github.com/UDST/orca.git orca
   git clone https://github.com/UDST/pandana.git pandana
   git clone https://github.com/UDST/developer.git developer
   git clone https://github.com/UDST/choicemodels.git choicemodels
   ```
   
4. **Set environment variables:** 
	* `PYTHONPATH` points to the downloaded UDST repositories, as well as this repository, `psrc_urbansim`. 
	* `DATA_HOME` points to the directory where data is stored (minus the `data` subdirectory, see the next bullet about data preparation). The code will look for the data file in `$DATA_HOME/data`.
	* `PATH` should include the Anaconda directory (needed if not working in a conda environment).
	
	There are a few different ways to set those variables other than the usual OS-specific ways:
  
  a) (recommended) If you are working with a conda environment, one can do the setting as follows:

   ```
  conda env config vars set PYHTONPATH="D:/udst/psrc_urbansim;D:/udst/urbansim;D:/udst/urbansim_defaults;D:/udst/orca;D:/udst/choicemodels;D:/udst/developer;D:/udst/pandana"
  conda env config vars set DATA_HOME="D:/udst/psrc_urbansim"
  ```
     
   To check if it worked, re-activate the environment and list all environment variables in it:
     
   ```
     conda deactivate
     conda activate urbansim2
     conda env config vars list
   ```

   b) If you plan to switch between Opus and UrbanSim-2, but do not work with a conda environment, put these settings into a file that can be executed prior to working in the UrbanSim-2 environment. E.g. create a file ``setpath.bat`` with 

   ```
   SET PYTHONPATH=D:/udst/psrc_urbansim;D:/udst/urbansim;D:/udst/urbansim_defaults;D:/udst/orca;D:/udst/choicemodels;D:/udst/developer;D:/udst/pandana
   SET DATA_HOME=D:/udst/psrc_urbansim
   SET PATH=c:/Anaconda;c:/Anaconda/Scripts;%PATH%
   ```
   
   If you prefer to work with Git Bash, you can put something like this into a file called ``setpath.sh``:
   
   ```
   DIR=/d/udst
   export PYTHONPATH=$DIR/psrc_urbansim:$DIR/urbansim:$DIR/urbansim_defaults:$DIR/orca:$DIR/choicemodels:$DIR/developer:$DIR/pandana
   export DATA_HOME=$DIR/psrc_urbansim
   export PATH=/c/Anaconda:/c/Anaconda/Scripts:$PATH
   ```
       
6. Create a base year dataset as an hdf5 file by running the script [``psrc_urbansim/data/conversion/cache_to_hdf5.py``](https://github.com/psrc/urbansim2/tree/master/data/conversion/cache_to_hdf5.py) (see [more info](https://github.com/psrc/urbansim2/tree/master/data/conversion)). Move the resulting file into ``psrc_urbansim/data``.

    Note that up-to-date base year files are kept on modelsrv3 in `/d/udst/psrc_urbansim/data`. For the use with python3, use the files with suffix "*py3.h5".

7. Put the name of the data file into `psrc_urbansim/configs/settings.yaml` (simulation, estimation) or `psrc_urbansim/configs/settings_allocation.yaml` (allocation), in node `store`.

### Code Update

The code is evolving, so update it regularly.

```
cd /d/udst/urbansim
git pull
cd ../orca
git pull
cd ../pandana
git pull
cd ../urbansim_defaults
git pull
```
... etc. This can be automated as has been done on modelsrv3, see the next section.

### Setup Note for modelsrv3

On modelsrv3, the packages are already installed, as well as the baseyear data is available. To update the code, open a Git Bash and do:

```
cd /d/udst
./update_all.sh
```

The script iterates over the packages and pulls from the corresponding repositories.

To set the environment variables in step 4, depending where you want to run UrbanSim-2:

1. **Windows command line:** Open a terminal, go into the ``d:/udst`` directory and do:

   ```
   setpath.bat
   ```
 
2. **Git Bash**: Open a Git Bash window and do 
 

   ```
   cd /d/udst
   source setpath.sh
   ```

In both cases, it changes the environment only for the session in this terminal or bash window.

The base year data are stored in ``/d/udst/psrc_urbansim/data``. The file to be used for estimation is ``psrc_estimation_2018_py3.h5``. For a simulation use ``psrc_base_year_2018_py3.h5``. For allocation use ``psrc_base_year_2018_alloc_py3.h5``.


## Using UrbanSim-2

Note that the code is under construction and not everythng will work. Here is documentation of its [status](https://github.com/psrc/urbansim2/wiki/Implementation-status). 

### Estimation

Model estimation is controlled from the file ``estimate.py``. In that file, uncomment a line corresponding to the model to be estimated. For example, for estimating residential real estate price model, make sure the line  

```
orca.run(["repmres_estimate"])
```

is uncommented, while all other lines except imports are commented out. Put the name of the estimation data file into ``configs/settings.yaml`` under the node ``store``.

Then run 

```
python estimate.py
```

The UI of the various models is implemented in ``psrc_urbansim/models.py``. For the example above, we'll find the following section in the ``models.py`` file:

```
@orca.step('repmres_estimate')
def repmres_estimate(parcels, zones, gridcells):
    return utils.hedonic_estimate("repmres.yaml", parcels, [zones, gridcells],
                                  out_cfg = "repmrescoef.yaml")
```

This tells us that the model is using a specification defined in the file  ``configs/repmres.yaml``. Estimated coefficients are stored in ``configs/repmrescoef.yaml``.  New variables should be implemented in the directory ``psrc_urbansim/vars``, in a file that corresponds to the affected dataset. 



### Simulation

A simulation can be started from the file ``simulate.py``. Here, uncomment all models you want to run and define the set of simulation years. Outputs will be written into a file defined in the argument ``data_out``.

```
python simulate.py
```


## Pushing Changes to GitHub

For now, since everything is under development, we will push all our changes into the master branch (unless you want to have your own experimental branch). Python3 code is in the `dev_python3` branch.

### Exclusions

There are a few files that are either automatically overwritten by the estimation procedure (e.g. yaml config files) or that have temporary changes not to be committed (e.g. estimate.py, simulate.py) and thus, we want them to be excluded from commits. There are bash scripts in the main directory that can help with that. First check with ``git status`` if some of these "unwanted" files are in line for a commit. If it is the case, BEFORE you run commit, you can use (from a bash terminal):

* ``source gitexclude_yaml.sh``: excludes all yaml files found in the configs directory.
* ``source gitexclude.sh filename``: excludes a specific file given by the filename.
* ``source gitinclude.sh filename``: includes a previously excluded file given by the filename. 

Then run commit and push. For example, you made changes in the HLCM specification that should be committed, but you also ran test estimations of other models results of which should not be pushed to GitHub, neither the estimate.py file. In such a case, you can do:

```
source gitexclude_yaml.sh
source gitinclude.sh configs/hlcm.yaml
source gitexclude.sh estimate.py
git status
git commit -am 'describe your changes'
git push
```

The first line excludes all yaml files while the second line "unexcludes" hlcm.yaml and the third line adds the estimate.py file to the exclusions. Always check with ``git status`` that it is doing what you want.

### Merging

If you excluded a file and somebody else makes changes to it that collide with yours, the ``git pull`` command will most likely throw an error. There are various ways to deal with it depending on if you want to keep your changes or not. For keeping your changes, look at the [git stash documentation](https://git-scm.com/book/en/v1/Git-Tools-Stashing). If you want to throw your changes away, do 

```
git checkout filename
```

with filename being the file you want to overwrite.
