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

The packages are already installed there as well as the baseyear data is available. To set the environment variables in step 4 and 6, open a terminal, go into ``d:/synthicity`` and run ``setpath.bat``. It changes the environment only for the session in this terminal window.

### Estimation

To estimate, run the ``estimate.py`` script. The configuration of the REPM can be found in ``configs/repm.yaml`` where the outputs are also written into. New variables can be defined in ``psrc_urbansim/psrc_urbansim/variables.py``.

### Simulation

To simulate, run the script ``simulate.py``. It contains a name of the output file.

