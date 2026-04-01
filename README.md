# PSRC UrbanSim

This is an urbansim-2 implementation of the PSRC land use model. It is a python package that contains PSRC-specfic modifications to the urbansim package developed by UrbanSim Inc. (former Synthicity). 


## Installation

### Setup:

1. Install UV package manager:
   - Windows:
      ```
      powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
      ```
   - MacOS/Linux
      ```
      curl -LsSf https://astral.sh/uv/install.sh | sh
      ```

2. Clone the pandas23_refactor branch of the urbansim2 repository:
   ```
   git clone -b pandas23_refactor https://github.com/psrc/urbansim2.git
   ```

3. Create the UV venv and activate it:
   ```
   cd urbansim2
   uv sync
   .venv\Scripts\activate
   ```

6. Create a base year dataset as an hdf5 file by running the script [``data/conversion/cache_to_hdf5.py``](https://github.com/psrc/urbansim2/tree/master/data/conversion/cache_to_hdf5.py) (see [more info](https://github.com/psrc/urbansim2/tree/master/data/conversion)). Move the resulting file into ``data/``.

    Note that up-to-date base year files are kept on the N drive (Model Data 2) in `N:\base_year_2018_inputs\urbansim2_inputs`. For the use with python3, use the files with suffix "*py3.h5".

7. Put the name of the data file into `configs/settings.yaml` (simulation, estimation) or `configs/settings_allocation.yaml` (allocation), in node `store`.

8. Update the paths to `data_dir` and `output_dir` in `configs/settings.yaml`

### Using UrbanSim2

1. Run UrbanSim
   - Activate the venv if it isn't already by navigating to the UrbanSim2 repo then running:
      ```
      .venv\Scripts\activate
      ```
   - Allocation mode:
      ```
      psrc_urbansim run_allocation -c "<path to config directory>"
      ```
   - Simulation mode:
      ```
      psrc_urbansim run_simulation -c "<path to config directory>"
      ```

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
 
### Indicators

For generating indicators, see  [this page](https://github.com/psrc/urbansim2/tree/master/indicators).
