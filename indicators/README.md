# Indicators script

This script creates indicator csv/tab files from an UrbanSim2 hdf5 output file.


## Setup

Create a 'csv\_store' directory inside the indicators directory and place these files in it: cities.csv, subregs.csv, growth\_centers.csv, parcels\_geos.csv. They can be exported from the base year database. The parcels\_geos.csv can be created via the `export_variables.py` script. It's a parcels table with columns: 

```
parcel_id, parcel_sqft, growth_center_id,faz_id, city_id, subreg_id,target_id, control_id, control_hct_id,
county_id, tod_id, zone_id, plan_type_id, census_tract_id
```

Note that this file is mainly needed for situations when the various ids are not in the parcels table that was included in the simulation. Thus, it allows to run indicators for new geographies, created after the simulation finished.


## Running the script

To create the indicator files do the following steps:

1. Edit `indicators_settings.yaml` with:
  * The urbansim output file name (node `store`)
  * The output directory for these indicators (node `output_directory`)
  * The base year (node `base_year`)
  * The years for which to produce indicator files (either node `years` listing the individual years, or for annual indicators, node `years_all` as start year and end year)
  * The indicators and their corresponding geography levels (node `indicators` - defines indicators as one file per indicator and geography with columns being the years; node `dataset_tables` - defines indicators as one file per year with columns being indicators).
2.  To run `run_indicators.py` in bash:

```
cd /d/udst
source setpath.sh
cd psrc_urbansim/indicators
python run_indicators.py

```
 Or to run `run_indicators.py` in the command line:

```
d:
cd udst
setpath.bat
cd psrc_urbansim/indicators
python run_indicators.py

```

The resulting csv and tab files will be in the directory defined by the `output_directory` node.