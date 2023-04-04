# Indicators script

This script creates indicator csv/tab files from an Urbansim2 Simulation output hdf5 file.


## Setup

Create a 'csv\_store' directory inside the indicators directory and place these files in it: cities.csv, subregs.csv, growth\_centers.csv, parcels\_geos.csv. They can be exported from from the base year database. The parcels\_geos.csv can be created via the `export_variables.py` script. It's a parcels table with columns: 

```
parcel_id, parcel_sqft, growth_center_id,faz_id, city_id, subreg_id,
county_id, tod_id, zone_id, plan_type_id, census_tract_id
```




## Running the script

To create the indicator files do the following steps:

1. Edit indicators_settings.yaml with:
  * The urbansim output file name 
  * The base year
  * The years for which to produce indicator files
  * The indicators and their corresponding geography levels
2.  To run "run_indicators.py" in bash:

```
cd /d/udst
source setpath.sh
cd psrc_urbansim/indicators
python run_indicators.py

```
 Or to run "run_indicators.py" in the command line:

```
d:
cd udst
setpath.bat
cd psrc_urbansim/indicators
python run_indicators.py

```

The resulting csv and tab files will be in the 'indicators' directory.