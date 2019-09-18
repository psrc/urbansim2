# Indicators script

This script creates indicator csv/tab files from an Urbansim2 Simulation output hdf5 file.

## Setup

Create a 'csv_store' directory inside the indicators directory and place a copy of parcels_geos.csv and growth_centers.csv.  These files can be download here (coming soon).


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