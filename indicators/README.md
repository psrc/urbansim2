# Indicators script

These scripts create indicator csv/tab files from an Urbansim2 Simulation output hdf5 file.  
## Setup

1. Create a 'csv_store' directory inside the indicators directory and place a copy of parcels_geos.csv and growth_centers.csv.  These files can be download here (coming soon).

2. To run this scripts it is assumed that Urbansim2 has been installed and that setpath.bat or setpath.sh has been ran to establish the home directory.


## Running the script

To create the indicator files do the following steps:

1. Edit indicators_settings.yaml with:
 * The urbansim output file name
 * The base year
 * The years for which to produce indicator files
 * The indicators and their corresponding geography levels
2.  Run "run_indicators.py"

```
python run_indicators.py

```

The resulting csv and tab files will be in the 'indicators' directory.