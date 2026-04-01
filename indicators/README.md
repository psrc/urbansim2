# Indicators

Creates indicator csv/tab files from an UrbanSim2 hdf5 output file.


## Setup

Place CSV lookup files (growth\_centers.csv, parcels\_geos.csv, controls.csv, control\_hcts.csv, targets.csv) in the `data/csv_store/` directory. They can be exported from the base year database. The parcels\_geos.csv can be created via the `export_variables.py` script. It's a parcels table with columns: 

```
parcel_id, parcel_sqft, growth_center_id, faz_id, city_id, subreg_id, target_id, control_id, control_hct_id,
county_id, tod_id, zone_id, plan_type_id, census_tract_id
```

Note that this file is mainly needed for situations when the various ids are not in the parcels table that was included in the simulation. Thus, it allows to run indicators for new geographies, created after the simulation finished.


## Running indicators

1. Edit `configs/settings_indicators.yaml` with:
  * The results HDF5 filename (node `store`, resolved relative to `output_dir` from `settings.yaml`)
  * The years for which to produce indicator files (either node `years` listing the individual years, or for annual indicators, node `years_all` as start year and end year)
  * The indicators and their corresponding geography levels (node `indicators` - defines indicators as one file per indicator and geography with columns being the years; node `dataset_tables` - defines indicators as one file per year with columns being indicators).
  * Note: `base_year`, `data_dir`, and `output_dir` are inherited from `settings.yaml`.

2. Run via the CLI:

```
psrc_urbansim run_indicators -c "<path to configs>"
```

The resulting csv and tab files will be placed in the `output_dir` defined in `settings.yaml`.