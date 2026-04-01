# Plan: CLI-based run_indicators command

## Overview
Add `psrc_urbansim run_indicators -c "<path to configs>"` CLI command following the same pattern as `run_allocation` and `run_simulation`. Move indicator config to configs/, move csv_store to data/, eliminate DATA_HOME dependency, merge settings.yaml with the indicators config, and output CSV/tab files to a subfolder named after the store file (with "results" replaced by "indicators").

## Move static files

1. **Move csv_store files to data/**
   - Move `indicators/csv_store/controls.csv`, `control_hcts.csv`, `growth_centers.csv`, `parcels_geos.csv`, `targets.csv` to `data/csv_store/` (preserve subfolder)
   - Move `indicators/csv_store/.gitignore` as well

2. **Create `configs/settings_indicators.yaml`**
   - Adapted from `indicators/indicators_settings.yaml`
   - Remove absolute paths for `store` and `output_directory`; `store` becomes just a filename resolved relative to `output_dir` from settings.yaml
   - Remove `output_directory` (use `output_dir` from settings.yaml)
   - Remove `base_year` (comes from settings.yaml)
   - Keep: `years`, `years_all`, `compute_dataset_tables`, `indicators`, `dataset_tables`, `new_datasets`

## Extract and merge indicator logic into the package

3. **Create `psrc_urbansim/mod/indicators.py`** — merged content from both `indicators/indicators.py` and `indicators/data.py`
   - **Data source section** (from data.py): all orca injectables (`store`, `csv_store`, `base_year`, `year`, `fileyear`), helper functions (`store_table_list`, `find_table_in_store`, `find_cities_in_store`), and ~20 orca table definitions (`buildings`, `zones`, `households`, `jobs`, `parcels`, `parcel_zoning`, `fazes`, `cities`, `counties`, `alldata`, `growth_centers`, `parcels_geos`, `targets`, `controls`, `control_hcts`, etc.)
   - **Indicator section** (from indicators.py): `datasets` dict, `geography_alias`, `table_alias` dicts, `create_csv()`, `create_tab()` helper functions, `ind_table_dic` and `create_tables()` function, orca injectables (`years_to_run`, `is_annual`), orca steps (`add_new_datasets`, `compute_indicators`, `compute_datasets`)
   - **New**: `indicators_outdir()` helper — derives output subfolder from store filename, replacing "results" with "indicators" (e.g., `results_alloc_20260330.h5` → `output/indicators_alloc_20260330/`)
   - Remove hardcoded `DATA_HOME` env var; all paths come from the merged config
   - Remove `settings_file` and `settings` injectables (config will be injected by the CLI)
   - `csv_store` injectable uses `settings['data_dir'] + '/csv_store'`

## CLI

5. **Implement `psrc_urbansim/cli/run_indicators.py`** — following run_allocation.py pattern
   - `add_run_args(parser)` — adds `-c`/`--configs_dir` argument
   - `run_indicators(configs_dir)`:
     - Load `settings.yaml` from configs_dir
     - Load `settings_indicators.yaml` from configs_dir, deep_merge into config
     - Set `orca.settings` and inject `settings`
     - Open HDF5 store from `config['output_dir'] / config['store']` and inject as `store`
     - Inject `configs_dir`
     - Import `psrc_urbansim.variables` (top-level) for computed columns
     - Import `psrc_urbansim.mod.indicators` (top-level) to register orca tables and steps
     - Compute `years_to_run` from config via `ind_mod.years_to_run(config)`
     - Run orca with steps: `['add_new_datasets', 'compute_indicators', 'compute_datasets']` over `years_to_run`
     - Call `ind_mod.create_tables(outdir)` where `outdir = ind_mod.indicators_outdir(config)`
     - Close the HDF5 store
   - `run(args)` — calls `run_indicators(args.configs_dir)`
   - `if __name__ == "__main__"` block

6. **Update `psrc_urbansim/cli/__init__.py`** — add `from . import run_indicators`

7. **Update `psrc_urbansim/cli/main.py`** — register `run_indicators` subcommand

## Cleanup

8. **Delete legacy files**
   - Delete `indicators/indicators.py` (logic moved to `psrc_urbansim/mod/indicators.py`)
   - Delete `indicators/data.py` (merged into `psrc_urbansim/mod/indicators.py`)
   - Delete `indicators/data_DM.py` (experimental/unused)
   - Delete `indicators/indicators_settings.yaml` (moved to configs)
   - Delete `indicators/csv_store/` directory (moved to data/csv_store)
   - Delete `indicators/__init__.py` (no longer needed after data.py merge)
   - Keep `indicators/README.md` (updated to reflect new CLI usage)
   - Keep `indicators/run_indicators_AY.py`, `indicators/export_variables.py`

## Relevant files

### New/modified
- `psrc_urbansim/cli/run_indicators.py` — new CLI command
- `psrc_urbansim/cli/main.py` — registered `run_indicators` subcommand
- `psrc_urbansim/cli/__init__.py` — added `run_indicators` import
- `psrc_urbansim/mod/indicators.py` — merged module with data sources + orca steps/helpers
- `configs/settings_indicators.yaml` — new config file (adapted from indicators_settings.yaml)
- `data/csv_store/` — new location for CSV lookup files
- `indicators/README.md` — updated usage instructions

### Deleted
- `indicators/indicators.py` — logic moved to `psrc_urbansim/mod/indicators.py`
- `indicators/data.py` — merged into `psrc_urbansim/mod/indicators.py`
- `indicators/__init__.py` — no longer needed
- `indicators/data_DM.py` — experimental/unused
- `indicators/indicators_settings.yaml` — moved to `configs/`
- `indicators/csv_store/` — moved to `data/csv_store/`

## Verification

All checks passed:
1. `psrc_urbansim run_indicators -c "configs"` produces 78 CSV/tab indicator files in `output/indicators_alloc_20260330/`
2. HDF5 store opens correctly using the output_dir-relative path
3. CSV lookup files load correctly from `data/csv_store/`
4. `psrc_urbansim --help` shows all three subcommands
5. `psrc_urbansim run_indicators --help` shows the `-c` argument
6. All 7 years processed (2023, 2025, 2030, 2035, 2040, 2044, 2050)

## Decisions

- Config merge: settings.yaml + settings_indicators.yaml (deep_merge), matching run_allocation pattern
- DATA_HOME: eliminated; all paths from merged config
- Store path: resolved as `output_dir / store_filename`
- csv_store: physically moved to `data/csv_store/`, path derived from `data_dir`
- Config name: `settings_indicators.yaml` to match `settings_allocation.yaml` convention
- data.py merged into `psrc_urbansim/mod/indicators.py` (changed from original plan to keep it separate; merging avoids sys.path issues and simplifies imports)
- psrc_urbansim.variables import retained for computed columns
- Output subfolder: CSV/tab files saved to `output/<store_stem>/` where "results" in the store name is replaced with "indicators" (e.g., `output/indicators_alloc_20260330/`)
- Python environment: uses `.venv` (not Anaconda)
- Legacy indicators.py, data_DM.py, indicators_settings.yaml, csv_store/ deleted after CLI works
