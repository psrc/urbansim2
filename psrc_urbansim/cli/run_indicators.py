import os
import sys
import orca
import logging
import pandas as pd
import psrc_urbansim.variables
from psrc_urbansim.src.utils import deep_merge
import argparse
import yaml
from pathlib import Path
from psrc_urbansim.src.cli_utils import timestr, add_run_args, find_latest_results

logging.basicConfig(level=logging.INFO)


def run_indicators(configs_dir):
    """Run indicators from simulation results."""
    # Import here to avoid registering indicator Orca tables when this module
    # is imported only for CLI argument wiring.
    import psrc_urbansim.mod.indicators as ind_mod

    # Load and merge settings
    with open(Path(f"{configs_dir}/settings.yaml")) as f:
        config = yaml.safe_load(f)
    with open(Path(f"{configs_dir}/settings_indicators.yaml")) as f:
        ind_config = yaml.safe_load(f) or {}
        deep_merge(ind_config, config)

    # Resolve the results store: use explicit 'store' from settings_indicators.yaml
    # if provided, otherwise auto-detect the most recent results h5.
    if 'store' in ind_config and ind_config['store']:
        store_path = Path(config['output_dir']) / ind_config['store']
        config['store'] = ind_config['store']
    else:
        store_path = Path(find_latest_results(config['output_dir']))
        # Set store in config so downstream code (e.g. indicators_outdir) can use it
        config['store'] = str(store_path.relative_to(config['output_dir']))

    logging.info('Using results store: %s', store_path)

    # Inject settings into orca
    orca.settings = config
    orca.injectable('settings', cache=True)(lambda: config)

    # Open results HDF5 store
    my_store = pd.HDFStore(str(store_path), mode='r')
    orca.add_injectable('store', my_store, cache=True)
    orca.add_injectable('configs_dir', configs_dir, cache=True)

    # Compute years to run
    years = ind_mod.years_to_run(config)

    # Validate years, then run indicator orca steps
    ind_mod.validate_years(config, my_store, years)
    orca.run(['add_new_datasets', 'compute_indicators', 'compute_datasets'],
             iter_vars=years)

    # Save collected indicator tables to disk
    outdir = ind_mod.indicators_outdir(config)
    ind_mod.create_tables(outdir)

    # Close the HDF5 store
    my_store.close()

    logging.info('Indicators finished')


def run(args):
    """Run indicators subcommand."""
    run_indicators(args.configs_dir)
    sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()
    sys.exit(run(args))
