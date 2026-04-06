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

logging.basicConfig(level=logging.INFO)

timestr = pd.Timestamp.now().strftime("%Y%m%d")


def add_run_args(parser, multiprocess=True):
    """
    Run command args
    """
    parser.add_argument(
        "-c",
        "--configs_dir",
        type=str,
        metavar="PATH",
        help="path to configs dir",
    )


def run_indicators(configs_dir):
    """Run indicators from simulation results."""
    # Import here to avoid registering indicator Orca tables when this module
    # is imported only for CLI argument wiring.
    import psrc_urbansim.mod.indicators as ind_mod

    # Load and merge settings
    with open(Path(f"{configs_dir}/settings.yaml")) as f:
        config = yaml.safe_load(f)
    with open(Path(f"{configs_dir}/settings_indicators.yaml")) as f:
        deep_merge(yaml.safe_load(f), config)

    # Inject settings into orca
    orca.settings = config
    orca.injectable('settings', cache=True)(lambda: config)

    # Open results HDF5 store
    store_path = Path(config['output_dir']) / config['store']
    my_store = pd.HDFStore(str(store_path), mode='r')
    orca.add_injectable('store', my_store, cache=True)
    orca.add_injectable('configs_dir', configs_dir, cache=True)

    # Compute years to run
    years = ind_mod.years_to_run(config)

    # Run indicator orca steps
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
