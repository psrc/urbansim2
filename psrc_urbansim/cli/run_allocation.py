import os
import psrc_urbansim.src.models
import psrc_urbansim.workplace_models
import psrc_urbansim.src.developer_models
import orca
import logging
import pandas as pd
import psrc_urbansim.vars.variables_interactions
import psrc_urbansim.vars.variables_generic
from psrc_urbansim.src.utils import deep_merge
from os import sys
import argparse
import yaml
from pathlib import Path

debug = False # switch this to True for detailed debug messages
log_into_file = True # should log messages go into a file (True) or be printed into the console (False)

FORMAT = '%(asctime)s %(name)s %(levelname)s %(message)s'
timestr = pd.Timestamp.now().strftime("%Y%m%d")
log_file = None

if debug:
     if log_into_file:
          log_file = "log_allocation_debug_" + timestr + ".txt"
     loglevel = logging.DEBUG
else:
     if log_into_file:
          log_file = "log_allocation_" + timestr + ".txt"
     loglevel = logging.INFO   

logging.basicConfig(level = loglevel, filename = log_file, format = FORMAT, datefmt = '%H:%M:%S', filemode = 'w')


@orca.injectable('simfile')
def simfile(config):
     outfile = Path(config['output_dir']) / ("results_alloc_" + timestr + ".h5")
     if os.path.exists(outfile):
          os.remove(outfile)
     return str(outfile)


def tables_in_base_year(config):
     h5store_path = Path(config['data_dir']) / config['store']
     h5store = pd.HDFStore(h5store_path, mode="r")
     store_table_names = orca.get_injectable('store_table_names_dict')
     return [t for t in orca.list_tables() if t in h5store or store_table_names.get(t, "UnknownTable") in h5store]


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

def run_allocation(configs_dir):
     # merge settings.yaml with settings_allocation.yaml
     with open(Path(f"{configs_dir}/settings.yaml")) as f:
          config = yaml.safe_load(f)
          with open(Path(f"{configs_dir}/settings_allocation.yaml")) as af:
               deep_merge(yaml.safe_load(af), config)

     orca.settings = config
     orca.injectable('settings', cache=True)(lambda: config)
     my_store = pd.HDFStore(Path(config['data_dir']) / config["store"], mode='r')
     orca.add_injectable('store', my_store, cache=True)
     orca.add_injectable('configs_dir', configs_dir, cache=True)

     @orca.injectable('control_years', cache=True)
     def control_years():
          return list(range(2020, 2041, 5)) + [2044, 2050]

     @orca.injectable('isCY', cache=False)
     def isCY(year, control_years):
          return year in control_years

     outfile = simfile(config)

     orca.run([
          'start_year',

          # Misc
          #######
          "update_household_previous_building_id",
          "update_buildings_lag1",
          "update_building_type_condo",

          # REPM
          #######
          "repmres_simulate",          # residential REPM
          "repmnr_simulate",           # non-residential REPM

          # Transition
          #######
          # (Must run hh/job transition models first in order to 
          # determine the demand for the developer model.)
          "households_transition_alloc",     # 
          "jobs_transition_alloc",           #

          # Developer 
          #######
          "process_mpds",
          #"cap_residential_development",
          #"cap_nonresidential_development",     
          "proforma_feasibility_alloc",
          "developer_picker_alloc",

          # Density boosts
          #######
          "boost_residential_density",
          "boost_nonresidential_density",

          # Events models
          #######
          'households_events_model',
          'households_zone_control_hct_events_model',

          # Transition again, in case events model changed the counts
          #######     
          "households_transition_alloc",     # 
          "jobs_transition_alloc",           #

          # Relocate and place households
          #######
          "households_relocation_alloc",
          "hlcm_simulate_alloc",

          # Relocate and place jobs
          #######
          "jobs_relocation_alloc",     
          "elcm_simulate_alloc",             # ELCM
          "governmental_jobs_scaling_alloc",
          "update_persons_jobs",

          # Scaling of unplaced HHs and jobs in control years
          #######
          'scaling_unplaced_households',
          'scaling_unplaced_jobs',

          # Workplace models
          #######
          "wahcm_simulate_alloc",
          "wplcm_simulate",

          # Cleanup subreg_id
          #######
          "delete_subreg_geo_from_households",
          "delete_subreg_geo_from_jobs",

          'end_year'

     ], iter_vars=range(config['start_year'], config['end_year'] + 1),
        data_out=outfile, out_base_tables=tables_in_base_year(config),
        compress=True, out_run_local=True)

     logging.info('Allocation finished')

def run(args):
    run_allocation(args.configs_dir)
    sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()
    sys.exit(run(args))