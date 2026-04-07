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
import shutil
import yaml
from pathlib import Path
from psrc_urbansim.src.cli_utils import timestr, get_username, unique_output_dir, add_run_args, tables_in_base_year

debug = False # switch this to True for detailed debug messages
log_into_file = True # should log messages go into a file (True) or be printed into the console (False)

FORMAT = '%(asctime)s %(name)s %(levelname)s %(message)s'

if debug:
     loglevel = logging.DEBUG
else:
     loglevel = logging.INFO


@orca.injectable('simfile')
def simfile(config, output_run_dir):
     outfile = output_run_dir / ("results_alloc_" + timestr + ".h5")
     outfile.parent.mkdir(parents=True, exist_ok=True)
     if os.path.exists(outfile):
          os.remove(outfile)
     return str(outfile)


def run_allocation(configs_dir):
     # merge settings.yaml with settings_allocation.yaml
     with open(Path(f"{configs_dir}/settings.yaml")) as f:
          config = yaml.safe_load(f)
          with open(Path(f"{configs_dir}/settings_allocation.yaml")) as af:
               deep_merge(yaml.safe_load(af), config)
     output_run_dir = unique_output_dir(Path(config['output_dir']) / ("run_alloc_" + timestr + "_" + get_username()))

     # Set up logging into the output directory
     log_file = None
     if log_into_file:
          output_run_dir.mkdir(parents=True, exist_ok=True)
          if debug:
               log_file = str(output_run_dir / ("log_allocation_debug_" + timestr + ".txt"))
          else:
               log_file = str(output_run_dir / ("log_allocation_" + timestr + ".txt"))
     logging.basicConfig(level = loglevel, filename = log_file, format = FORMAT, datefmt = '%H:%M:%S', filemode = 'w', force = True)

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

     orca.add_injectable('output_run_dir', output_run_dir, cache=True)
     outfile = simfile(config, output_run_dir)

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

     # Copy settings files to the output folder
     for settings_file in ["settings.yaml", "settings_allocation.yaml"]:
          src = Path(configs_dir) / settings_file
          if src.exists():
               shutil.copy2(src, output_run_dir / settings_file)
     logging.info('Copied settings files to %s', output_run_dir)

     logging.info('Allocation finished')

def run(args):
    run_allocation(args.configs_dir)
    sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()
    sys.exit(run(args))