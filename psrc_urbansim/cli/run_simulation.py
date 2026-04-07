import os
#import urbansim_defaults.models
import psrc_urbansim.src.models
import psrc_urbansim.workplace_models
import psrc_urbansim.src.developer_models
import orca
import logging
import pandas as pd
import psrc_urbansim.vars.variables_interactions
import psrc_urbansim.vars.variables_generic
from urbansim.utils import yamlio
from urbansim.utils import misc
import pandas as pd
from urbansim.utils import yamlio
from urbansim.utils import misc
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
     outfile = output_run_dir / ("results_sim_" + timestr + ".h5")
     outfile.parent.mkdir(parents=True, exist_ok=True)
     if os.path.exists(outfile):
          os.remove(outfile)
     return str(outfile)

# @orca.injectable('settings', cache=True)
# def settings(configs_dir):
#     settings = yaml.safe_load(open(Path(f"{configs_dir}/settings.yaml")))
#     orca.settings = settings
#     return settings


# @orca.injectable('simfile')
# def simfile():
#      return "simresult20220222test.h5"

# # remove results file if exists
# outfile = simfile()
# if os.path.exists(outfile):
#      os.remove(outfile)


def run_model(configs_dir):
     config = yaml.safe_load(open(Path(f"{configs_dir}/settings.yaml")))
     output_run_dir = unique_output_dir(Path(config['output_dir']) / ("run_sim_" + timestr + "_" + get_username()))

     # Set up logging into the output directory
     log_file = None
     if log_into_file:
          output_run_dir.mkdir(parents=True, exist_ok=True)

          if debug:
               log_file = str(output_run_dir / ("log_simulation_debug_" + timestr + ".txt"))
          else:
               log_file = str(output_run_dir / ("log_simulation_" + timestr + ".txt"))
     logging.basicConfig(level = loglevel, filename = log_file, format = FORMAT, datefmt = '%H:%M:%S', filemode = 'w', force = True)

     orca.settings = config
     orca.injectable('settings', cache=True)(lambda: config)
     my_store = pd.HDFStore(Path(config['data_dir']) / config["store"],mode='r')
     orca.add_injectable('store', my_store, cache=True)
     orca.add_injectable('configs_dir', configs_dir, cache=True)
     #config = settings(configs_dir)
     #os.environ['DATA_HOME'] = config['data_dir']
     orca.add_injectable('output_run_dir', output_run_dir, cache=True)
     outfile = simfile(config, output_run_dir)
     
     orca.run([
     # Misc
     #######
     #"update_misc_building_columns",
     "update_household_previous_building_id",
     "update_buildings_lag1",
     
     # REPM
     #######
     "repmres_simulate",          # residential REPM
     "repmnr_simulate",           # non-residential REPM
     
     # Developer 
     #######
     #"process_mpds",
     "proforma_feasibility",
     "developer_picker",

    # Household models
    #######
     "households_transition",     # households transition
     "households_relocation",     # households relocation model
     "hlcm_simulate",
    
    # Employment models
    #######
     "jobs_transition",           # jobs transition
     "jobs_relocation",
     "elcm_simulate",             # employment location choice
     "update_persons_jobs",
     "governmental_jobs_scaling",
    
    # Workplace models
    #######    
    "wahcm_simulate",
    "wplcm_simulate"],
    iter_vars= range(config['start_year'], config['end_year'] + 1),
    data_out=outfile, 
    out_base_tables=tables_in_base_year(config),
    compress=True, out_run_local=True)


     # Copy settings.yaml to the output folder
     settings_src = Path(configs_dir) / "settings.yaml"
     settings_dst = output_run_dir / "settings.yaml"
     shutil.copy2(settings_src, settings_dst)
     logging.info('Copied settings.yaml to %s', settings_dst)

     logging.info('Simulation finished')

def run(args):
    run_model(args.configs_dir)
    sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()
    sys.exit(run(args))