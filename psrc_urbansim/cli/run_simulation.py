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
import yaml
from pathlib import Path

#os.environ['DATA_HOME'] = "C:\\Stefan\\urbansim_update_test\\urbansim2"

logging.basicConfig(level=logging.INFO)


@orca.injectable('simfile')
def simfile(config):
     outfile = Path(config['output_dir']) / config['output_store']
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

     
# get tables from input file
def tables_in_base_year(config):
     h5store_path = Path(config['data_dir']) / config['store']
     h5store = pd.HDFStore(h5store_path, mode="r")
     store_table_names = orca.get_injectable('store_table_names_dict')
     return [t for t in orca.list_tables() if t in h5store or store_table_names.get(t, ' ') in h5store]

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

def run_model(configs_dir):
     config = yaml.safe_load(open(Path(f"{configs_dir}/settings.yaml")))
     orca.settings = config
     orca.injectable('settings', cache=True)(lambda: config)
     my_store = pd.HDFStore(Path(config['data_dir']) / config["store"],mode='r')
     orca.add_injectable('store', my_store, cache=True)
     orca.add_injectable('configs_dir', configs_dir, cache=True)
     #config = settings(configs_dir)
     #os.environ['DATA_HOME'] = config['data_dir']
     outfile = simfile(config)
     
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


     logging.info('Simulation finished')

def run(args):
    run_model(args.configs_dir)
    sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()
    sys.exit(run(args))