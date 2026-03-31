import os
import time
#import urbansim_defaults.models
import psrc_urbansim.models
import psrc_urbansim.workplace_models
import psrc_urbansim.developer_models
import orca
import yaml
import logging
import pandas as pd
import psrc_urbansim.vars.variables_interactions
import psrc_urbansim.vars.variables_generic
from urbansim.utils import yamlio
from urbansim.utils import misc

debug = False # switch this to True for detailed debug messages
log_into_file = True # should log messages go into a file (True) or be printed into the console (False)

FORMAT = '%(asctime)s %(name)s %(levelname)s %(message)s'
timestr = time.strftime("%Y%m%d")
log_file = None

if debug:
     if log_into_file:
          log_file = "log_simulation_debug_" + timestr + ".txt"
     loglevel = logging.DEBUG
else:
     if log_into_file:
          log_file = "log_simulation_" + timestr + ".txt"
     loglevel = logging.INFO   

logging.basicConfig(level = loglevel, filename = log_file, format = FORMAT, datefmt = '%H:%M:%S', filemode = 'w')

@orca.injectable('simfile')
def simfile():
     return "results_sim_" + timestr + ".h5"

@orca.injectable('settings', cache=True)
def settings():
     # load settings.yaml
     with open(os.path.join(misc.configs_dir(), "settings.yaml")) as f:
          settings = yaml.load(f, Loader=yaml.FullLoader)
          # monkey patch on the settings object since it's pretty global
          # but will also be available as injectable
          orca.settings = settings
          return settings

# remove results file if exists
outfile = simfile()
if os.path.exists(outfile):
     os.remove(outfile)
     
settings = settings()

# get tables from input file
def tables_in_base_year():
     h5store = pd.HDFStore(os.path.join(misc.data_dir(), settings['store']), mode="r")
     store_table_names = orca.get_injectable('store_table_names_dict')
     return [t for t in orca.list_tables() if t in h5store or store_table_names.get(t, "UnknownTable") in h5store]

orca.run([
     'start_year',
     
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
    'update_persons_jobs',
    "governmental_jobs_scaling",
    
    # Workplace models
    #######    
    "wahcm_simulate",
    "wplcm_simulate",
    
    'end_year'

], #iter_vars= range(2015,2051), 
         iter_vars= range(2024,2051), 
         data_out=outfile, out_base_tables=tables_in_base_year(),
   compress=True, out_run_local=True)


logging.info('Simulation finished')
