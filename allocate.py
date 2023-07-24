# #############################################
# Runs the PSRC allocation mode of urbansim.
# #############################################
# Set the name of the input file as well as various other allocation-related settings 
#    in configs/settings_allocation.yaml
# Set generic (non-allocation related) settings in configs/settings.yaml
# Set the base year in psrc_urbansim/datasources.py
# #############################################

import os
import time
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
from psrc_urbansim.utils import deep_merge

debug = False # switch this to True for detailed debug messages
log_into_file = True # should log messages go into a file (True) or be printed into the console (False)

FORMAT = '%(asctime)s %(name)s %(levelname)s %(message)s'
timestr = time.strftime("%Y%m%d")
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
def simfile():
     return "results_alloc_BY2018_20230426.h5"

@orca.injectable('settings', cache=True)
def settings():
     # merges settings.yaml with settings_allocation.yaml 
     with open(os.path.join(misc.configs_dir(), "settings.yaml")) as f:
          settings = yaml.load(f, Loader=yaml.FullLoader)
          with open(os.path.join(misc.configs_dir(), "settings_allocation.yaml")) as af:
               deep_merge(yaml.load(af, Loader=yaml.FullLoader), settings)
          # monkey patch on the settings object since it's pretty global
          # but will also be available as injectable
          orca.settings = settings
          return settings

@orca.injectable('control_years', cache=True)
def control_years():
     return list(range(2020, 2041, 5)) + [2044, 2050]

@orca.injectable('isCY', cache=False)
def isCY(year, control_years):
     return year in control_years

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


# models for control years
orca.run([
     'start_year',
     
     'households_events_model',
     'households_zone_events_model',
     
     ## Misc
     ########
     #"update_household_previous_building_id",
     #"update_buildings_lag1",
     
     ## REPM
     ########
     #"repmres_simulate",          # residential REPM
     #"repmnr_simulate",           # non-residential REPM
     
     ## Transition
     ########
     ## (Must run hh/job transition models first in order to 
     ## determine the demand for the developer model.)
     #"households_transition_alloc",     # 
     #"jobs_transition_alloc",           #
     
     ## Developer 
     ########
     #"process_mpds",
     #"cap_residential_development",
     #"cap_nonresidential_development",     
     #"proforma_feasibility_alloc",
     #"developer_picker_alloc",
     
     ## Density boosts
     ########
     #"boost_residential_density",
     #"boost_nonresidential_density",
          
     ## Relocate and place households
     ########
     #"households_relocation_alloc",
     #"hlcm_simulate_alloc",
    
    ## Relocate and place jobs
    ########
     #"jobs_relocation_alloc",     
     #"elcm_simulate_alloc",             # ELCM
     #"governmental_jobs_scaling_alloc",
     #"update_persons_jobs",

    ## Scaling of unplaced HHs and jobs in control years
    ########
    #'scaling_unplaced_households',
    #'scaling_unplaced_jobs',
    
    ## Workplace models
    ########
    #"wahcm_simulate_alloc",
    #"wplcm_simulate",
    
    ## Cleanup city_id
    ########
    #"delete_subreg_geo_from_households",
    #"delete_subreg_geo_from_jobs",
    
     'end_year'

], iter_vars=range(2019,2051), data_out=outfile, out_base_tables=tables_in_base_year(),
   compress=True, out_run_local=True)


logging.info('Allocation finished')


# TODO:
# =====
# - transition model samples agents regardless of the size of the group from which it's sampled
#   -> if no agents of a group present nothing is sampled and the results do not match CTs 
# - not all MPDs are contained in the mpds dataset
# - no agent events model implemented
