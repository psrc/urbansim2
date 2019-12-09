import os
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

logging.basicConfig(level=logging.INFO)


@orca.injectable('simfile')
def simfile():
     return "results_alloc_stc_20191209.h5"

@orca.injectable('settings', cache=True)
def settings():
     # merges settings.yaml with settings_allocation.yaml 
     with open(os.path.join(misc.configs_dir(), "settings.yaml")) as f:
          settings = yaml.load(f)
          with open(os.path.join(misc.configs_dir(), "settings_allocation.yaml")) as af:
               deep_merge(yaml.load(af), settings)
          # monkey patch on the settings object since it's pretty global
          # but will also be available as injectable
          orca.settings = settings
          return settings

@orca.injectable('control_years', cache=True)
def control_years():
     return [2017] + range(2020, 2051, 5)

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
     "proforma_feasibility_alloc",
     "developer_picker_alloc",
     
     # Misc
     #######
     "update_misc_building_columns",
     "update_household_previous_building_id",
     "update_buildings_lag1",
     "update_persons_jobs",
     
     # Relocate and place households
     #######
     "households_relocation_alloc",
     "hlcm_simulate_alloc",
    
    # Relocate and place jobs
    #######
     "jobs_relocation_alloc",     
     "elcm_simulate_alloc",             # ELCM
     "governmental_jobs_scaling_alloc",

    # Scaling of unplaced HHs and jobs in control years
    #######
    'scaling_unplaced_households',
    'scaling_unplaced_jobs',
    
    # Workplace models
    #######
    "wahcm_simulate_alloc",
    "wplcm_simulate",
    
    # Cleanup city_id
    #######
    "delete_subreg_geo_from_households",
    "delete_subreg_geo_from_jobs"

], iter_vars=range(2015,2051), data_out=outfile, out_base_tables=tables_in_base_year(),
   compress=True, out_run_local=True)


logging.info('Allocation finished')


