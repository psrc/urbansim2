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
     return "allocresult20190610.h5"

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
     return [2017] + range(2020, 2050)

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
     return [t for t in orca.list_tables() if t in h5store or store_table_names.get(t, None) in h5store]


# models for control years
orca.run([
     # REPM
     #"repmres_simulate",          # residential REPM
     #"repmnr_simulate",           # non-residential REPM
     
     # Transition
     # (Must run hh/job transition models first in order to 
     # determine the demand for the developer model.)
     "households_transition_alloc",     # 
     "jobs_transition_alloc",           #
     
     # Developer 
     #"proforma_feasibility_alloc",
     #"developer_picker_alloc",
     
     # Misc
     "update_misc_building_columns",
     "update_household_previous_building_id",
     "update_buildings_lag1",
     
     # Relocate and place households
     #"households_relocation_alloc",
     #"hlcm_simulate_alloc",
    
    # Relocate and place jobs
     "jobs_relocation_alloc",     
    # "elcm_simulate_alloc",             # ELCM
    "governmental_jobs_scaling_alloc",
    #"wahcm_simulate",
    #"wplcm_simulate",
    #"clear_cache"
], iter_vars=[2017, 2018], data_out=outfile, out_base_tables=tables_in_base_year(),
   compress=True, out_run_local=True)


logging.info('Allocation finished')


