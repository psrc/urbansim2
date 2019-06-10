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
     return "simresult20181008.h5"

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
    # Must run hh/job transition models first in order to 
    # determine the demand for the developer model.
     #"households_transition",     # TODO: check if it can handle subreg controls
     #"jobs_transition",           #
    "proforma_feasibility_CY", # has relaxed redevelopment filter
    "developer_picker_CY",     # runs DM for each subreg separately

    "update_household_previous_building_id",
    "update_buildings_lag1",
    "repmres_simulate",          # residential REPM
    "repmnr_simulate",          # non-residential REPM           

    "households_relocation",     # TODO: use no rates for CY
    "hlcm_simulate",

    "jobs_relocation",           # TODO: use no rates for CY
    'update_persons_jobs',          
    "elcm_simulate",             # employment location choice
    "governmental_jobs_scaling",
    "wahcm_simulate",
    "wplcm_simulate",
    #"clear_cache"
], iter_vars=[2015, 2016], data_out=outfile, out_base_tables=tables_in_base_year(),
   compress=True, out_run_local=True)


logging.info('Allocation finished')


