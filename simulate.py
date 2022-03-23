import os
#import urbansim_defaults.models
import psrc_urbansim.models
import psrc_urbansim.workplace_models
import psrc_urbansim.developer_models
import orca
import logging
import pandas as pd
import psrc_urbansim.vars.variables_interactions
import psrc_urbansim.vars.variables_generic
from urbansim.utils import yamlio
from urbansim.utils import misc

logging.basicConfig(level=logging.INFO)


@orca.injectable('simfile')
def simfile():
     return "simresult20220222test.h5"

# remove results file if exists
outfile = simfile()
if os.path.exists(outfile):
     os.remove(outfile)
     
# get tables from input file
def tables_in_base_year():
     h5store = pd.HDFStore(os.path.join(misc.data_dir(),  
                         yamlio.yaml_to_dict(str_or_buffer=os.path.join(misc.configs_dir(), 
                                                            "settings.yaml"))['store']), mode="r")
     store_table_names = orca.get_injectable('store_table_names_dict')
     return [t for t in orca.list_tables() if t in h5store or store_table_names.get(t, ' ') in h5store]


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
    'update_persons_jobs',
    "governmental_jobs_scaling",
    
    # Workplace models
    #######    
    "wahcm_simulate",
    "wplcm_simulate",

], #iter_vars= range(2015,2051), 
         iter_vars= range(2019,2020), 
         data_out=outfile, out_base_tables=tables_in_base_year(),
   compress=True, out_run_local=True)


logging.info('Simulation finished')
