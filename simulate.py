import os
import psrc_urbansim.models
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
     return "simresult20170404.h5"

# remove results file if exists
outfile = simfile()
if os.path.exists(outfile):
     os.remove(outfile)
     
# get tables from input file
def tables_in_base_year():
     h5store = pd.HDFStore(os.path.join(misc.data_dir(),  
                         yamlio.yaml_to_dict(str_or_buffer=os.path.join(misc.configs_dir(), 
                                                            "settings.yaml"))['store']), mode="r")
     return [t for t in orca.list_tables() if t in h5store]

orca.run([
#    "add_lag1_tables",
    "proforma_feasibility",
    "residential_developer",      # residential proforma model
    "non_residential_developer",  # non-residential proforma model
    
    "repmres_simulate",          # residential REPM
    "repmnr_simulate",            # non-residential REPM

    "households_transition",     # households transition
    "households_relocation",     # households relocation model
    "hlcm_simulate",            # households location choice

    "jobs_transition",           # jobs transition
    "jobs_relocation",           # jobs relocation model
    "elcm_simulate",             # employment location choice
    "governmental_jobs_scaling"

], iter_vars=[2015, 2016], data_out=outfile, out_base_tables=tables_in_base_year(),
   compress=True, out_run_local=True)

logging.info('Simulation finished')
