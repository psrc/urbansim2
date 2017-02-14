import os
import psrc_urbansim.models
import orca
import logging

logging.basicConfig(level=logging.INFO)


@orca.injectable('simfile')
def simfile():
     return "simresult.h5"

# remove results file if exists
outfile = simfile()
if os.path.exists(outfile):
     os.remove(outfile)

orca.run([
#    "add_lag1_tables",
#    "repmres_simulate",           # residential REPM
#    "repmnr_simulate",            # non-residential REPM

    "households_transition",     # households transition
    "households_relocation",     # households relocation model
#    "hlcm_simulate",            # households location choice

#    "jobs_transition",           # jobs transition
#    "jobs_relocation",           # jobs relocation model
#    "elcm_simulate",             # employment location choice
#    "governmental_jobs_scaling"

], iter_vars=[2015, 2016], data_out=outfile)

logging.info('Simulation finished')