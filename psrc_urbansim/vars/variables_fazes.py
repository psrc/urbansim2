import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# FAZES VARIABLES
#####################

@orca.column('fazes', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(fazes, households):
    return households.persons.groupby(households.faz_id).size().\
           reindex(fazes.index).fillna(0)

@orca.column('fazes', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(fazes, jobs):
    return jobs.sector_id.groupby(jobs.faz_id).size().\
           reindex(fazes.index).fillna(0)
