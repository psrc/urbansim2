import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils


#####################
# ZONES VARIABLES
#####################

@orca.column('zones', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(zones, households):
    return households.persons.groupby(households.zone_id).size().\
           reindex(zones.index).fillna(0)

@orca.column('zones', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(zones, jobs):
    return jobs.sector_id.groupby(jobs.zone_id).size().\
           reindex(zones.index).fillna(0)
