import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# TRACT-CITY VARIABLES
#####################

@orca.column('tractcity', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(tractcity, parcels):
    return parcels.number_of_households.groupby(parcels.tractcity_id).sum().\
           reindex(tractcity.index).fillna(0)

@orca.column('tractcity', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(tractcity, parcels):
    return parcels.number_of_jobs.groupby(parcels.tractcity_id).sum().\
           reindex(tractcity.index).fillna(0)
