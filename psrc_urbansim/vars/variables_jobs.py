import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# JOBS VARIABLES (in alphabetic order)
#####################

@orca.column('jobs', 'faz_id', cache=True)
def faz_id(jobs, zones):
    return misc.reindex(zones.faz_id, jobs.zone_id)

@orca.column('jobs', 'grid_id', cache=True)
def grid_id(jobs, parcels):
    return misc.reindex(parcels.grid_id, jobs.parcel_id)

@orca.column('jobs', 'parcel_id', cache=True, cache_scope='step')
def parcel_id(jobs, buildings):
    return misc.reindex(buildings.parcel_id, jobs.building_id)

@orca.column('jobs', 'tractcity_id', cache=True)
def tractcity_id(jobs, parcels):
    return misc.reindex(parcels.tractcity_id, jobs.parcel_id)

@orca.column('jobs', 'zone_id', cache=True)
def zone_id(jobs, buildings):
    return misc.reindex(buildings.zone_id, jobs.building_id)



