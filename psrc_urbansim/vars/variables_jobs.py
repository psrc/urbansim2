import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# JOBS VARIABLES (in alphabetic order)
#####################

def is_in_sector_group(group_name, jobs, employment_sectors, employment_sector_groups, employment_sector_group_definitions):
    group = employment_sector_groups.index[employment_sector_groups['name'] == group_name]
    idx = [jobs.sector_id.values, group[0]*np.ones(jobs.sector_id.size)]
    midx = pd.MultiIndex.from_arrays(idx, names=('sector_id', 'group_id'))
    return np.logical_not(np.isnan(employment_sector_group_definitions.dummy[midx])).reset_index("group_id").dummy    

@orca.column('jobs', 'faz_id', cache=True)
def faz_id(jobs, zones):
    return misc.reindex(zones.faz_id, jobs.zone_id)

@orca.column('jobs', 'grid_id', cache=True)
def grid_id(jobs, parcels):
    return misc.reindex(parcels.grid_id, jobs.parcel_id)

@orca.column('jobs', 'is_in_sector_group_retail', cache=True)
def is_in_sector_group_retail(jobs, employment_sectors, employment_sector_groups, employment_sector_group_definitions):
    return is_in_sector_group("retail", jobs, employment_sectors, employment_sector_groups, employment_sector_group_definitions)

@orca.column('jobs', 'parcel_id', cache=True, cache_scope='step')
def parcel_id(jobs, buildings):
    return misc.reindex(buildings.parcel_id, jobs.building_id)

@orca.column('jobs', 'tractcity_id', cache=True)
def tractcity_id(jobs, parcels):
    return misc.reindex(parcels.tractcity_id, jobs.parcel_id)

@orca.column('jobs', 'zone_id', cache=True)
def zone_id(jobs, buildings):
    return misc.reindex(buildings.zone_id, jobs.building_id)



