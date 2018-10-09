import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# jobs_for_estimation VARIABLES (in alphabetic order)
#####################

def is_in_sector_group(group_name, jobs_for_estimation, employment_sectors, employment_sector_groups, employment_sector_group_definitions):
    group = employment_sector_groups.index[employment_sector_groups['name'] == group_name]
    idx = [jobs_for_estimation.sector_id.values, group[0]*np.ones(jobs_for_estimation.sector_id.size)]
    midx = pd.MultiIndex.from_arrays(idx, names=('sector_id', 'group_id'))
    res = np.logical_not(np.isnan(employment_sector_group_definitions.dummy[midx])).reset_index("group_id").dummy
    res.index = jobs_for_estimation.index
    return res

@orca.column('jobs_for_estimation', 'faz_id', cache=True)
def faz_id(jobs_for_estimation, zones):
    return misc.reindex(zones.faz_id, jobs_for_estimation.zone_id)

@orca.column('jobs_for_estimation', 'grid_id', cache=True)
def grid_id(jobs_for_estimation, parcels):
    return misc.reindex(parcels.grid_id, jobs_for_estimation.parcel_id)

@orca.column('jobs_for_estimation', 'is_in_sector_group_retail', cache=True)
def is_in_sector_group_retail(jobs_for_estimation, employment_sectors, employment_sector_groups, employment_sector_group_definitions):
    return is_in_sector_group("retail", jobs_for_estimation, employment_sectors, employment_sector_groups, employment_sector_group_definitions)

@orca.column('jobs_for_estimation', 'parcel_id', cache=True, cache_scope='step')
def parcel_id(jobs_for_estimation, buildings):
    return misc.reindex(buildings.parcel_id, jobs_for_estimation.building_id)

@orca.column('jobs_for_estimation', 'tractcity_id', cache=True)
def tractcity_id(jobs_for_estimation, parcels):
    return misc.reindex(parcels.tractcity_id, jobs_for_estimation.parcel_id)

@orca.column('jobs_for_estimation', 'zone_id', cache=True)
def zone_id(jobs_for_estimation, buildings):
    return misc.reindex(buildings.zone_id, jobs_for_estimation.building_id)



