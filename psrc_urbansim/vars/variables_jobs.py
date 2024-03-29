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
    idx = [jobs.sector_id.values, group[0]*np.ones(jobs.sector_id.size, dtype = "int32")]
    midx = pd.MultiIndex.from_arrays(idx, names=('sector_id', 'group_id'))
    res = pd.Series(midx.isin(employment_sector_group_definitions.index), index = jobs.index)
    res.index = jobs.index
    return res

@orca.column('jobs', 'city_id', cache=True, cache_scope='step')
def city_id(jobs, parcels):
    if "city_id" in jobs.local_columns:
        # this hack is needed for allocation mode, since orca 
        # gives priority to computed columns instead of local columns        
        return jobs.local.city_id 
    return misc.reindex(parcels.city_id, jobs.parcel_id)

@orca.column('jobs', 'county_id', cache=True, cache_scope='step')
def county_id(jobs, parcels):
    if "county_id" in jobs.local_columns:    
        return jobs.local.county_id 
    return misc.reindex(parcels.county_id, jobs.parcel_id)

@orca.column('jobs', 'subreg_id', cache=True, cache_scope='step')
def subreg_id(jobs, parcels):
    if "subreg_id" in jobs.local_columns:
        # this hack is needed for allocation mode, since orca 
        # gives priority to computed columns instead of local columns        
        return jobs.local.subreg_id 
    return misc.reindex(parcels.subreg_id, jobs.parcel_id)	


@orca.column('jobs', 'district_id', cache=True, cache_scope='step')
def district_id(jobs, zones):
    return misc.reindex(zones.district_id, jobs.job_zone_id)

@orca.column('jobs', 'faz_id', cache=True, cache_scope='step')
def faz_id(jobs, zones):
    return misc.reindex(zones.faz_id, jobs.job_zone_id)

@orca.column('jobs', 'growth_center_id', cache=True, cache_scope='step')
def growth_center_id(jobs, parcels, parcels_geos):
    if "growth_center_id" in parcels.columns:
        return misc.reindex(parcels.growth_center_id, jobs.parcel_id)	
    return misc.reindex(parcels_geos.growth_center_id, jobs.parcel_id)	

@orca.column('jobs', 'grid_id', cache=True, cache_scope='step')
def grid_id(jobs, parcels):
    return misc.reindex(parcels.grid_id, jobs.parcel_id)

@orca.column('jobs', 'is_in_sector_group_basic', cache=True)
def is_in_sector_group_retail(jobs, employment_sectors, employment_sector_groups, employment_sector_group_definitions):
    return is_in_sector_group("basic", jobs, employment_sectors, employment_sector_groups, employment_sector_group_definitions)

@orca.column('jobs', 'is_in_sector_group_retail', cache=True)
def is_in_sector_group_retail(jobs, employment_sectors, employment_sector_groups, employment_sector_group_definitions):
    return is_in_sector_group("retail", jobs, employment_sectors, employment_sector_groups, employment_sector_group_definitions)

@orca.column('jobs', 'is_in_sector_group_edu', cache=True)
def is_in_sector_group_retail(jobs, employment_sectors, employment_sector_groups, employment_sector_group_definitions):
    return is_in_sector_group("edu", jobs, employment_sectors, employment_sector_groups, employment_sector_group_definitions)

@orca.column('jobs', 'number_of_jobs', cache=False, cache_scope='step')
def number_of_jobs(jobs):
    return pd.Series(np.ones(len(jobs)), index=jobs.index)

@orca.column('jobs', 'parcel_id', cache=True, cache_scope='step')
def parcel_id(jobs, buildings):
    return misc.reindex(buildings.parcel_id, jobs.building_id)

@orca.column('jobs', 'tractcity_id', cache=True, cache_scope='step')
def tractcity_id(jobs, parcels):
    return misc.reindex(parcels.tractcity_id, jobs.parcel_id)

@orca.column('jobs', 'job_zone_id', cache=True, cache_scope='step')
def job_zone_id(jobs, buildings):
    return misc.reindex(buildings.zone_id, jobs.building_id)

@orca.column('jobs', 'twa_logsum_hbw_1', cache=True, cache_scope='iteration')
def twa_logsum_hbw_1(jobs, zones):
    return misc.reindex(zones.trip_weighted_average_logsum_hbw_am_income_1, jobs.job_zone_id)

@orca.column('jobs', 'twa_logsum_hbw_2', cache=True, cache_scope='iteration')
def twa_logsum_hbw_2(jobs, zones):
    return misc.reindex(zones.trip_weighted_average_logsum_hbw_am_income_2, jobs.job_zone_id)

@orca.column('jobs', 'twa_logsum_hbw_3', cache=True, cache_scope='iteration')
def twa_logsum_hbw_3(jobs, zones):
    return misc.reindex(zones.trip_weighted_average_logsum_hbw_am_income_3, jobs.job_zone_id)

@orca.column('jobs', 'twa_logsum_hbw_4', cache=True, cache_scope='iteration')
def twa_logsum_hbw_4(jobs, zones):
    return misc.reindex(zones.trip_weighted_average_logsum_hbw_am_income_4, jobs.job_zone_id)

@orca.column('jobs', 'vacant_jobs', cache=False, cache_scope='step')
def vacant_jobs(jobs, persons):
    vacant = pd.Series(np.zeros(len(jobs)), index=jobs.index)
    counts = persons.job_id.value_counts()
    counts = counts[counts.index > 0] # index can be -1 for unplaced jobs
    vacant.update(counts)
    vacant = jobs.number_of_jobs - vacant
    return vacant 


@orca.column('jobs', 'target_id', cache=True, cache_scope='step')
def target_id(jobs, parcels):
    return misc.reindex(parcels.target_id, jobs.parcel_id)

@orca.column('jobs', 'control_id', cache=True, cache_scope='step')
def control_id(jobs, parcels):
    return misc.reindex(parcels.control_id, jobs.parcel_id)

@orca.column('jobs', 'control_hct_id', cache=True, cache_scope='step')
def control_id(jobs, parcels):
    return misc.reindex(parcels.control_hct_id, jobs.parcel_id)
