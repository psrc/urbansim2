import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# BUILDINGS VARIABLES (in alphabetic order)
#####################

@orca.column('buildings', 'building_sqft', cache=True, cache_scope='iteration')
def building_sqft(buildings):
    results = np.zeros(buildings.local.shape[0],dtype=np.int32)
    where_res = np.where(buildings.residential_units > 0)[0]
    results[where_res] = buildings.residential_units.iloc[where_res] * buildings.sqft_per_unit.iloc[where_res]
    where_nonres = np.where(buildings.non_residential_sqft > 0)[0]
    results[where_nonres] = results[where_nonres] + buildings.non_residential_sqft.iloc[where_nonres]
    return pd.Series(results, index=buildings.index)

@orca.column('buildings', 'building_type_name', cache=True, cache_scope='iteration')
def building_type_name(buildings, building_types):
    return misc.reindex(building_types.building_type_name, buildings.building_type_id)
    
@orca.column('buildings', 'faz_id', cache=True)
def faz_id(buildings, zones):
    return misc.reindex(zones.faz_id, buildings.zone_id)

@orca.column('buildings', 'job_spaces', cache=False)
def job_spaces(buildings):
    # TODO: get base year as an argument
    results = np.zeros(buildings.local.shape[0],dtype=np.int32)
    iexisting = np.where(buildings.year_built <= 2014)[0]
    ifuture = np.where(buildings.year_built > 2014)[0]
    results[iexisting] = buildings.job_capacity.iloc[iexisting]
    results[ifuture] = ((buildings.non_residential_sqft /
            buildings.sqft_per_job).fillna(0).astype('int')).iloc[ifuture]
    return pd.Series(results, index=buildings.index)

@orca.column('buildings', 'multifamily_type', cache=True, cache_scope='iteration')
def multifamily_type(buildings):
    return ((buildings.building_type_id == 4) | (buildings.building_type_id == 12)).astype("int16")

@orca.column('buildings', 'number_of_governmental_jobs', cache=True, cache_scope='step')
def number_of_governmental_jobs(buildings, jobs):
    return jobs.sector_id.groupby(jobs.building_id[np.in1d(jobs.sector_id, [18, 19])]).size().reindex(buildings.index).fillna(0).astype("int32")

@orca.column('buildings', 'number_of_jobs', cache=True, cache_scope='step')
def number_of_jobs(buildings, jobs):
    return jobs.sector_id.groupby(jobs.building_id).size().reindex(buildings.index).fillna(0).astype("int32")

@orca.column('buildings', 'sqft_per_job', cache=True, cache_scope='iteration')
def sqft_per_job(buildings, building_sqft_per_job):
    series1 = building_sqft_per_job.building_sqft_per_job.to_frame()    
    series2 = pd.DataFrame({'zone_id': buildings.zone_id, 'building_type_id': buildings.building_type_id}, index=buildings.index)
    df = pd.merge(series2, series1, left_on=['zone_id', 'building_type_id'], right_index=True, how="left")   
    return df.building_sqft_per_job

@orca.column('buildings', 'tractcity_id', cache=True)
def tractcity_id(buildings, parcels):
    return misc.reindex(parcels.tractcity_id, buildings.parcel_id)

@orca.column('buildings', 'vacant_job_spaces', cache=False)
def vacant_job_spaces(buildings, jobs):
    return buildings.job_spaces.sub(
        jobs.building_id.value_counts(), fill_value=0)

@orca.column('buildings', 'vacant_residential_units', cache=False, cache_scope='iteration')
def vacant_residential_units(buildings, households):
    return buildings.residential_units.sub(
        households.building_id.value_counts(), fill_value=0)

@orca.column('buildings', 'zone_id', cache=True)
def zone_id(buildings, parcels):
    return misc.reindex(parcels.zone_id, buildings.parcel_id)










