import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils
#import urbansim_defaults.datasources
#####################
# buildings_lag1 VARIABLES (in alphabetic order)
#####################

@orca.column('buildings_lag1', 'age', cache=True, cache_scope='iteration')
def age(buildings_lag1, year):
    year_built = buildings_lag1.year_built
    year_built[buildings_lag1.has_valid_age_built==0] = np.nan
    return np.maximum(0, year - year_built)

@orca.column('buildings_lag1', 'avg_price_per_unit_in_zone', cache=True, cache_scope='iteration')
def avg_price_per_unit_in_zone(buildings_lag1, zones):
    zone_avg_price = buildings_lag1.unit_price.groupby(buildings_lag1.zone_id).mean()
    return misc.reindex(zone_avg_price, buildings_lag1.zone_id)

@orca.column('buildings_lag1', 'building_sqft', cache=True, cache_scope='iteration')
def building_sqft(buildings_lag1):
    results = np.zeros(buildings_lag1.local.shape[0],dtype=np.int32)
    where_res = np.where(buildings_lag1.residential_units > 0)[0]
    results[where_res] = buildings_lag1.residential_units.iloc[where_res] * buildings_lag1.sqft_per_unit.iloc[where_res]
    where_nonres = np.where(buildings_lag1.non_residential_sqft > 0)[0]
    results[where_nonres] = results[where_nonres] + buildings_lag1.non_residential_sqft.iloc[where_nonres]
    return pd.Series(results, index=buildings_lag1.index)

@orca.column('buildings_lag1', 'building_sqft_per_unit', cache=True, cache_scope='iteration')
def building_sqft_per_unit(buildings_lag1):
    a = buildings_lag1.residential_units.replace(0, np.nan)
    return buildings_lag1.building_sqft.divide(a).fillna(0)

@orca.column('buildings_lag1', 'building_type_name', cache=True, cache_scope='iteration')
def building_type_name(buildings_lag1, building_types):
    return misc.reindex(building_types.building_type_name, buildings_lag1.building_type_id)

@orca.column('buildings_lag1', 'employment_density_wwd', cache=True, cache_scope='step')
def employment_density_wwd(buildings_lag1, parcels):
    return misc.reindex(parcels.employment_density_wwd, buildings_lag1.parcel_id)

@orca.column('buildings_lag1', 'employment_retail_wwd', cache=True, cache_scope='step')
def employment_retail_wwd(buildings_lag1, parcels):
    return misc.reindex(parcels.employment_retail_wwd, buildings_lag1.parcel_id)

@orca.column('buildings_lag1', 'faz_id', cache=True)
def faz_id(buildings_lag1, zones):
    return misc.reindex(zones.faz_id, buildings_lag1.zone_id)

@orca.column('buildings_lag1', 'has_valid_age_built', cache=True, cache_scope='iteration')
def has_valid_age_built(buildings_lag1, settings):
    return buildings_lag1.year_built > settings.get('abs_min_year_built', 1800)

@orca.column('buildings_lag1', 'is_commercial', cache=True, cache_scope='iteration')
def is_commercial(buildings_lag1):
    return (buildings_lag1.building_type_name == 'commercial').astype("int16")

@orca.column('buildings_lag1', 'is_governmental', cache=True, cache_scope='iteration')
def is_governmental(buildings_lag1, building_types):
    return (misc.reindex(building_types.generic_building_type_description, buildings_lag1.building_type_id) == 'government').astype("int16")

@orca.column('buildings_lag1', 'is_industrial', cache=True, cache_scope='iteration')
def is_industrial(buildings_lag1):
    return (buildings_lag1.building_type_name == 'industrial').astype("int16")

@orca.column('buildings_lag1', 'is_mixed_use', cache=True, cache_scope='iteration')
def is_mixed_use(buildings_lag1):
    return (buildings_lag1.building_type_name == 'mixed_use').astype("int16")

@orca.column('buildings_lag1', 'is_multifamily', cache=True, cache_scope='iteration')
def is_multifamily(buildings_lag1):
    return (buildings_lag1.building_type_name == 'multi_family_residential').astype("int16")

@orca.column('buildings_lag1', 'is_office', cache=True, cache_scope='iteration')
def is_office(buildings_lag1):
    return (buildings_lag1.building_type_name == 'office').astype("int16")

@orca.column('buildings_lag1', 'is_residential', cache=True, cache_scope='iteration')
def is_residential(buildings_lag1, building_types):
    return (misc.reindex(building_types.is_residential, buildings_lag1.building_type_id) == 1).astype("bool8")

@orca.column('buildings_lag1', 'is_tcu', cache=True, cache_scope='iteration')
def is_tcu(buildings_lag1):
    return (buildings_lag1.building_type_name == 'tcu').astype("int16")

@orca.column('buildings_lag1', 'is_warehouse', cache=True, cache_scope='iteration')
def is_warehouse(buildings_lag1):
    return (buildings_lag1.building_type_name == 'warehousing').astype("int16")

@orca.column('buildings_lag1', 'job_spaces', cache=False)
def job_spaces(buildings_lag1):
    # TODO: get base year as an argument
    results = np.zeros(buildings_lag1.local.shape[0],dtype=np.int32)
    iexisting = np.where(buildings_lag1.year_built <= 2014)[0]
    ifuture = np.where(buildings_lag1.year_built > 2014)[0]
    results[iexisting] = buildings_lag1.job_capacity.iloc[iexisting]
    results[ifuture] = ((buildings_lag1.non_residential_sqft /
            buildings_lag1.sqft_per_job).fillna(0).astype('int')).iloc[ifuture]
    return pd.Series(results, index=buildings_lag1.index)

@orca.column('buildings_lag1', 'large_area_id', cache=True)
def large_area_id(buildings_lag1, parcels):
    return misc.reindex(parcels.large_area_id, buildings_lag1.parcel_id)

@orca.column('buildings_lag1', 'ln_price_residual', cache=True, cache_scope='iteration')
def ln_price_residual(buildings_lag1):
    from .abstract_variables import abstract_iv_residual
    return abstract_iv_residual(np.log(buildings_lag1.price_per_unit), np.log(buildings_lag1.avg_price_per_unit_in_zone),
                                buildings_lag1.price_per_unit > 0)

@orca.column('buildings_lag1', 'mortgage_cost', cache=True, cache_scope='iteration')
def mortgage_cost(buildings_lag1, parcels):
    pbsqft = misc.reindex(parcels.building_sqft_pcl, buildings_lag1.parcel_id).replace(0, np.nan)
    return (0.06/12 * (1+0.06/12)**360)/((((1+0.06/12)**360)-1)*12) * (
        buildings_lag1.unit_price * buildings_lag1.building_sqft_per_unit + 
        buildings_lag1.sqft_per_unit.divide(pbsqft).fillna(0) * 
        misc.reindex(parcels.land_value, buildings_lag1.parcel_id))

@orca.column('buildings_lag1', 'multifamily_generic_type', cache=True, cache_scope='iteration')
def multifamily_generic_type(buildings_lag1):
    return ((buildings_lag1.building_type_id == 4) | (buildings_lag1.building_type_id == 12)).astype("int16")

@orca.column('buildings_lag1', 'number_of_governmental_jobs', cache=True, cache_scope='step')
def number_of_governmental_jobs(buildings_lag1, jobs):
    return jobs.sector_id.groupby(jobs.building_id[np.in1d(jobs.sector_id, [18, 19])]).size().reindex(buildings_lag1.index).fillna(0).astype("int32")

@orca.column('buildings_lag1', 'number_of_households', cache=True, cache_scope='step')
def number_of_households(buildings_lag1, households):
    return households.building_id.groupby(households.building_id).size().reindex(buildings_lag1.index).fillna(0).astype("int32")

@orca.column('buildings_lag1', 'number_of_jobs', cache=True, cache_scope='step')
def number_of_jobs(buildings_lag1, jobs):
    return jobs.sector_id.groupby(jobs.building_id).size().reindex(buildings_lag1.index).fillna(0).astype("int32")

@orca.column('buildings_lag1', 'number_of_jobs', cache=True, cache_scope='step')
def number_of_non_home_based_jobs(buildings_lag1, jobs):
    return (jobs['home_based_status']==0).groupby(jobs.building_id).sum().reindex(buildings_lag1.index).fillna(0).astype("int32")

@orca.column('buildings_lag1', 'number_of_retail_jobs', cache=True, cache_scope='step')
def number_of_retail_jobs(buildings_lag1, jobs):
    return jobs.is_in_sector_group_retail.groupby(jobs.building_id).sum().reindex(buildings_lag1.index).fillna(0).astype("int32")

@orca.column('buildings_lag1', 'price_per_unit', cache=True)
def price_per_unit(buildings_lag1):
    """Price per sqft"""
    return buildings_lag1.unit_price * buildings_lag1.building_sqft_per_unit

@orca.column('buildings_lag1', 'sqft_per_job', cache=True, cache_scope='iteration')
def sqft_per_job(buildings_lag1, building_sqft_per_job):
    series1 = building_sqft_per_job.building_sqft_per_job.to_frame()    
    series2 = pd.DataFrame({'zone_id': buildings_lag1.zone_id, 'building_type_id': buildings_lag1.building_type_id}, index=buildings_lag1.index)
    df = pd.merge(series2, series1, left_on=['zone_id', 'building_type_id'], right_index=True, how="left")   
    return df.building_sqft_per_job

@orca.column('buildings_lag1', 'tractcity_id', cache=True)
def tractcity_id(buildings_lag1, parcels):
    return misc.reindex(parcels.tractcity_id, buildings_lag1.parcel_id)

@orca.column('buildings_lag1', 'twa_logsum_hbw_1', cache=True, cache_scope='iteration')
def twa_logsum_hbw_1(buildings_lag1, zones):
    return misc.reindex(zones.trip_weighted_average_logsum_hbw_am_income_1, buildings_lag1.zone_id)

@orca.column('buildings_lag1', 'twa_logsum_hbw_2', cache=True, cache_scope='iteration')
def twa_logsum_hbw_2(buildings_lag1, zones):
    return misc.reindex(zones.trip_weighted_average_logsum_hbw_am_income_2, buildings_lag1.zone_id)

@orca.column('buildings_lag1', 'twa_logsum_hbw_3', cache=True, cache_scope='iteration')
def twa_logsum_hbw_3(buildings_lag1, zones):
    return misc.reindex(zones.trip_weighted_average_logsum_hbw_am_income_3, buildings_lag1.zone_id)

@orca.column('buildings_lag1', 'twa_logsum_hbw_4', cache=True, cache_scope='iteration')
def twa_logsum_hbw_4(buildings_lag1, zones):
    return misc.reindex(zones.trip_weighted_average_logsum_hbw_am_income_4, buildings_lag1.zone_id)

@orca.column('buildings_lag1', 'unit_price', cache=True)
def unit_price(buildings_lag1, parcels):
    """total parcel value per unit (either building_sqft or DU)"""
    return misc.reindex(parcels.unit_price, buildings_lag1.parcel_id)

@orca.column('buildings_lag1', 'vacant_job_spaces', cache=False)
def vacant_job_spaces(buildings_lag1, jobs):
    return buildings_lag1.job_spaces.sub(
        jobs.building_id.value_counts(), fill_value=0)

@orca.column('buildings_lag1', 'vacant_residential_units', cache=False, cache_scope='iteration')
def vacant_residential_units(buildings_lag1, households):
    return buildings_lag1.residential_units.sub(
        households.building_id.value_counts(), fill_value=0)

@orca.column('buildings_lag1', 'zone_id', cache=True)
def zone_id(buildings_lag1, parcels):
    return misc.reindex(parcels.zone_id, buildings_lag1.parcel_id)

@orca.column('buildings_lag1', 'building_zone_id', cache=True)
def building_zone_id(buildings_lag1, parcels):
    return misc.reindex(parcels.zone_id, buildings_lag1.parcel_id)


# Functions
def number_of_jobs_of_sector_from_zone(sector, buildings_lag1, zones, jobs):
    from .variables_zones import number_of_jobs_of_sector
    return misc.reindex(number_of_jobs_of_sector(sector, zones, jobs), buildings_lag1.zone_id)








