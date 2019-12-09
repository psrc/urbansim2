import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# buildings VARIABLES (in alphabetic order)
#####################

@orca.column('buildings', 'age', cache=True, cache_scope='step')
def age(buildings, year):
    year_built = buildings.year_built
    year_built[buildings.has_valid_age_built==0] = np.nan
    return np.maximum(0, year - year_built)

#@orca.column('buildings', 'avg_price_per_unit_in_zone', cache=True, cache_scope='iteration')
#def avg_price_per_unit_in_zone(buildings, zones):
#    zone_avg_price = buildings.unit_price.groupby(buildings.zone_id).mean()
#    return misc.reindex(zone_avg_price, buildings.zone_id)

@orca.column('buildings', 'avg_residential_price_per_unit_in_zone', cache=True, cache_scope='step')
def avg_residential_price_per_unit_in_zone(buildings, zones):
    zone_avg_price = buildings.unit_price_residential.groupby(buildings.zone_id).mean()
    return misc.reindex(zone_avg_price, buildings.zone_id)

@orca.column('buildings', 'building_sqft', cache=True, cache_scope='step')
def building_sqft(buildings):
    results = np.zeros(buildings.local.shape[0],dtype=np.int32)
    where_res = np.where(buildings.residential_units > 0)[0]
    results[where_res] = buildings.residential_units.iloc[where_res] * buildings.sqft_per_unit_imputed.iloc[where_res]
    where_nonres = np.where(buildings.non_residential_sqft > 0)[0]
    results[where_nonres] = results[where_nonres] + buildings.non_residential_sqft.iloc[where_nonres]
    return pd.Series(results, index=buildings.index)

@orca.column('buildings', 'building_sqft_per_unit', cache=True, cache_scope='step')
def building_sqft_per_unit(buildings):
    a = buildings.residential_units.replace(0, np.nan)
    return buildings.building_sqft.divide(a).fillna(0)

@orca.column('buildings', 'building_type_name', cache=True, cache_scope='step')
def building_type_name(buildings, building_types):
    return misc.reindex(building_types.building_type_name, buildings.building_type_id)

@orca.column('buildings', 'city_id', cache=True, cache_scope='iteration')
def city_id(buildings, parcels):
    return misc.reindex(parcels.city_id, buildings.parcel_id)

@orca.column('buildings', 'county_id', cache=True, cache_scope='iteration')
def county_id(buildings, parcels):
    return misc.reindex(parcels.county_id, buildings.parcel_id)

@orca.column('buildings', 'employment_density_wwd', cache=True, cache_scope='step')
def employment_density_wwd(buildings, parcels):
    return misc.reindex(parcels.employment_density_wwd, buildings.parcel_id)

@orca.column('buildings', 'employment_retail_wwd', cache=True, cache_scope='step')
def employment_retail_wwd(buildings, parcels):
    return misc.reindex(parcels.employment_retail_wwd, buildings.parcel_id)

@orca.column('buildings', 'faz_id', cache=True, cache_scope='step')
def faz_id(buildings, zones):
    return misc.reindex(zones.faz_id, buildings.zone_id)

@orca.column('buildings', 'growth_center_id', cache=True)
def growth_center_id(buildings, parcels, parcels_geos):
    if "growth_center_id" in parcels.columns:
        return misc.reindex(parcels.growth_center_id, buildings.parcel_id)
    return misc.reindex(parcels_geos.growth_center_id, buildings.parcel_id)	

@orca.column('buildings', 'has_valid_age_built', cache=True, cache_scope='step')
def has_valid_age_built(buildings, settings):
    return buildings.year_built > settings.get('abs_min_year_built', 1800)

@orca.column('buildings', 'is_commercial', cache=True, cache_scope='step')
def is_commercial(buildings):
    return (buildings.building_type_name == 'commercial').astype("int16")

@orca.column('buildings', 'is_condo', cache=True, cache_scope='step')
def is_condo(buildings):
    return (buildings.building_type_name == 'condo_residential').astype("int16")

@orca.column('buildings', 'is_governmental', cache=True, cache_scope='step')
def is_governmental(buildings, building_types):
    return (misc.reindex(building_types.generic_building_type_description, buildings.building_type_id) == 'government').astype("int16")

@orca.column('buildings', 'is_industrial', cache=True, cache_scope='step')
def is_industrial(buildings):
    return (buildings.building_type_name == 'industrial').astype("int16")

@orca.column('buildings', 'is_mixed_use', cache=True, cache_scope='step')
def is_mixed_use(buildings):
    return (buildings.building_type_name == 'mixed_use').astype("int16")

@orca.column('buildings', 'is_multifamily', cache=True, cache_scope='step')
def is_multifamily(buildings):
    return (buildings.building_type_name == 'multi_family_residential').astype("int16")

@orca.column('buildings', 'is_office', cache=True, cache_scope='step')
def is_office(buildings):
    return (buildings.building_type_name == 'office').astype("int16")

@orca.column('buildings', 'is_residential', cache=True, cache_scope='step')
def is_residential(buildings, building_types):
    return (misc.reindex(building_types.is_residential, buildings.building_type_id) == 1).astype("bool8")

@orca.column('buildings', 'is_singlefamily', cache=True, cache_scope='step')
def is_singlefamily(buildings):
    return (buildings.building_type_name == 'single_family_residential').astype("int16")

@orca.column('buildings', 'is_tcu', cache=True, cache_scope='step')
def is_tcu(buildings):
    return (buildings.building_type_name == 'tcu').astype("int16")

@orca.column('buildings', 'is_warehouse', cache=True, cache_scope='step')
def is_warehouse(buildings):
    return (buildings.building_type_name == 'warehousing').astype("int16")

@orca.column('buildings', 'job_spaces', cache=False)
def job_spaces(buildings):
    # TODO: get base year as an argument
    base_year = 2014
    results = np.zeros(buildings.local.shape[0],dtype=np.int32)
    is_existing = np.logical_or(buildings.year_built <= base_year, buildings.job_capacity > 0)
    iexisting = np.where(is_existing)[0]
    ifuture = np.where(np.logical_not(is_existing))[0]
    results[iexisting] = buildings.job_capacity.iloc[iexisting]
    results[ifuture] = ((buildings.non_residential_sqft /
            buildings.sqft_per_job).fillna(0).astype('int')).iloc[ifuture]
    return pd.Series(results, index=buildings.index)

@orca.column('buildings', 'large_area_id', cache=True)
def large_area_id(buildings, parcels):
    return misc.reindex(parcels.large_area_id, buildings.parcel_id)

#@orca.column('buildings', 'ln_price_residual', cache=True, cache_scope='step')
#def ln_price_residual(buildings):
#    from abstract_variables import abstract_iv_residual
#    return abstract_iv_residual(np.log(buildings.price_per_unit), np.log(buildings.avg_price_per_unit_in_zone),
#                                buildings.price_per_unit > 0)

@orca.column('buildings', 'ln_price_residual_residential', cache=True, cache_scope='step')
def ln_price_residual_residential(buildings):
    from abstract_variables import abstract_iv_residual
    return abstract_iv_residual(np.log(buildings.unit_price_residential), np.log(buildings.avg_residential_price_per_unit_in_zone),
                                buildings.unit_price_residential > 0).fillna(0)

#@orca.column('buildings', 'mortgage_cost', cache=True, cache_scope='step')
#def mortgage_cost(buildings, parcels):
#    pbsqft = misc.reindex(parcels.building_sqft_pcl, buildings.parcel_id).replace(0, np.nan)
#    return (0.06/12 * (1+0.06/12)**360)/((((1+0.06/12)**360)-1)*12) * (
#        buildings.unit_price * buildings.building_sqft_per_unit + 
#        buildings.sqft_per_unit.divide(pbsqft).fillna(0) * 
#        misc.reindex(parcels.land_value, buildings.parcel_id))

@orca.column('buildings', 'mortgage_cost', cache=True, cache_scope='step')
def mortgage_cost(buildings, parcels):
    return (buildings.unit_price_residential * .06).fillna(0)

@orca.column('buildings', 'multifamily_generic_type', cache=True, cache_scope='step')
def multifamily_generic_type(buildings):
    return ((buildings.building_type_id == 4) | (buildings.building_type_id == 12)).astype("int16")

@orca.column('buildings', 'number_of_governmental_jobs', cache=True, cache_scope='step')
def number_of_governmental_jobs(buildings, jobs):
    return jobs.sector_id.groupby(jobs.building_id[np.in1d(jobs.sector_id, [12, 13])]).size().reindex(buildings.index).fillna(0).astype("int32")

@orca.column('buildings', 'number_of_households', cache=True, cache_scope='step')
def number_of_households(buildings, households):
    return households.building_id.groupby(households.building_id).size().reindex(buildings.index).fillna(0).astype("int32")

@orca.column('buildings', 'number_of_jobs', cache=True, cache_scope='step')
def number_of_jobs(buildings, jobs):
    return jobs.sector_id.groupby(jobs.building_id).size().reindex(buildings.index).fillna(0).astype("int32")

@orca.column('buildings', 'number_of_non_home_based_jobs', cache=True, cache_scope='step')
def number_of_non_home_based_jobs(buildings, jobs):
    return (jobs['home_based_status']==0).groupby(jobs.building_id).sum().reindex(buildings.index).fillna(0).astype("int32")

@orca.column('buildings', 'number_of_retail_jobs', cache=True, cache_scope='step')
def number_of_retail_jobs(buildings, jobs):
    return jobs.is_in_sector_group_retail.groupby(jobs.building_id).sum().reindex(buildings.index).fillna(0).astype("int32")

#@orca.column('buildings', 'price_per_unit', cache=True)
#def price_per_unit(buildings):
#    """Price per sqft"""
#    return buildings.unit_price * buildings.building_sqft_per_unit

@orca.column('buildings', 'residential_sqft', cache=True)
def residential_sqft(buildings):
    return buildings.residential_units * buildings.sqft_per_unit_imputed

@orca.column('buildings', 'sqft_per_job', cache=True, cache_scope='step')
def sqft_per_job(buildings, building_sqft_per_job):
    return _bld_sqft_per_job(buildings, building_sqft_per_job)

def _bld_sqft_per_job(buildings, building_sqft_per_job):
    series1 = building_sqft_per_job.building_sqft_per_job.to_frame()    
    series2 = pd.DataFrame({'zone_id': buildings.zone_id, 'building_type_id': buildings.building_type_id}, index=buildings.index)
    df = pd.merge(series2, series1, left_on=['zone_id', 'building_type_id'], right_index=True, how="left")   
    return df.building_sqft_per_job    


@orca.column('buildings', 'sqft_per_unit_imputed', cache=True, cache_scope='step')
def sqft_per_unit_imputed(buildings):
    # Imputes sqft_per_unit for residential buildings if missing, using regional median split by type
    is_mf = (buildings.is_multifamily == 1) & (buildings.residential_units > 0)
    is_sf = (buildings.is_singlefamily == 1) & (buildings.residential_units > 0)
    is_condo = (buildings.is_condo == 1) & (buildings.residential_units > 0)
    is_other_res = (buildings.is_residential == 1) & (buildings.residential_units > 0) & (buildings.is_multifamily == 0) & (buildings.is_singlefamily == 0) & (buildings.is_condo == 0)
    is_non_res = (buildings.is_residential == 0) & (buildings.residential_units > 0)
    results = buildings.sqft_per_unit.copy()
    results[is_mf] = results[is_mf].replace(0, buildings.sqft_per_unit[is_mf].median())
    results[is_sf] = results[is_sf].replace(0, buildings.sqft_per_unit[is_sf].median())
    results[is_condo] = results[is_condo].replace(0, buildings.sqft_per_unit[is_condo].median())
    results[is_other_res] = results[is_other_res].replace(0, buildings.sqft_per_unit[is_other_res].median())
    results[is_non_res] = results[is_non_res].replace(0, results[(is_mf | is_sf | is_condo)].median())
    return results

# @orca.column('buildings', 'target_vacancy_rate', cache=True, cache_scope='iteration')
# def faz_id(buildings, zones):
    # return misc.reindex(target_vacancies.target_vacancy_rate, buildings.index)

@orca.column('buildings', 'tractcity_id', cache=True)
def tractcity_id(buildings, parcels):
    return misc.reindex(parcels.tractcity_id, buildings.parcel_id)

@orca.column('buildings', 'twa_logsum_hbw_1', cache=True, cache_scope='step')
def twa_logsum_hbw_1(buildings, zones):
    return misc.reindex(zones.trip_weighted_average_logsum_hbw_am_income_1, buildings.zone_id)

@orca.column('buildings', 'twa_logsum_hbw_2', cache=True, cache_scope='step')
def twa_logsum_hbw_2(buildings, zones):
    return misc.reindex(zones.trip_weighted_average_logsum_hbw_am_income_2, buildings.zone_id)

@orca.column('buildings', 'twa_logsum_hbw_3', cache=True, cache_scope='step')
def twa_logsum_hbw_3(buildings, zones):
    return misc.reindex(zones.trip_weighted_average_logsum_hbw_am_income_3, buildings.zone_id)

@orca.column('buildings', 'twa_logsum_hbw_4', cache=True, cache_scope='step')
def twa_logsum_hbw_4(buildings, zones):
    return misc.reindex(zones.trip_weighted_average_logsum_hbw_am_income_4, buildings.zone_id)

@orca.column('buildings', 'unit_price', cache=True) # needed for indicators (new development shiny app)
def unit_price(buildings, parcels):
    """total parcel value per unit (either building_sqft or DU)"""
    return misc.reindex(parcels.unit_price, buildings.parcel_id)

@orca.column('buildings', 'unit_price_residential', cache=True)
def unit_price_residential(buildings, parcels):
    """total parcel value per unit (either building_sqft or DU)"""
    return misc.reindex(parcels.unit_price_residential, buildings.parcel_id)

@orca.column('buildings', 'vacant_job_spaces', cache=False)
def vacant_job_spaces(buildings):
    return buildings.job_spaces.sub(buildings.number_of_non_home_based_jobs, fill_value=0)

@orca.column('buildings', 'vacant_residential_units', cache=False, cache_scope='step')
def vacant_residential_units(buildings, households):
    counts = households.building_id.value_counts()
    counts = counts[counts.index >= 0] # index can be -1 for unplaced households
    return buildings.residential_units.sub(counts, fill_value=0)

@orca.column('buildings', 'zone_id', cache=True, cache_scope='iteration')
def zone_id(buildings, parcels):
    return misc.reindex(parcels.zone_id, buildings.parcel_id)

@orca.column('buildings', 'building_zone_id', cache=True)
def building_zone_id(buildings, parcels):
    return misc.reindex(parcels.zone_id, buildings.parcel_id)

@orca.column('buildings', 'parcel_land_value', cache=True)
def parcel_land_value(buildings, parcels):
    return misc.reindex(parcels.land_cost, buildings.parcel_id)

@orca.column('buildings', 'pbsqft', cache=True)
def pbsqft(buildings, parcels):
    return misc.reindex(parcels.building_sqft_pcl, buildings.parcel_id).replace(0, np.nan)

# Functions
def number_of_jobs_of_sector_from_zone(sector, buildings, zones, jobs):
    from variables_zones import number_of_jobs_of_sector
    return misc.reindex(number_of_jobs_of_sector(sector, zones, jobs), buildings.zone_id)








