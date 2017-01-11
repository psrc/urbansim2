import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# BUILDINGS VARIABLES (in alphabetic order)
#####################

@orca.column('buildings', 'avg_price_per_unit_in_zone', cache=True, cache_scope='iteration')
def avg_price_per_unit_in_zone(buildings, zones):
    zone_avg_price = buildings.unit_price.groupby(buildings.zone_id).mean()
    return misc.reindex(zone_avg_price, buildings.zone_id)

@orca.column('buildings', 'building_sqft', cache=True, cache_scope='iteration')
def building_sqft(buildings):
    results = np.zeros(buildings.local.shape[0],dtype=np.int32)
    where_res = np.where(buildings.residential_units > 0)[0]
    results[where_res] = buildings.residential_units.iloc[where_res] * buildings.sqft_per_unit.iloc[where_res]
    where_nonres = np.where(buildings.non_residential_sqft > 0)[0]
    results[where_nonres] = results[where_nonres] + buildings.non_residential_sqft.iloc[where_nonres]
    return pd.Series(results, index=buildings.index)

@orca.column('buildings', 'building_sqft_per_unit', cache=True, cache_scope='iteration')
def building_sqft_per_unit(buildings):
    a = buildings.residential_units.replace(0, np.nan)
    return buildings.building_sqft.divide(a).fillna(0)

@orca.column('buildings', 'building_type_name', cache=True, cache_scope='iteration')
def building_type_name(buildings, building_types):
    return misc.reindex(building_types.building_type_name, buildings.building_type_id)
    
@orca.column('buildings', 'empden_zone_s1', cache=True, cache_scope='step')
def empden_zone_s1(buildings, zones, jobs):
    return density_of_jobs_of_sector_from_zone(1, buildings, zones, jobs)

@orca.column('buildings', 'empden_zone_s2', cache=True, cache_scope='step')
def empden_zone_s2(buildings, zones, jobs):
    return density_of_jobs_of_sector_from_zone(2, buildings, zones, jobs)

@orca.column('buildings', 'empden_zone_s3', cache=True, cache_scope='step')
def empden_zone_s3(buildings, zones, jobs):
    return density_of_jobs_of_sector_from_zone(3, buildings, zones, jobs)

@orca.column('buildings', 'empden_zone_s4', cache=True, cache_scope='step')
def empden_zone_s4(buildings, zones, jobs):
    return density_of_jobs_of_sector_from_zone(4, buildings, zones, jobs)

@orca.column('buildings', 'empden_zone_s5', cache=True, cache_scope='step')
def empden_zone_s5(buildings, zones, jobs):
    return density_of_jobs_of_sector_from_zone(5, buildings, zones, jobs)

@orca.column('buildings', 'empden_zone_s6', cache=True, cache_scope='step')
def empden_zone_s6(buildings, zones, jobs):
    return density_of_jobs_of_sector_from_zone(6, buildings, zones, jobs)

@orca.column('buildings', 'empden_zone_s7', cache=True, cache_scope='step')
def empden_zone_s7(buildings, zones, jobs):
    return density_of_jobs_of_sector_from_zone(7, buildings, zones, jobs)

@orca.column('buildings', 'empden_zone_s8', cache=True, cache_scope='step')
def empden_zone_s8(buildings, zones, jobs):
    return density_of_jobs_of_sector_from_zone(8, buildings, zones, jobs)

@orca.column('buildings', 'empden_zone_s9', cache=True, cache_scope='step')
def empden_zone_s9(buildings, zones, jobs):
    return density_of_jobs_of_sector_from_zone(9, buildings, zones, jobs)

@orca.column('buildings', 'empden_zone_s10', cache=True, cache_scope='step')
def empden_zone_s10(buildings, zones, jobs):
    return density_of_jobs_of_sector_from_zone(10, buildings, zones, jobs)

@orca.column('buildings', 'empden_zone_s11', cache=True, cache_scope='step')
def empden_zone_s11(buildings, zones, jobs):
    return density_of_jobs_of_sector_from_zone(11, buildings, zones, jobs)

@orca.column('buildings', 'empden_zone_s12', cache=True, cache_scope='step')
def empden_zone_s12(buildings, zones, jobs):
    return density_of_jobs_of_sector_from_zone(12, buildings, zones, jobs)

@orca.column('buildings', 'employment_density_wwd', cache=True, cache_scope='step')
def employment_density_wwd(buildings, parcels):
    return misc.reindex(parcels.employment_density_wwd, buildings.parcel_id)

@orca.column('buildings', 'employment_retail_wwd', cache=True, cache_scope='step')
def employment_retail_wwd(buildings, parcels):
    return misc.reindex(parcels.employment_retail_wwd, buildings.parcel_id)

@orca.column('buildings', 'faz_id', cache=True)
def faz_id(buildings, zones):
    return misc.reindex(zones.faz_id, buildings.zone_id)

@orca.column('buildings', 'is_commercial', cache=True, cache_scope='iteration')
def is_commercial(buildings):
    return (buildings.building_type_name == 'commercial').astype("int16")

@orca.column('buildings', 'is_industrial', cache=True, cache_scope='iteration')
def is_industrial(buildings):
    return (buildings.building_type_name == 'industrial').astype("int16")

@orca.column('buildings', 'is_mixed_use', cache=True, cache_scope='iteration')
def is_mixed_use(buildings):
    return (buildings.building_type_name == 'mixed_use').astype("int16")

@orca.column('buildings', 'is_office', cache=True, cache_scope='iteration')
def is_office(buildings):
    return (buildings.building_type_name == 'office').astype("int16")

@orca.column('buildings', 'is_tcu', cache=True, cache_scope='iteration')
def is_tcu(buildings):
    return (buildings.building_type_name == 'tcu').astype("int16")

@orca.column('buildings', 'is_warehouse', cache=True, cache_scope='iteration')
def is_warehouse(buildings):
    return (buildings.building_type_name == 'warehousing').astype("int16")

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

@orca.column('buildings', 'large_area_id', cache=True)
def large_area_id(buildings, parcels):
    return misc.reindex(parcels.large_area_id, buildings.parcel_id)

@orca.column('buildings', 'ln_price_residual', cache=True, cache_scope='iteration')
def ln_price_residual(buildings):
    from abstract_variables import abstract_iv_residual
    return abstract_iv_residual(np.log(buildings.price_per_unit), np.log(buildings.avg_price_per_unit_in_zone),
                                buildings.price_per_unit > 0)

@orca.column('buildings', 'mortgage_cost', cache=True, cache_scope='iteration')
def mortgage_cost(buildings, parcels):
    pbsqft = misc.reindex(parcels.building_sqft_pcl, buildings.parcel_id).replace(0, np.nan)
    return (0.06/12 * (1+0.06/12)**360)/((((1+0.06/12)**360)-1)*12) * (
        buildings.unit_price * buildings.building_sqft_per_unit + 
        buildings.sqft_per_unit.divide(pbsqft).fillna(0) * 
        misc.reindex(parcels.land_value, buildings.parcel_id))

@orca.column('buildings', 'multifamily_type', cache=True, cache_scope='iteration')
def multifamily_type(buildings):
    return ((buildings.building_type_id == 4) | (buildings.building_type_id == 12)).astype("int16")

@orca.column('buildings', 'number_of_governmental_jobs', cache=True, cache_scope='step')
def number_of_governmental_jobs(buildings, jobs):
    return jobs.sector_id.groupby(jobs.building_id[np.in1d(jobs.sector_id, [18, 19])]).size().reindex(buildings.index).fillna(0).astype("int32")

@orca.column('buildings', 'number_of_jobs', cache=True, cache_scope='step')
def number_of_jobs(buildings, jobs):
    return jobs.sector_id.groupby(jobs.building_id).size().reindex(buildings.index).fillna(0).astype("int32")

@orca.column('buildings', 'number_of_retail_jobs', cache=True, cache_scope='step')
def number_of_retail_jobs(buildings, jobs):
    return jobs.is_in_sector_group_retail.groupby(jobs.building_id).sum().reindex(buildings.index).fillna(0).astype("int32")

@orca.column('buildings', 'price_per_unit', cache=True)
def price_per_unit(buildings):
    """Price per sqft"""
    return buildings.unit_price * buildings.building_sqft_per_unit

@orca.column('buildings', 'sqft_per_job', cache=True, cache_scope='iteration')
def sqft_per_job(buildings, building_sqft_per_job):
    series1 = building_sqft_per_job.building_sqft_per_job.to_frame()    
    series2 = pd.DataFrame({'zone_id': buildings.zone_id, 'building_type_id': buildings.building_type_id}, index=buildings.index)
    df = pd.merge(series2, series1, left_on=['zone_id', 'building_type_id'], right_index=True, how="left")   
    return df.building_sqft_per_job

@orca.column('buildings', 'tractcity_id', cache=True)
def tractcity_id(buildings, parcels):
    return misc.reindex(parcels.tractcity_id, buildings.parcel_id)

@orca.column('buildings', 'twa_logsum_hbw_1', cache=True, cache_scope='iteration')
def twa_logsum_hbw_1(buildings, zones):
    return misc.reindex(zones.trip_weighted_average_logsum_hbw_am_income_1, buildings.zone_id)

@orca.column('buildings', 'twa_logsum_hbw_2', cache=True, cache_scope='iteration')
def twa_logsum_hbw_2(buildings, zones):
    return misc.reindex(zones.trip_weighted_average_logsum_hbw_am_income_2, buildings.zone_id)

@orca.column('buildings', 'twa_logsum_hbw_3', cache=True, cache_scope='iteration')
def twa_logsum_hbw_3(buildings, zones):
    return misc.reindex(zones.trip_weighted_average_logsum_hbw_am_income_3, buildings.zone_id)

@orca.column('buildings', 'twa_logsum_hbw_4', cache=True, cache_scope='iteration')
def twa_logsum_hbw_4(buildings, zones):
    return misc.reindex(zones.trip_weighted_average_logsum_hbw_am_income_4, buildings.zone_id)

@orca.column('buildings', 'unit_price', cache=True)
def unit_price(buildings, parcels):
    """total parcel value per unit (either building_sqft or DU)"""
    return misc.reindex(parcels.unit_price, buildings.parcel_id)

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

@orca.column('buildings', 'building_zone_id', cache=True)
def building_zone_id(buildings, parcels):
    return misc.reindex(parcels.zone_id, buildings.parcel_id)


# Functions
def number_of_jobs_of_sector_from_zone(sector, buildings, zones, jobs):
    from variables_zones import number_of_jobs_of_sector
    return misc.reindex(number_of_jobs_of_sector(sector, zones, jobs), buildings.zone_id)

def density_of_jobs_of_sector_from_zone(sector, buildings, zones, jobs):
    from variables_zones import number_of_jobs_of_sector
    zone_density = number_of_jobs_of_sector(sector, zones, jobs)/zones.acres
    zone_density[~np.isfinite(zone_density)] = 0
    return misc.reindex(zone_density, buildings.zone_id)







