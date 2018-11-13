import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# PARCELS VARIABLES (in alphabetic order)
#####################

@orca.column('parcels', 'acres_wwd', cache=True, cache_scope='iteration')
def acres_wwd(parcels):
    return parcels.parcel_sqft_wwd / 43560.0

@orca.column('parcels', 'ave_unit_size', cache=True, cache_scope='iteration')
def ave_unit_size(parcels, buildings):
    # Median building sqft per residential unit over zones
    return get_ave_unit_size_by_zone(buildings.is_residential == 1, buildings, parcels)

@orca.column('parcels', 'ave_unit_size_sf', cache=True, cache_scope='iteration')
def ave_unit_size_sf(parcels, buildings):
    # Median building sqft per single-family residential unit over zones
    return get_ave_unit_size_by_zone(buildings.is_singlefamily == 1, buildings, parcels)

@orca.column('parcels', 'ave_unit_size_mf', cache=True, cache_scope='iteration')
def ave_unit_size_mf(parcels, buildings):
    # Median building sqft per multi-family residential unit over zones
    return get_ave_unit_size_by_zone(buildings.is_multifamily == 1, buildings, parcels)

@orca.column('parcels', 'ave_unit_size_condo', cache=True, cache_scope='iteration')
def ave_unit_size_condo(parcels, buildings):
    # Median building sqft per condo residential unit over zones
    return get_ave_unit_size_by_zone(buildings.is_condo == 1, buildings, parcels)

@orca.column('parcels', 'average_income', cache=True, cache_scope='iteration')
def average_income(parcels, households):
    return households.income.groupby(households.parcel_id).mean().\
           reindex(parcels.index).fillna(0)
    
@orca.column('parcels', 'avg_building_age', cache=True, cache_scope='iteration')
def avg_building_age(parcels, buildings):
    reg_median = buildings.age.median()
    return buildings.age.groupby(buildings.parcel_id).mean().\
           reindex(parcels.index).fillna(reg_median)

@orca.column('parcels', 'blds_with_valid_age', cache=True, cache_scope='iteration')
def blds_with_valid_age(parcels, buildings):
    return buildings.has_valid_age_built.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'building_density_wwd', cache=True, cache_scope='iteration')
def building_density_wwd(parcels):
    return (parcels.building_sqft_wwd / parcels.parcel_sqft_wwd).replace(np.inf, 0).fillna(0)

@orca.column('parcels', 'building_sqft_pcl', cache=True, cache_scope='iteration')
def building_sqft_pcl(parcels, buildings):
    return buildings.building_sqft.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'building_sqft_wwd', cache=True, cache_scope='iteration')
def building_sqft_wwd(parcels, gridcells, settings):
    from abstract_variables import abstract_within_walking_distance_parcels
    return abstract_within_walking_distance_parcels("building_sqft_pcl", parcels, gridcells, settings)

@orca.column('parcels', 'capacity_opportunity_non_gov', cache=True, cache_scope='iteration')
def capacity_opportunity_non_gov(parcels):
    # use as a redevelopment filter
    return np.logical_or(parcels.building_sqft_pcl == 0, # if no buildings on parcels return True
        # OR the following chain of ANDs
        (parcels.max_developable_capacity/parcels.building_sqft_pcl > 3)& # parcel is not utilized
        (parcels.number_of_governmental_buildings == 0)& # no governmental buildings
        (parcels.avg_building_age >= 10)& # buildings older than 20 years
        np.logical_or( # if condo, the utilization should have a higher bar (it's more difficult to get all condo owners to agree)
            parcels.max_developable_capacity / parcels.building_sqft_pcl > 6, 
            parcels.land_use_type_id <> 15
            )&
        (parcels.job_capacity < 500)& # do not turn down buildings with lots of jobs
        (parcels.total_improvement_value / parcels.parcel_sqft < 250) # do not turn down expensive mansions
    )

@orca.column('parcels', 'developable_capacity', cache=True, cache_scope='forever')
def developable_capacity(parcels):
    return np.maximum(parcels.max_developable_capacity - parcels.building_sqft_pcl, 0)
                             
@orca.column('parcels', 'employment_density_wwd', cache=True, cache_scope='step')
def employment_density_wwd(parcels):
    return (parcels.number_of_jobs_wwd / parcels.acres_wwd).replace(np.inf, 0).fillna(0)

@orca.column('parcels', 'employment_retail_wwd', cache=True, cache_scope='iteration')
def employment_retail_wwd(parcels, gridcells, settings):
    from abstract_variables import abstract_within_walking_distance_parcels
    return abstract_within_walking_distance_parcels("number_of_retail_jobs", parcels, gridcells, settings)

@orca.column('parcels', 'existing_units', cache=True, cache_scope='iteration')
def existing_units(parcels):
    results = np.zeros(parcels.local.shape[0], dtype=np.int32)
    for name in ["building_sqft_pcl", "parcel_sqft", "residential_units"]:
        if name == "building_sqft_pcl":
            w = np.where(parcels.unit_name == 'building_sqft')[0]
        else:
            w = np.where(parcels.unit_name == name)[0]
        results[w] = parcels[name].iloc[w].astype(np.int32)
    return pd.Series(results, index=parcels.index)

@orca.column('parcels', 'faz_id', cache=True)
def faz_id(parcels, zones):
    return misc.reindex(zones.faz_id, parcels.zone_id)

@orca.column('parcels', 'income_per_person_wwd', cache=True, cache_scope='iteration')
def income_per_person_wwd(parcels, gridcells, settings):
    from abstract_variables import abstract_within_walking_distance_parcels
    return (abstract_within_walking_distance_parcels("total_income", parcels, gridcells, settings)/parcels.population_wwd).fillna(0)

@orca.column('parcels', 'invfar', cache=True, cache_scope='iteration')
def invfar(parcels):
    return (parcels.parcel_sqft.astype(float)/parcels.building_sqft_pcl.astype(float)).replace(np.inf, 0).fillna(0)

@orca.column('parcels', 'is_park', cache=True, cache_scope='iteration')
def is_park(parcels):
    return (parcels.land_use_type_id == 19)

@orca.column('parcels', 'job_capacity', cache=True, cache_scope='iteration')
def job_capacity(parcels, buildings):
    return buildings.job_capacity.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'land_area', cache=True, cache_scope='iteration')
def land_area(parcels, buildings):
    return buildings.land_area.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'land_cost', cache=True, cache_scope='iteration')
def land_cost(parcels): # toal value of the parcel
    return parcels.land_value + parcels.total_improvement_value

@orca.column('parcels', 'lnemp20da', cache=True, cache_scope='iteration')
def lnemp20da(parcels, zones):
    return np.log1p(misc.reindex(zones.jobs_within_20_min_tt_hbw_am_drive_alone, parcels.zone_id))

@orca.column('parcels', 'max_developable_capacity', cache=True, cache_scope='forever')
def max_developable_capacity(parcels, parcel_zoning):
    #med_bld_sqft_per_du = int((parcels.building_sqft_pcl / parcels.residential_units).quantile())
    med_bld_sqft_per_du = 1870 # median of building sqft per unit in 2014
    values = parcel_zoning.maximum.copy()
    subset = values.loc[values.index.get_level_values('constraint_type') == 'units_per_acre']
    values.update((subset /43560.0 * med_bld_sqft_per_du).astype(values.dtype))
    return values.groupby(level="parcel_id").max().reindex(parcels.index).fillna(0)

@orca.column('parcels', 'max_dua', cache=True, cache_scope='forever')
def max_dua(parcels, parcel_zoning):
    return parcel_zoning.local.xs("units_per_acre", level="constraint_type").maximum.groupby(level="parcel_id").min().\
           reindex(parcels.index)

@orca.column('parcels', 'max_far', cache=True, cache_scope='forever')
def max_far(parcels, parcel_zoning):
    return parcel_zoning.local.xs("far", level="constraint_type").maximum.groupby(level="parcel_id").min().\
           reindex(parcels.index)
  
@orca.column('parcels', 'max_height', cache=True, cache_scope='forever')
def max_height(parcels, parcel_zoning):
    return parcel_zoning.local.max_height.groupby(level="parcel_id").min().reindex(parcels.index)

@orca.column('parcels', 'nonres_building_sqft', cache=True, cache_scope='iteration')
def nonres_building_sqft(parcels, buildings):
    return (buildings.building_sqft * (~buildings.is_residential)).groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'number_of_buildings', cache=True, cache_scope='iteration')
def number_of_buildings(parcels, buildings):
    return buildings.parcel_id.groupby(buildings.parcel_id).size().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'number_of_good_public_schools', cache=True, cache_scope='iteration')
def number_of_good_public_schools(parcels, schools):
    return ((schools.total_score >= 8)*(schools.public == 1)).groupby(schools.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'number_of_good_public_schools_within_3000_radius', cache=True, cache_scope='iteration')
def number_of_good_public_schools_within_3000_radius(parcels, gridcells, settings):
    from abstract_variables import abstract_within_walking_distance_parcels
    return abstract_within_walking_distance_parcels("number_of_good_public_schools", parcels, gridcells, settings, walking_radius=3000)

@orca.column('parcels', 'number_of_governmental_buildings', cache=True, cache_scope='iteration')
def number_of_governmental_buildings(parcels, buildings):
    return (buildings.is_governmental == 1).groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(parcels, households):
    return households.persons.groupby(households.parcel_id).size().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'number_of_households_wwd', cache=True, cache_scope='iteration')
def number_of_households_wwd(parcels, gridcells, settings):
    from abstract_variables import abstract_within_walking_distance_parcels
    return abstract_within_walking_distance_parcels("number_of_households", parcels, gridcells, settings)

@orca.column('parcels', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(parcels, jobs):
    return jobs.parcel_id.groupby(jobs.parcel_id).size().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'number_of_jobs_wwd', cache=True, cache_scope='iteration')
def number_of_jobs_wwd(parcels, gridcells, settings):
    from abstract_variables import abstract_within_walking_distance_parcels
    return abstract_within_walking_distance_parcels("number_of_jobs", parcels, gridcells, settings)

@orca.column('parcels', 'number_of_retail_jobs', cache=True, cache_scope='iteration')
def number_of_retail_jobs(parcels, jobs):
    return jobs.is_in_sector_group_retail.groupby(jobs.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'parcel_size', cache=True, cache_scope='forever')
def parcel_size(parcels):
    return parcels.parcel_sqft

@orca.column('parcels', 'parcel_sqft_wwd', cache=True, cache_scope='iteration')
def parcel_sqft_wwd(parcels, gridcells, settings):
    from abstract_variables import abstract_within_walking_distance_parcels
    return abstract_within_walking_distance_parcels("parcel_sqft", parcels, gridcells, settings)

@orca.column('parcels', 'park_area', cache=True, cache_scope='iteration')
def park_area(parcels):
    return ((parcels.land_use_type_id == 19) * parcels.parcel_sqft)

@orca.column('parcels', 'park_area_wwd', cache=True, cache_scope='iteration')
def park_area_wwd(parcels, gridcells, settings):
    from abstract_variables import abstract_within_walking_distance_parcels
    return abstract_within_walking_distance_parcels("park_area", parcels, gridcells, settings)

@orca.column('parcels', 'population_density_wwd', cache=True, cache_scope='step')
def population_density_wwd(parcels):
    return (parcels.population_wwd / parcels.acres_wwd).replace(np.inf, 0).fillna(0)

@orca.column('parcels', 'population_pcl', cache=True, cache_scope='iteration')
def population_pcl(parcels, households):
    return households.persons.groupby(households.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'population_wwd', cache=True, cache_scope='iteration')
def population_wwd(parcels, gridcells, settings):
    from abstract_variables import abstract_within_walking_distance_parcels
    return abstract_within_walking_distance_parcels("population_pcl", parcels, gridcells, settings)

@orca.column('parcels', 'residential_units', cache=True, cache_scope='iteration')
def residential_units(parcels, buildings):
    return buildings.residential_units.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'retail_density_wwd', cache=True, cache_scope='step')
def retail_density_wwd(parcels):
    return (parcels.employment_retail_wwd / parcels.acres_wwd).replace(np.inf, 0).fillna(0)

@orca.column('parcels', 'total_income', cache=True, cache_scope='iteration')
def total_income(parcels, households):
    return households.income.groupby(households.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'total_improvement_value', cache=True, cache_scope='iteration')
def total_improvement_value(parcels, buildings):
    return buildings.improvement_value.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'total_job_spaces', cache=True, cache_scope='iteration')
def total_job_spaces(parcels, buildings):
    return buildings.job_spaces.groupby(buildings.parcel_id).sum().\
        reindex(parcels.index).fillna(0)

@orca.column('parcels', 'total_land_value_per_sqft', cache=True, cache_scope='iteration')
def total_land_value_per_sqft(parcels):
    return (parcels.land_cost/parcels.parcel_sqft).replace(np.inf, 0).fillna(0)

@orca.column('parcels', 'unit_name', cache=True)
def unit_name(parcels, land_use_types):
    return misc.reindex(land_use_types.unit_name, parcels.land_use_type_id)

@orca.column('parcels', 'unit_price', cache=True, cache_scope='iteration')
def unit_price(parcels):
    return ((parcels.land_value + parcels.total_improvement_value)/parcels.existing_units).replace(np.inf, 0).fillna(0)

@orca.column('parcels', 'unit_price_trunc', cache=True, cache_scope='iteration')
def unit_price_trunc(parcels):
    price = parcels.unit_price
    price[price < 1] = 1
    price[price > 1500] = 1500
    return price

# Functions
def get_ave_unit_size_by_zone(is_in, buildings, parcels):
    # Median building sqft per residential unit over zones
    # is_in is a logical Series giving the filter for subsetting the buildings
    # Values for parcels in zones with no residential buildings are imputed 
    # using the regional median.
    bsu = buildings.building_sqft_per_unit[is_in].replace(0, np.nan) # so that zeros are not counted
    reg_median = bsu.median()
    return buildings.building_sqft_per_unit[is_in].groupby(buildings.zone_id[is_in]).median().\
           reindex(parcels.index).fillna(reg_median).replace(0, reg_median)
