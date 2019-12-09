import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# PARCELS VARIABLES (in alphabetic order)
#####################

@orca.column('parcels', 'acres_wwd', cache=True, cache_scope='step')
def acres_wwd(parcels):
    return parcels.parcel_sqft_wwd / 43560.0

@orca.column('parcels', 'ave_unit_size', cache=True, cache_scope='step')
def ave_unit_size(parcels, buildings):
    # Median building sqft per residential unit over zones
    return get_ave_unit_size_by_zone(buildings.is_residential == 1, buildings, parcels)

@orca.column('parcels', 'ave_unit_size_sf', cache=True, cache_scope='step')
def ave_unit_size_sf(parcels, buildings):
    # Median building sqft per single-family residential unit over zones
    return get_ave_unit_size_by_zone(buildings.is_singlefamily == 1, buildings, parcels)

@orca.column('parcels', 'ave_unit_size_mf', cache=True, cache_scope='step')
def ave_unit_size_mf(parcels, buildings):
    # Median building sqft per multi-family residential unit over zones
    return get_ave_unit_size_by_zone(buildings.is_multifamily == 1, buildings, parcels)

@orca.column('parcels', 'ave_unit_size_condo', cache=True, cache_scope='step')
def ave_unit_size_condo(parcels, buildings):
    # Median building sqft per condo residential unit over zones
    return get_ave_unit_size_by_zone(buildings.is_condo == 1, buildings, parcels)

@orca.column('parcels', 'average_income', cache=True, cache_scope='step')
def average_income(parcels, households):
    return households.income.groupby(households.parcel_id).mean().\
           reindex(parcels.index).fillna(0)
    
@orca.column('parcels', 'avg_building_age', cache=True, cache_scope='step')
def avg_building_age(parcels, buildings):
    reg_median = buildings.age.median()
    return buildings.age.groupby(buildings.parcel_id).mean().\
           reindex(parcels.index).fillna(reg_median)

@orca.column('parcels', 'blds_with_valid_age', cache=True, cache_scope='step')
def blds_with_valid_age(parcels, buildings):
    return buildings.has_valid_age_built.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'building_density_wwd', cache=True, cache_scope='step')
def building_density_wwd(parcels):
    return (parcels.building_sqft_wwd / parcels.parcel_sqft_wwd).replace(np.inf, 0).fillna(0)

@orca.column('parcels', 'building_sqft_pcl', cache=True, cache_scope='step')
def building_sqft_pcl(parcels, buildings):
    return buildings.building_sqft.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'building_sqft_wwd', cache=True, cache_scope='step')
def building_sqft_wwd(parcels, gridcells, settings):
    from abstract_variables import abstract_within_walking_distance_parcels
    return abstract_within_walking_distance_parcels("building_sqft_pcl", parcels, gridcells, settings)


@orca.column('parcels', 'county_id', cache=True, cache_scope='iteration')
def county_id(parcels, cities):
    return misc.reindex(cities.county_id, parcels.city_id)

@orca.column('parcels', 'capacity_opportunity_non_gov', cache=True, cache_scope='iteration')
def capacity_opportunity_non_gov(parcels):
    # use as a redevelopment filter (includes vacant parcels)
    return np.logical_or(parcels.building_sqft_pcl == 0, # if no buildings on parcels return True
        # OR the following chain of ANDs
        ((parcels.max_developable_capacity/parcels.building_sqft_pcl) > 3) & # parcel is not utilized
        (parcels.number_of_governmental_buildings == 0) & # no governmental buildings
        (parcels.avg_building_age >= 10) & # buildings older than 10 years
        np.logical_or( # if condo, the utilization should have a higher bar (it's more difficult to get all condo owners to agree)
            (parcels.max_developable_capacity / parcels.building_sqft_pcl) > 6, 
            parcels.land_use_type_id <> 15
            )&
        (parcels.job_capacity < 500)& # do not turn down buildings with lots of jobs
        ((parcels.total_improvement_value / parcels.parcel_sqft) < 250) # do not turn down expensive mansions
    )

@orca.column('parcels', 'capacity_opportunity_non_gov_relaxed', cache=True, cache_scope='step')
def capacity_opportunity_non_gov_relaxed(parcels):
    # use as a redevelopment filter in allocation mode (includes vacant parcels)
    return np.logical_or(parcels.building_sqft_pcl == 0, # if no buildings on parcels return True
        # OR the following chain of ANDs
        ((parcels.max_developable_capacity/parcels.building_sqft_pcl) > 1.1) & # parcel is not utilized
        (parcels.number_of_governmental_buildings == 0) & # no governmental buildings
        (parcels.avg_building_age >= 1) & # buildings older than 1 year
        np.logical_or( # if condo, the utilization should have a higher bar (it's more difficult to get all condo owners to agree)
            (parcels.max_developable_capacity / parcels.building_sqft_pcl) > 6, 
            parcels.land_use_type_id <> 15
            ) &
        (parcels.job_capacity < 500) & # do not turn down buildings with lots of jobs
        ((parcels.total_improvement_value / parcels.parcel_sqft) < 250) # do not turn down expensive mansions
    )

@orca.column('parcels', 'commercial_job_spaces', cache=True, cache_scope='step')
def commercial_job_spaces(parcels, buildings):
    return get_units_by_type(buildings.building_type_name == "commercial", buildings, parcels, units_attribute = "job_spaces")

@orca.column('parcels', 'condo_residential_units', cache=True, cache_scope='step')
def condo_residential_units(parcels, buildings):
    return get_units_by_type(buildings.building_type_name == "condo_residential", buildings, parcels)

@orca.column('parcels', 'developable_capacity', cache=True, cache_scope='forever')
def developable_capacity(parcels):
    return np.maximum(parcels.max_developable_capacity - parcels.building_sqft_pcl, 0)
                             
@orca.column('parcels', 'employment_density_wwd', cache=True, cache_scope='step')
def employment_density_wwd(parcels):
    return (parcels.number_of_jobs_wwd / parcels.acres_wwd).replace(np.inf, 0).fillna(0)

@orca.column('parcels', 'employment_retail_wwd', cache=True, cache_scope='step')
def employment_retail_wwd(parcels, gridcells, settings):
    from abstract_variables import abstract_within_walking_distance_parcels
    return abstract_within_walking_distance_parcels("number_of_retail_jobs", parcels, gridcells, settings)

@orca.column('parcels', 'existing_units', cache=True, cache_scope='step')
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


# @orca.column('parcels', 'growth_center_id', cache=True, cache_scope='iteration')
# def growth_center_id(parcels, parcels_geos):
    # print 'in parcels - growth_center_id'
    # return misc.reindex(parcels_geos.growth_center_id, parcels.faz_id)	

@orca.column('parcels', 'industrial_job_spaces', cache=True, cache_scope='step')
def industrial_job_spaces(parcels, buildings):
    return get_units_by_type(buildings.building_type_name == "industrial", buildings, parcels, units_attribute = "job_spaces")

@orca.column('parcels', 'income_per_person_wwd', cache=True, cache_scope='step')
def income_per_person_wwd(parcels, gridcells, settings):
    from abstract_variables import abstract_within_walking_distance_parcels
    return (abstract_within_walking_distance_parcels("total_income", parcels, gridcells, settings)/parcels.population_wwd).fillna(0)

@orca.column('parcels', 'invfar', cache=True, cache_scope='step')
def invfar(parcels):
    return (parcels.parcel_sqft.astype(float)/parcels.building_sqft_pcl.astype(float)).replace(np.inf, 0).fillna(0)

@orca.column('parcels', 'is_park', cache=True, cache_scope='step')
def is_park(parcels):
    return (parcels.land_use_type_id == 19)

@orca.column('parcels', 'job_capacity', cache=True, cache_scope='step')
def job_capacity(parcels, buildings):
    return buildings.job_capacity.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'land_area', cache=True, cache_scope='step')
def land_area(parcels, buildings):
    return buildings.land_area.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'land_cost', cache=True, cache_scope='step')
def land_cost(parcels): # toal value of the parcel
    return parcels.land_value + parcels.total_improvement_value

@orca.column('parcels', 'lnemp20da', cache=True, cache_scope='step')
def lnemp20da(parcels, zones):
    return np.log1p(misc.reindex(zones.jobs_within_20_min_tt_hbw_am_drive_alone, parcels.zone_id))

@orca.column('parcels', 'max_coverage', cache=True, cache_scope='forever')
def max_coverage(parcels, zoning_heights):
    cov = misc.reindex(zoning_heights.max_coverage, parcels.plan_type_id)
    cov[cov <= 0] = 0.8 # default
    return cov

@orca.column('parcels', 'max_developable_capacity', cache=True, cache_scope='forever')
def max_developable_capacity(parcels, parcel_zoning):
    #med_bld_sqft_per_du = int((parcels.building_sqft_pcl / parcels.residential_units).quantile())
    med_bld_sqft_per_du = 1870 # median of building sqft per unit in 2014
    values = parcel_zoning.local.loc[:, ["max_du", "max_far"]]
    values.loc[:, "max_far_from_dua"] = values.max_du / 43560.0 * med_bld_sqft_per_du
    return (values[["max_far", "max_far_from_dua"]].max(axis = 1)*parcels.parcel_sqft).reindex(parcels.index).fillna(0)

@orca.column('parcels', 'max_developable_nonresidential_capacity', cache=True, cache_scope='forever')
def max_developable_nonresidential_capacity(parcels, parcel_zoning):
    #med_bld_sqft_per_du = int((parcels.building_sqft_pcl / parcels.residential_units).quantile())
    #med_bld_sqft_per_du = 1870 # median of building sqft per unit in 2014
    #values = parcel_zoning.local.loc[:, ["max_du", "max_far"]]
    values = parcel_zoning.local.loc[:, ["max_far"]]
    #values.loc[:, "max_far_from_dua"] = values.max_du / 43560.0 * med_bld_sqft_per_du
    return (values.max_far * parcels.parcel_sqft).reindex(parcels.index).fillna(0)
	#return (parcel_zoning.max_far * parcels.parcel_sqft).reindex(parcels.index).fillna(0)

@orca.column('parcels', 'max_developable_residential_capacity', cache=True, cache_scope='forever')
def max_developable_residential_capacity(parcels, parcel_zoning):
    #med_bld_sqft_per_du = int((parcels.building_sqft_pcl / parcels.residential_units).quantile())
    med_bld_sqft_per_du = 1870 # median of building sqft per unit in 2014
    values = parcel_zoning.local.loc[:, ["max_du"]]
    values.loc[:, "max_far_from_dua"] = values.max_du / 43560.0 * med_bld_sqft_per_du
    return (values.max_far_from_dua * parcels.parcel_sqft).reindex(parcels.index).fillna(0)

@orca.column('parcels', 'max_dua', cache=True, cache_scope='forever')
def max_dua(parcels, zoning_heights):
    return misc.reindex(zoning_heights.max_du, parcels.plan_type_id)

@orca.column('parcels', 'max_far', cache=True, cache_scope='forever')
def max_far(parcels, zoning_heights):
    return misc.reindex(zoning_heights.max_far, parcels.plan_type_id)
  
@orca.column('parcels', 'max_height', cache=True, cache_scope='forever')
def max_height(parcels, zoning_heights):
    return misc.reindex(zoning_heights.max_height, parcels.plan_type_id)

@orca.column('parcels', 'multi_family_residential_units', cache=True, cache_scope='step')
def multi_family_residential_units(parcels, buildings):
    return get_units_by_type(buildings.building_type_name == "multi_family_residential", buildings, parcels)

@orca.column('parcels', 'nonres_building_sqft', cache=True, cache_scope='step')
def nonres_building_sqft(parcels, buildings):
    #    """Total sqft of non-resiential buldings"""
    return (buildings.building_sqft * (~buildings.is_residential)).groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'nonres_sqft', cache=True, cache_scope='step')
def nonres_sqft(parcels, buildings):
    #    """Total sqft of non-resiential space- includes non-res sqft in mixed use buildings"""
    return buildings.non_residential_sqft.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'number_of_buildings', cache=True, cache_scope='step')
def number_of_buildings(parcels, buildings):
    return buildings.parcel_id.groupby(buildings.parcel_id).size().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'number_of_good_public_schools', cache=True, cache_scope='step')
def number_of_good_public_schools(parcels, schools):
    return ((schools.total_score >= 8)*(schools.public == 1)).groupby(schools.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'number_of_good_public_schools_within_3000_radius', cache=True, cache_scope='step')
def number_of_good_public_schools_within_3000_radius(parcels, gridcells, settings):
    from abstract_variables import abstract_within_walking_distance_parcels
    return abstract_within_walking_distance_parcels("number_of_good_public_schools", parcels, gridcells, settings, walking_radius=3000)

@orca.column('parcels', 'number_of_governmental_buildings', cache=True, cache_scope='step')
def number_of_governmental_buildings(parcels, buildings):
    return (buildings.is_governmental == 1).groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'number_of_households', cache=True, cache_scope='step')
def number_of_households(parcels, households):
    return households.persons.groupby(households.parcel_id).size().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'number_of_households_wwd', cache=True, cache_scope='step')
def number_of_households_wwd(parcels, gridcells, settings):
    from abstract_variables import abstract_within_walking_distance_parcels
    return abstract_within_walking_distance_parcels("number_of_households", parcels, gridcells, settings)

@orca.column('parcels', 'number_of_jobs', cache=True, cache_scope='step')
def number_of_jobs(parcels, jobs):
    return jobs.parcel_id.groupby(jobs.parcel_id).size().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'number_of_jobs_wwd', cache=True, cache_scope='step')
def number_of_jobs_wwd(parcels, gridcells, settings):
    from abstract_variables import abstract_within_walking_distance_parcels
    return abstract_within_walking_distance_parcels("number_of_jobs", parcels, gridcells, settings)

@orca.column('parcels', 'number_of_retail_jobs', cache=True, cache_scope='step')
def number_of_retail_jobs(parcels, jobs):
    return jobs.is_in_sector_group_retail.groupby(jobs.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'office_job_spaces', cache=True, cache_scope='step')
def office_job_spaces(parcels, buildings):
    return get_units_by_type(buildings.building_type_name == "office", buildings, parcels, units_attribute = "job_spaces")

@orca.column('parcels', 'parcel_size', cache=True, cache_scope='forever')
def parcel_size(parcels):
    return parcels.parcel_sqft

@orca.column('parcels', 'parcel_sqft_wwd', cache=True, cache_scope='step')
def parcel_sqft_wwd(parcels, gridcells, settings):
    from abstract_variables import abstract_within_walking_distance_parcels
    return abstract_within_walking_distance_parcels("parcel_sqft", parcels, gridcells, settings)

@orca.column('parcels', 'park_area', cache=True, cache_scope='step')
def park_area(parcels):
    return ((parcels.land_use_type_id == 19) * parcels.parcel_sqft)

@orca.column('parcels', 'park_area_wwd', cache=True, cache_scope='step')
def park_area_wwd(parcels, gridcells, settings):
    from abstract_variables import abstract_within_walking_distance_parcels
    return abstract_within_walking_distance_parcels("park_area", parcels, gridcells, settings)

@orca.column('parcels', 'population_density_wwd', cache=True, cache_scope='step')
def population_density_wwd(parcels):
    return (parcels.population_wwd / parcels.acres_wwd).replace(np.inf, 0).fillna(0)

@orca.column('parcels', 'population_pcl', cache=True, cache_scope='step')
def population_pcl(parcels, households):
    return households.persons.groupby(households.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'population_wwd', cache=True, cache_scope='step')
def population_wwd(parcels, gridcells, settings):
    from abstract_variables import abstract_within_walking_distance_parcels
    return abstract_within_walking_distance_parcels("population_pcl", parcels, gridcells, settings)

@orca.column('parcels', 'residential_units', cache=True, cache_scope='step')
def residential_units(parcels, buildings):
    return buildings.residential_units.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'residential_sqft', cache=True, cache_scope='step')
def residential_sqft(parcels, buildings):
    return buildings.residential_sqft.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'retail_density_wwd', cache=True, cache_scope='step')
def retail_density_wwd(parcels):
    return (parcels.employment_retail_wwd / parcels.acres_wwd).replace(np.inf, 0).fillna(0)

@orca.column('parcels', 'single_family_residential_units', cache=True, cache_scope='step')
def single_family_residential_units(parcels, buildings):
    return get_units_by_type(buildings.building_type_name == "single_family_residential", buildings, parcels)

@orca.column('parcels', 'tcu_job_spaces', cache=True, cache_scope='step')
def tcu_job_spaces(parcels, buildings):
    return get_units_by_type(buildings.building_type_name == "tcu", buildings, parcels, units_attribute = "job_spaces")

@orca.column('parcels', 'total_income', cache=True, cache_scope='step')
def total_income(parcels, households):
    return households.income.groupby(households.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'total_improvement_value', cache=True, cache_scope='step')
def total_improvement_value(parcels, buildings):
    return buildings.improvement_value.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'total_job_spaces', cache=True, cache_scope='step')
def total_job_spaces(parcels, buildings):
    return buildings.job_spaces.groupby(buildings.parcel_id).sum().\
        reindex(parcels.index).fillna(0)

@orca.column('parcels', 'total_land_value_per_sqft', cache=True, cache_scope='step')
def total_land_value_per_sqft(parcels):
    return (parcels.land_cost/parcels.parcel_sqft).replace(np.inf, 0).fillna(0)

@orca.column('parcels', 'unit_name', cache=True)
def unit_name(parcels, land_use_types):
    return misc.reindex(land_use_types.unit_name, parcels.land_use_type_id)

@orca.column('parcels', 'unit_price', cache=True, cache_scope='step')
def unit_price(parcels):
    return ((parcels.land_value + parcels.total_improvement_value)/parcels.existing_units).replace(np.inf, 0).fillna(0)

#@orca.column('parcels', 'unit_price_residential2', cache=True, cache_scope='step')
#def unit_price_residential2(parcels, buildings):
#    results = np.zeros(parcels.local.shape[0]) 
#    # some parcels are all res or have no residential sqft
#    all_res_ix = np.where((parcels.building_sqft_pcl == parcels.nonres_building_sqft2) & (parcels.residential_units > 0))[0]
#    mixed_ix = np.where((parcels.building_sqft_pcl <> parcels.nonres_building_sqft2) & (parcels.residential_units > 0))[0]
#    # assume all square footage is residential for ones missing
#    results[all_res_ix] =  ((parcels.land_value + parcels.total_improvement_value)/parcels.residential_units).iloc[all_res_ix]
#    results[mixed_ix] =  (((parcels.land_value + parcels.total_improvement_value)/parcels.residential_units) * ((parcels.building_sqft_pcl - parcels.nonres_building_sqft2) / parcels.building_sqft_pcl)).iloc[mixed_ix]

#    res_unit_price =  pd.Series(results, index=parcels.index)
    
#    # now deal with parcels that are missing total value
#    df = parcels.to_frame(['zone_id', 'land_cost', 'residential_units'])
#    df['res_unit_price'] = res_unit_price
#    # dont want to include 0 in median calc
#    df['res_unit_price'].replace(0, np.nan, inplace = True)
#    # get the zonal median res unit price
#    df['zonal_median_unit_price'] = df.groupby('zone_id')['res_unit_price'].transform('median')
#    # apply median to parcels that have no total value
#    df['res_unit_price'] = np.where((df.land_cost == 0) & (df.residential_units > 0), df['zonal_median_unit_price'], df['res_unit_price'])
#    df['res_unit_price'].replace(np.nan, 0, inplace = True)
    
#    return df['res_unit_price']

@orca.column('parcels', 'unit_price_residential', cache=True, cache_scope='step')
def unit_price_residential(parcels, buildings):

    res_unit_price = (parcels.land_cost * (parcels.residential_sqft/parcels.building_sqft_pcl)) / parcels.residential_units
    
    # now deal with parcels that are missing total value
    df = parcels.to_frame(['zone_id', 'land_cost', 'residential_units'])
    df['res_unit_price'] = res_unit_price
    # dont want to include 0 in median calc
    df['res_unit_price'].replace(0, np.nan, inplace = True)
    # get the zonal median res unit price
    df['zonal_median_unit_price'] = df.groupby('zone_id')['res_unit_price'].transform('median')
    # apply median to parcels that have no total value
    df['res_unit_price'] = np.where((df.land_cost == 0) & (df.residential_units > 0), df['zonal_median_unit_price'], df['res_unit_price'])
    df['res_unit_price'].replace(np.nan, 0, inplace = True)
    
    return df['res_unit_price']


#    missing_ix = np.where((parcels.building_sqft_pcl == parcels.nonres_building_sqft) & (parcels.residential_units > 0))[0]
#    a = ((parcels.building_sqft_pcl - parcels.nonres_building_sqft) / parcels.building_sqft_pcl).replace(np.inf, 0).fillna(0)
#    x = (((parcels.land_value + parcels.total_improvement_value)/parcels.residential_units).replace(np.inf, 0).fillna(0)) * a
#    test = np.where((parcels.building_sqft_pcl == parcels.nonres_building_sqft) & (parcels.residential_units > 0))[0]
#    x.update(test, (parcels.land_value + parcels.total_improvement_value)/parcels.residential_units.replace(np.inf, 0).fillna(0))
    

#    residential_sqft = parcels.building_sqft_pcl - parcels.nonres_building_sqft
##df['total_sqft'] = df['residential_sqft']+df['non_residential_sqft']
#    total_value = parcels.land_value + parcels.total_improvement_value
#    percent_residential_sqft = residential_sqft/parcels.building_sqft_pcl
#    total_residential_value = percent_residential_sqft * total_value
#    return (total_residential_value/parcels.residential_units).replace(np.inf, 0).fillna(0)


@orca.column('parcels', 'unit_price_trunc', cache=True, cache_scope='step')
def unit_price_trunc(parcels):
    price = parcels.unit_price
    price[price < 1] = 1
    price[price > 1500] = 1500
    return price

@orca.column('parcels', 'warehousing_job_spaces', cache=True, cache_scope='step')
def warehousing_job_spaces(parcels, buildings):
    return get_units_by_type(buildings.building_type_name == "warehousing", buildings, parcels, units_attribute = "job_spaces")

# Functions
def get_ave_unit_size_by_zone(is_in, buildings, parcels):
    # Median building sqft per residential unit over zones
    # is_in is a logical Series giving the filter for subsetting the buildings
    # Values for parcels in zones with no residential buildings are imputed 
    # using the regional median.
    bsu = buildings.building_sqft_per_unit[is_in].replace(0, np.nan) # so that zeros are not counted
    reg_median = bsu.median()
    return misc.reindex(buildings.building_sqft_per_unit[is_in].groupby(buildings.zone_id[is_in]).median(), parcels.zone_id).\
           fillna(reg_median).replace(0, reg_median)

#def get_ave_parcel_res_value_by_zone(is_in, parcels):
#    # Median building sqft per residential unit over zones
#    # is_in is a logical Series giving the filter for subsetting the parcels
#    # Values for parcels in zones with no residential buildings are imputed 
#    # using the regional median.
#    df = parcels.to_frame(['zone_id', 'total_value', 'residential_units'])
#    zonal_median = pd.Series(0, df.index)
#    df['total_value2'] = np.where((df.total_value > 0) & (df.residential_units > 0), df['total_value'], 0).replace(0, np.nan)
#    df['total_value2'].replace(0, np.nan, inplace = True)
#    df['zonal_median'] = df.groupby('zone_id')['total_value'].transform('median')
#    zonal_median.update(df['zonal_median'])

    
#    res_ix = np.where((parcels.total_value > 0) & (parcels.residential_units > 0))[0]
#    res_median = (parcels.total_value).iloc[res_ix].median()
#    d['zonal_median'] = df.groupby('zone_id')['total_value'].transform('std')
#    #return parcels.total_value[is_in].groupby(parcels.zone_id[is_in]).median()
#    test = parcels.total_value.iloc[res_ix].groupby(parcel.zone_id.iloc[res_ix]).median()
#    return test

def get_units_by_type(is_type, buildings, parcels, units_attribute = "residential_units"):
    return buildings[units_attribute][is_type].groupby(buildings.parcel_id[is_type]).sum().\
           reindex(parcels.index).fillna(0)    