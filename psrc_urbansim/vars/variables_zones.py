import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils
from abstract_variables import abstract_trip_weighted_average_from_home, abstract_weighted_access
from abstract_variables import abstract_access_within_threshold_variable_from_origin
from abstract_variables import abstract_travel_time_variable_to_destination

#####################
# ZONES VARIABLES (in alphabetic order)
#####################
@orca.column('zones', 'acres', cache=True, cache_scope='session')
def acres(zones):
    # sum of parcel sqft
    return zones.area / 43560.0

@orca.column('zones', 'area', cache=True, cache_scope='session')
def area(zones, parcels):
    # sum of parcel sqft
    return parcels.parcel_sqft.groupby(parcels.zone_id).sum().reindex(zones.index).fillna(0)

@orca.column('zones', 'avg_school_score', cache=True, cache_scope='iteration')
def avg_school_score(zones, fazes):
    return misc.reindex(fazes.avg_school_score, zones.faz_id)

@orca.column('zones', 'building_sqft', cache=True, cache_scope='iteration')
def building_sqft(zones, buildings):
    return buildings.sqft_per_unit.groupby(buildings.zone_id).sum().\
           reindex(zones.index).fillna(0)

@orca.column('zones', 'Business_Services', cache=True, cache_scope='iteration')
def Business_Services(zones, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 7)).groupby(jobs.job_zone_id).sum().\
	        reindex(zones.index).fillna(0)

@orca.column('zones', 'Construction', cache=True, cache_scope='iteration')
def Construction(zones, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 2)).groupby(jobs.job_zone_id).sum().\
	        reindex(zones.index).fillna(0)

@orca.column('zones', 'edu', cache=True, cache_scope='iteration')
def edu(zones, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 13)).groupby(jobs.job_zone_id).sum().\
	        reindex(zones.index).fillna(0)

@orca.column('zones', 'Food_Services', cache=True, cache_scope='iteration')
def Food_Services(zones, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 10)).groupby(jobs.job_zone_id).sum().\
	        reindex(zones.index).fillna(0)

@orca.column('zones', 'generalized_cost_hbw_am_drive_alone_to_bellevue_cbd')
def generalized_cost_hbw_am_drive_alone_to_bellevue_cbd(zones, travel_data):
    """Generalized cost for travel to the Bellevue CBD. It is the minimum of costs for travels to zones that have bellevue_cbd=1.
    """
    is_in_cbd = np.where(zones.bellevue_cbd == 1)[0]
    min_values = np.array(zones.local.shape[0]*[np.inf], dtype="float32")
    for zone in zones.index[is_in_cbd]:
        min_values = np.minimum(min_values, abstract_travel_time_variable_to_destination(travel_data.single_vehicle_to_work_travel_cost, zone))
    # zones within CBD get the minimum, so that all of them have the same number
    min_values.iloc[is_in_cbd] = min_values.iloc[is_in_cbd].min()
    return min_values

@orca.column('zones', 'generalized_cost_hbw_am_drive_alone_to_cbd')
def generalized_cost_hbw_am_drive_alone_to_cbd(zones):
    """Generalized cost for travel to either the Seattle CBD or Bellevue CBD, which ever is closer, i.e. take the minimum of these.
    """
    return np.minimum(zones.generalized_cost_hbw_am_drive_alone_to_seattle_cbd, zones.generalized_cost_hbw_am_drive_alone_to_bellevue_cbd)
    
@orca.column('zones', 'generalized_cost_hbw_am_drive_alone_to_seattle_cbd')
def generalized_cost_hbw_am_drive_alone_to_seattle_cbd(zones, travel_data):
    """Generalized cost for travel to the Seattle CBD. It is the minimum of costs for travels to zones that have seattle_cbd=1.
    """
    is_in_cbd = np.where(zones.seattle_cbd == 1)[0]
    min_values = np.array(zones.local.shape[0]*[np.inf], dtype="float32")
    for zone in zones.index[is_in_cbd]:
        min_values = np.minimum(min_values, abstract_travel_time_variable_to_destination(travel_data.single_vehicle_to_work_travel_cost, zone))
    # zones within CBD get the minimum, so that all of them have the same number
    min_values.iloc[is_in_cbd] = min_values.iloc[is_in_cbd].min()
    return min_values

@orca.column('zones', 'generalized_cost_weighted_access_to_employment_hbw_am_drive_alone')
def generalized_cost_weighted_access_to_employment_hbw_am_drive_alone(zones, travel_data):
    """Total employment in zone j divided by generalized cost from zone i to j,
    The travel time used is for the home-based-work am trips by auto with 
    drive-alone.
    """
    return abstract_weighted_access(travel_data.single_vehicle_to_work_travel_cost, zones.number_of_jobs)

@orca.column('zones', 'generalized_cost_weighted_access_to_population_hbw_am_drive_alone')
def generalized_cost_weighted_access_to_population_hbw_am_drive_alone(zones, travel_data):
    """Sum of population in zone j divided by generalized cost from zone i to j,
    The travel time used is for the home-based-work am trips by auto with 
    drive-alone.
    """
    return abstract_weighted_access(travel_data.single_vehicle_to_work_travel_cost, zones.population)

@orca.column('zones', 'government', cache=True, cache_scope='iteration')
def government(zones, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 12)).groupby(jobs.job_zone_id).sum().\
	        reindex(zones.index).fillna(0)

@orca.column('zones', 'Healthcare', cache=True, cache_scope='iteration')
def Healthcare(zones, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 9)).groupby(jobs.job_zone_id).sum().\
	        reindex(zones.index).fillna(0)

@orca.column('zones', 'jobs_within_10_min_tt_hbw_am_walk')
def jobs_within_10_min_tt_hbw_am_walk(zones, travel_data):    
    return abstract_access_within_threshold_variable_from_origin(travel_data.am_walk_time_in_minutes, zones.number_of_jobs, 10)

@orca.column('zones', 'jobs_within_10_min_tt_hbw_am_drive_alone')
def jobs_within_10_min_tt_hbw_am_drive_alone(zones, travel_data):    
    return abstract_access_within_threshold_variable_from_origin(travel_data.am_single_vehicle_to_work_travel_time, zones.number_of_jobs, 10)

@orca.column('zones', 'jobs_within_20_min_tt_hbw_am_drive_alone')
def jobs_within_20_min_tt_hbw_am_drive_alone(zones, travel_data):    
    return abstract_access_within_threshold_variable_from_origin(travel_data.am_single_vehicle_to_work_travel_time, zones.number_of_jobs, 20)

@orca.column('zones', 'jobs_within_20_min_tt_hbw_am_transit_walk')
def jobs_within_20_min_tt_hbw_am_transit_walk(zones, travel_data):    
    return abstract_access_within_threshold_variable_from_origin(travel_data.am_total_transit_time_walk, zones.number_of_jobs, 20)

@orca.column('zones', 'jobs_within_30_min_tt_hbw_am_transit_walk')
def jobs_within_30_min_tt_hbw_am_transit_walk(zones, travel_data):    
    return abstract_access_within_threshold_variable_from_origin(travel_data.am_total_transit_time_walk, zones.number_of_jobs, 30)

@orca.column('zones', 'jobs_within_30_min_tt_hbw_am_drive_alone')
def jobs_within_30_min_tt_hbw_am_drive_alone(zones, travel_data):
    return abstract_access_within_threshold_variable_from_origin(travel_data.am_single_vehicle_to_work_travel_time, zones.number_of_jobs, 30)

@orca.column('zones', 'Manuf', cache=True, cache_scope='iteration')
def Manuf(zones, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 3)).groupby(jobs.job_zone_id).sum().\
	        reindex(zones.index).fillna(0)

@orca.column('zones', 'median_income')
def median_income(zones, households):
    s = households.income.groupby(households.zone_id).quantile()
    return s.reindex(zones.index).fillna(s.quantile())

@orca.column('zones', 'median_parcel_sqft')
def median_parcel_sqft(zones, parcels):
    s = parcels.parcel_sqft.groupby(parcels.zone_id).quantile()
    return s.reindex(zones.index).fillna(s.quantile())

@orca.column('zones', 'Natural_resources', cache=True, cache_scope='iteration')
def Natural_resources(zones, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 1)).groupby(jobs.job_zone_id).sum().\
	        reindex(zones.index).fillna(0)

@orca.column('zones', 'nonres_sqft', cache=True, cache_scope='iteration')
def nonres_sqft(zones, buildings):
    return buildings.non_residential_sqft.groupby(buildings.zone_id).sum().\
           reindex(zones.index).fillna(0)
	
@orca.column('zones', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(zones, households):
    return households.persons.groupby(households.zone_id).size().\
           reindex(zones.index).fillna(0)

@orca.column('zones', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(zones, jobs):
    return jobs.job_zone_id.groupby(jobs.job_zone_id).size().\
           reindex(zones.index).fillna(0)

@orca.column('zones', 'number_of_jobs_per_acre', cache=True, cache_scope='iteration')
def number_of_jobs_per_acre(zones):
    return (zones.number_of_jobs/zones.acres).replace(np.inf,0).fillna(0)

@orca.column('zones', 'Personal_Services', cache=True, cache_scope='iteration')
def Personal_Services(zones, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 11)).groupby(jobs.job_zone_id).sum().\
	        reindex(zones.index).fillna(0)

@orca.column('zones', 'population', cache=True, cache_scope='iteration')
def population(zones, households):
    return households.persons.groupby(households.zone_id).sum().\
           reindex(zones.index).fillna(0)

@orca.column('zones', 'population_per_acre', cache=True, cache_scope='iteration')
def population_per_acre(zones):
    return (zones.population/zones.acres).replace(np.inf,0).fillna(0)

@orca.column('zones', 'Private_Ed', cache=True, cache_scope='iteration')
def Private_Ed(zones, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 8)).groupby(jobs.job_zone_id).sum().\
	        reindex(zones.index).fillna(0)

@orca.column('zones', 'residential_units', cache=True, cache_scope='iteration')
def residetial_units(zones, buildings):
    return buildings.residential_units.groupby(buildings.zone_id).sum().\
           reindex(zones.index).fillna(0)

@orca.column('zones', 'Retail', cache=True, cache_scope='iteration')
def Retail(zones, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 5)).groupby(jobs.job_zone_id).sum().\
	        reindex(zones.index).fillna(0)

def trip_weighted_average_logsum_hbw_am_income_category(zones, travel_data, income_category):
    return abstract_trip_weighted_average_from_home(travel_data["logsum_hbw_am_income_%s" % income_category], 
                                                    travel_data["am_pk_period_drive_alone_vehicle_trips"],
                                                    travel_data.index.get_level_values('from_zone_id'), zones)

@orca.column('zones', 'trip_weighted_average_logsum_hbw_am_income_1', cache=True, cache_scope='iteration')
def trip_weighted_average_logsum_hbw_am_income_1(zones, travel_data):
    return trip_weighted_average_logsum_hbw_am_income_category(zones, travel_data, 1)

@orca.column('zones', 'trip_weighted_average_logsum_hbw_am_income_2', cache=True, cache_scope='iteration')
def trip_weighted_average_logsum_hbw_am_income_2(zones, travel_data):
    return trip_weighted_average_logsum_hbw_am_income_category(zones, travel_data, 2)

@orca.column('zones', 'trip_weighted_average_logsum_hbw_am_income_3', cache=True, cache_scope='iteration')
def trip_weighted_average_logsum_hbw_am_income_3(zones, travel_data):
    return trip_weighted_average_logsum_hbw_am_income_category(zones, travel_data, 3)

@orca.column('zones', 'trip_weighted_average_logsum_hbw_am_income_4', cache=True, cache_scope='iteration')
def trip_weighted_average_logsum_hbw_am_income_4(zones, travel_data):
    return trip_weighted_average_logsum_hbw_am_income_category(zones, travel_data, 4)

@orca.column('zones', 'trip_weighted_average_time_hbw_from_home_am_drive_alone', cache=True, cache_scope='iteration')
def trip_weighted_average_time_hbw_from_home_am_drive_alone(zones, travel_data):
    return abstract_trip_weighted_average_from_home(travel_data["am_single_vehicle_to_work_travel_time"], 
                                                    travel_data["am_pk_period_drive_alone_vehicle_trips"],
                                                    travel_data.index.get_level_values('from_zone_id'), zones)
@orca.column('zones', 'WTU', cache=True, cache_scope='iteration')
def WTU(zones, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 4)).groupby(jobs.job_zone_id).sum().\
	        reindex(zones.index).fillna(0)
													
# Functions
def number_of_jobs_of_sector(sector, zones, jobs):
    return (jobs.sector_id==sector).groupby(jobs.job_zone_id).sum().reindex(zones.index).fillna(0).astype("int32")

def generalized_cost_hbw_am_drive_alone_to_zone(zone_id, travel_data):
    return 