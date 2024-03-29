import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc

from psrc_urbansim.vars.abstract_variables import abstract_travel_time_interaction_variable
from psrc_urbansim.vars.abstract_variables import abstract_logsum_interaction_variable

def network_distance_from_home_to_work(work_zones, location_zones):
    travel_data = orca.get_table("travel_data")
    worker = abstract_travel_time_interaction_variable(travel_data["single_vehicle_to_work_travel_distance"], work_zones, location_zones, direction_from_home = False)
    return worker.values

def generalized_cost_from_home_to_work(work_zones, location_zones):
    travel_data = orca.get_table("travel_data")
    worker = abstract_travel_time_interaction_variable(travel_data["single_vehicle_to_work_travel_cost"], work_zones, location_zones, direction_from_home = False)
    return worker.values

def set_worker2_to_worker1_if_needed(worker1, worker2):
    # if worker2 does not work and worker1 does, set it to the same value as worker1 in order for the desired quantity the same as worker1  
    fillidx = np.where(np.logical_and(np.isnan(worker2), np.isnan(worker1)==False))
    worker2.iloc[fillidx] = worker1.iloc[fillidx]
    return worker2
    
def avg_network_distance_from_home_to_work(work1_zones, work2_zones, location_zones):
    travel_data = orca.get_table("travel_data")
    worker1 = abstract_travel_time_interaction_variable(travel_data["single_vehicle_to_work_travel_distance"], work1_zones, location_zones, direction_from_home = False)
    worker2 = abstract_travel_time_interaction_variable(travel_data["single_vehicle_to_work_travel_distance"], work2_zones, location_zones, direction_from_home = False)
    worker2 = set_worker2_to_worker1_if_needed(worker1, worker2)
    worker1 = set_worker2_to_worker1_if_needed(worker2, worker1)
    res = (worker1 + worker2)/2.
    # if there are no workers set it to the regional mean; if the mean is na, set it to 0
    return res.fillna(res.mean()).fillna(0).values

def max_network_distance_from_home_to_work(work1_zones, work2_zones, location_zones):
    travel_data = orca.get_table("travel_data")
    worker1 = abstract_travel_time_interaction_variable(travel_data["single_vehicle_to_work_travel_distance"], work1_zones, location_zones, direction_from_home = False)
    worker2 = abstract_travel_time_interaction_variable(travel_data["single_vehicle_to_work_travel_distance"], work2_zones, location_zones, direction_from_home = False)
    worker2 = set_worker2_to_worker1_if_needed(worker1, worker2)
    worker1 = set_worker2_to_worker1_if_needed(worker2, worker1)   
    res = pd.concat([worker1, worker2], axis=1).max(axis=1)
    # if there are no workers set it to the regional max or a high number
    return res.fillna(res.max()).fillna(999999).values

def max_logsum_hbw_am_from_home_to_work(work1_zones, work2_zones, location_zones, agent_income_categories):
    travel_data = orca.get_table("travel_data")
    tm_dict = {1: travel_data["logsum_hbw_am_income_1"], 2: travel_data["logsum_hbw_am_income_2"], 
               3: travel_data["logsum_hbw_am_income_3"], 4: travel_data["logsum_hbw_am_income_4"]}
    worker1 = abstract_logsum_interaction_variable(tm_dict, agent_income_categories, work1_zones, location_zones, direction_from_home = False)
    worker2 = abstract_logsum_interaction_variable(tm_dict, agent_income_categories, work2_zones, location_zones, direction_from_home = False)
    fillidx = np.where(np.logical_and(np.isnan(worker2), np.isnan(worker1)==False))
    worker2[fillidx] = worker1[fillidx]
    fillidx = np.where(np.logical_and(np.isnan(worker1), np.isnan(worker2)==False))
    worker1[fillidx] = worker2[fillidx]
    res = np.maximum.reduce([worker1, worker2])
    return res

def max_logsum_hbw_am_from_home_to_work_wzone_logsum(work1_zones, work2_zones, location_zones, agent_income_categories, 
                                                          twa_logsum_1, twa_logsum_2, twa_logsum_3, twa_logsum_4,
                                                          zonal_threshold=-20):
    """If max_logsum_hbw_am_from_home_to_work < zonal_threshold, use avg_trip_weighted_zone_logsum."""
    res = max_logsum_hbw_am_from_home_to_work(work1_zones, work2_zones, location_zones, agent_income_categories)
    avg_trip = avg_trip_weighted_zone_logsum(agent_income_categories, twa_logsum_1, twa_logsum_2, twa_logsum_3, twa_logsum_4)
    fillidx = np.where(np.logical_or(np.isnan(res), res < zonal_threshold))
    res[fillidx] = avg_trip.iloc[fillidx]
    return res

def logsum_hbw_am_from_home_to_work(work_zones, location_zones, agent_income_categories):
    travel_data = orca.get_table("travel_data")
    tm_dict = {1: travel_data["logsum_hbw_am_income_1"], 2: travel_data["logsum_hbw_am_income_2"], 
               3: travel_data["logsum_hbw_am_income_3"], 4: travel_data["logsum_hbw_am_income_4"]}
    worker = abstract_logsum_interaction_variable(tm_dict, agent_income_categories, work_zones, location_zones, direction_from_home = False)
    return worker

def logsum_hbw_am_from_home_to_work_wzone_logsum(work_zones, location_zones, agent_income_categories, 
                                                          twa_logsum_1, twa_logsum_2, twa_logsum_3, twa_logsum_4,
                                                          zonal_threshold=-20):
    """If max_logsum_hbw_am_from_home_to_work < zonal_threshold, use avg_trip_weighted_zone_logsum."""
    res = logsum_hbw_am_from_home_to_work(work_zones, location_zones, agent_income_categories)
    avg_trip = avg_trip_weighted_zone_logsum(agent_income_categories, twa_logsum_1, twa_logsum_2, twa_logsum_3, twa_logsum_4)
    fillidx = np.where(np.logical_or(np.isnan(res), res < zonal_threshold))
    res[fillidx] = avg_trip.iloc[fillidx]
    return res
    
def avg_trip_weighted_zone_logsum(income_categories, twa_logsum_1, twa_logsum_2, twa_logsum_3, twa_logsum_4):
    return (income_categories == 1)*twa_logsum_1 + (income_categories == 2)*twa_logsum_2 + (income_categories == 3)*twa_logsum_3 + (income_categories == 4)*twa_logsum_4

def empden_zone_sector(sector, bzone_id):
    # non-interaction
    from .variables_zones import number_of_jobs_of_sector
    zones = orca.get_table('zones')
    zone_density = number_of_jobs_of_sector(sector, zones, orca.get_table('jobs'))/zones.acres
    zone_density[~np.isfinite(zone_density)] = 0
    return misc.reindex(zone_density, bzone_id)

def ln_am_total_transit_time_walk_from_home_to_work(work_zones, location_zones):
    travel_data = orca.get_table("travel_data")
    worker = np.log1p(abstract_travel_time_interaction_variable(travel_data["am_total_transit_time_walk"], work_zones, location_zones, direction_from_home = False))
    return worker.values 

from urbansim.models import dcm
dcm.network_distance_from_home_to_work = network_distance_from_home_to_work
dcm.avg_network_distance_from_home_to_work = avg_network_distance_from_home_to_work
dcm.max_logsum_hbw_am_from_home_to_work_wzone_logsum = max_logsum_hbw_am_from_home_to_work_wzone_logsum
dcm.logsum_hbw_am_from_home_to_work_wzone_logsum = logsum_hbw_am_from_home_to_work_wzone_logsum
dcm.empden_zone_sector = empden_zone_sector
dcm.generalized_cost_from_home_to_work = generalized_cost_from_home_to_work
dcm.ln_am_total_transit_time_walk_from_home_to_work = ln_am_total_transit_time_walk_from_home_to_work

#from psrc_urbansim import dcm_weighted_sampling 
#dcm_weighted_sampling.network_distance_from_home_to_work = network_distance_from_home_to_work
#dcm_weighted_sampling.avg_network_distance_from_home_to_work = avg_network_distance_from_home_to_work
#dcm_weighted_sampling.max_logsum_hbw_am_from_home_to_work_wzone_logsum = max_logsum_hbw_am_from_home_to_work_wzone_logsum
#dcm_weighted_sampling.logsum_hbw_am_from_home_to_work_wzone_logsum = logsum_hbw_am_from_home_to_work_wzone_logsum
#dcm_weighted_sampling.empden_zone_sector = empden_zone_sector
#dcm_weighted_sampling.generalized_cost_from_home_to_work = generalized_cost_from_home_to_work
#dcm_weighted_sampling.ln_am_total_transit_time_walk_from_home_to_work = ln_am_total_transit_time_walk_from_home_to_work