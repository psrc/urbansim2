import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc

from psrc_urbansim.vars.abstract_variables import abstract_travel_time_interaction_variable
from psrc_urbansim.vars.abstract_variables import abstract_logsum_interaction_variable

def network_distance_from_home_to_work(work_zones, location_zones):
    return abstract_travel_time_interaction_variable(orca.get_table("travel_data")["single_vehicle_to_work_travel_distance"], work_zones, location_zones, direction_from_home = False)

def avg_network_distance_from_home_to_work(work1_zones, work2_zones, location_zones):
    travel_data = orca.get_table("travel_data")
    worker1 = abstract_travel_time_interaction_variable(travel_data["single_vehicle_to_work_travel_distance"], work1_zones, location_zones, direction_from_home = False)
    worker2 = abstract_travel_time_interaction_variable(travel_data["single_vehicle_to_work_travel_distance"], work2_zones, location_zones, direction_from_home = False)
    # if worker2 does not work set it to the same value as worker1 in order for the average to be the worker1 distance  
    fillidx = np.where(np.logical_and(np.isnan(worker2), np.isnan(worker1)==False))
    worker2.iloc[fillidx] = worker1.iloc[fillidx]
    res = (worker1 + worker2)/2.
    # if there are no workers set it to the regional mean
    fillidx = np.where(np.isnan(res))
    res.iloc[fillidx] = res.mean()
    return res.values

def max_logsum_hbw_am_from_home_to_work(work1_zones, work2_zones, location_zones, agent_income_categories):
    travel_data = orca.get_table("travel_data")
    tm_dict = {1: travel_data["logsum_hbw_am_income_1"], 2: travel_data["logsum_hbw_am_income_2"], 
               3: travel_data["logsum_hbw_am_income_3"], 4: travel_data["logsum_hbw_am_income_4"]}
    worker1 = abstract_logsum_interaction_variable(tm_dict, agent_income_categories, work1_zones, location_zones, direction_from_home = False)
    worker2 = abstract_logsum_interaction_variable(tm_dict, agent_income_categories, work2_zones, location_zones, direction_from_home = False)
    # if worker2 does not work set it to the same value as worker1 in order for the maximum to be the worker1 value  
    fillidx = np.where(np.logical_and(np.isnan(worker2), np.isnan(worker1)==False))
    worker2[fillidx] = worker1[fillidx]
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
    
def avg_trip_weighted_zone_logsum(income_categories, twa_logsum_1, twa_logsum_2, twa_logsum_3, twa_logsum_4):
    return (income_categories == 1)*twa_logsum_1 + (income_categories == 2)*twa_logsum_2 + (income_categories == 3)*twa_logsum_3 + (income_categories == 4)*twa_logsum_4


from urbansim.models import dcm
dcm.avg_network_distance_from_home_to_work = avg_network_distance_from_home_to_work
dcm.max_logsum_hbw_am_from_home_to_work_wzone_logsum = max_logsum_hbw_am_from_home_to_work_wzone_logsum