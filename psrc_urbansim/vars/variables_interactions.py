import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc

from psrc_urbansim.vars.abstract_variables import abstract_travel_time_interaction_variable

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


from urbansim.models import dcm
dcm.avg_network_distance_from_home_to_work = avg_network_distance_from_home_to_work
