import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils
from psrc_urbansim.vars.variables_interactions import avg_network_distance_from_home_to_work

def is_worker_n(n, persons_for_estimation):
    return np.logical_and(persons_for_estimation.member_id == n, persons_for_estimation.job_id > 0)

#####################
# persons_for_estimation VARIABLES (in alphabetic order)
#####################

@orca.column('persons_for_estimation', 'faz_id', cache=True)
def faz_id(persons_for_estimation, zones):
    return misc.reindex(zones.faz_id, persons_for_estimation.zone_id)

@orca.column('persons_for_estimation', 'household_district_id', cache=True)
def district_id(persons_for_estimation, zones):
    return misc.reindex(zones.district_id, persons_for_estimation.zone_id)

@orca.column('persons_for_estimation', 'household_income_category', cache=True)
def household_income_category(persons_for_estimation, households_for_estimation):
    # calling households_for_estimation.income_category returns an non-indexed  
    # array, so converting to df for now. 
    df = households_for_estimation.to_frame()
    return misc.reindex(df.income_category, persons_for_estimation.household_id)
    
@orca.column('persons_for_estimation', 'parcel_id', cache=True)
def parcel_id(persons_for_estimation, households):
    return misc.reindex(households.parcel_id, persons_for_estimation.household_id)

@orca.column('persons_for_estimation', 'tractcity_id', cache=True)
def tractcity_id(persons_for_estimation, households):
    return misc.reindex(households.tractcity_id, persons_for_estimation.household_id)

@orca.column('persons_for_estimation', 'worker1', cache=True)
def worker1(persons_for_estimation):
    return is_worker_n(1, persons_for_estimation)

@orca.column('persons_for_estimation', 'worker2', cache=True)
def worker2(persons_for_estimation):
    return is_worker_n(2, persons_for_estimation)

@orca.column('persons_for_estimation', 'workplace_zone_id', cache=True)
def workplace_zone_id(persons_for_estimation, jobs):
    return misc.reindex(jobs.zone_id, persons_for_estimation.job_id).fillna(-1)

@orca.column('persons_for_estimation', 'zone_id', cache=True)
def zone_id(persons_for_estimation, households_for_estimation):
    return misc.reindex(households_for_estimation.zone_id, persons_for_estimation.household_id)



