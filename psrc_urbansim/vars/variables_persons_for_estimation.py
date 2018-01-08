import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

def is_worker_n(n, persons_for_estimation):
    return np.logical_and(persons_for_estimation.member_id == n, persons_for_estimation.job_id > 0)

#####################
# persons_for_estimation VARIABLES (in alphabetic order)
#####################

@orca.column('persons_for_estimation', 'faz_id', cache=True)
def faz_id(persons_for_estimation, zones):
    return misc.reindex(zones.faz_id, persons_for_estimation.zone_id)

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
def workplace_zone_id(persons_for_estimation, jobs_for_estimation):
    return misc.reindex(jobs_for_estimation.zone_id, persons_for_estimation.job_id)

@orca.column('persons_for_estimation', 'zone_id', cache=True)
def zone_id(persons_for_estimation, households_for_estimation):
    return misc.reindex(households_for_estimation.zone_id, persons_for_estimation.household_id)



