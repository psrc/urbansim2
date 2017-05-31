import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_parcels.utils

def is_worker_n(n, persons):
    return np.logical_and(persons.member_id == n, persons.job_id > 0)

#####################
# PERSONS VARIABLES (in alphabetic order)
#####################

@orca.column('persons', 'faz_id', cache=True)
def faz_id(persons, zones):
    return misc.reindex(zones.faz_id, persons.zone_id)

@orca.column('persons', 'parcel_id', cache=True)
def parcel_id(persons, households):
    return misc.reindex(households.parcel_id, persons.household_id)

@orca.column('persons', 'tractcity_id', cache=True)
def tractcity_id(persons, households):
    return misc.reindex(households.tractcity_id, persons.household_id)

@orca.column('persons', 'worker1', cache=True)
def worker1(persons):
    return is_worker_n(1, persons)

@orca.column('persons', 'worker2', cache=True)
def worker2(persons):
    return is_worker_n(2, persons)

@orca.column('persons', 'workplace_zone_id', cache=True)
def workplace_zone_id(persons, jobs):
    return misc.reindex(jobs.zone_id, persons.job_id)

@orca.column('persons', 'zone_id', cache=True)
def zone_id(persons, households):
    return misc.reindex(households.zone_id, persons.household_id)



