import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

def is_worker_n(n, persons):
    return np.logical_and(persons.member_id == n, persons.job_id > 0)

#####################
# PERSONS VARIABLES (in alphabetic order)
#####################

@orca.column('persons', 'city_id', cache=True, cache_scope='step')
def city_id(persons, households):
    return misc.reindex(households.city_id, persons.household_id)

@orca.column('persons', 'subreg_id', cache=True, cache_scope='step')
def subreg_id(persons, households):
    return misc.reindex(households.subreg_id, persons.household_id)

@orca.column('persons', 'faz_id', cache=True, cache_scope='step')
def faz_id(persons, zones):
    return misc.reindex(zones.faz_id, persons.household_zone_id)

@orca.column('persons', 'household_building_id', cache=True, cache_scope='step')
def household_building_id(persons, households):
    return misc.reindex(households.building_id, persons.household_id)

@orca.column('persons', 'prev_household_building_id', cache=True, cache_scope='step')
def prev_household_building_id(persons, households):
    return misc.reindex(households.previous_building_id, persons.household_id)

@orca.column('persons', 'household_district_id', cache=False)
def household_district_id(persons, zones):
    return misc.reindex(zones.district_id, persons.household_zone_id)

@orca.column('persons', 'household_income_category', cache=True, cache_scope='step')
def household_income_category(persons):
    # calling households_for_estimation.income_category returns an non-indexed  
    # array, so converting to df for now. 
    df = orca.get_raw_table('households').to_frame(orca.get_raw_table('households').local_columns + ['income_category'])
    return misc.reindex(df.income_category, persons.household_id)

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
    res = misc.reindex(jobs.job_zone_id, persons.job_id)
    res = res.fillna(0)
    return res

@orca.column('persons', 'household_zone_id', cache=True)
def household_zone_id(persons, households):
    return misc.reindex(households.zone_id, persons.household_id)

@orca.column('persons', 'prev_household_zone_id', cache=True)
def prev_household_zone_id(persons, households):
    return misc.reindex(households.prev_zone_id, persons.household_id)




