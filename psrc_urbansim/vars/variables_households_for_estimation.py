import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils
import psrc_urbansim.vars.variables_interactions 
#import avg_network_distance_from_home_to_work
#####################
# households_for_estimation VARIABLES (in alphabetic order)
#####################

@orca.column('households_for_estimation', 'building_type_id', cache=True)
def building_type_id(households_for_estimation, buildings):
    return misc.reindex(buildings.building_type_id, households_for_estimation.building_id)

@orca.column('households_for_estimation', 'city_id', cache=True)
def city_id(households_for_estimation, parcels):
    return misc.reindex(parcels.city_id, households_for_estimation.parcel_id)

@orca.column('households_for_estimation', 'faz_id', cache=True)
def faz_id(households_for_estimation, zones):
    return misc.reindex(zones.faz_id, households_for_estimation.zone_id)

@orca.column('households_for_estimation', 'grid_id', cache=True)
def grid_id(households_for_estimation, parcels):
    return misc.reindex(parcels.grid_id, households_for_estimation.parcel_id)

@orca.column('households_for_estimation', 'income_category', cache=True)
def income_category(households_for_estimation, settings):
    income_breaks = settings.get('income_breaks', [34000, 64000, 102000])
    res = households_for_estimation.income.values
    res[:] = np.nan
    res[(households_for_estimation.income < income_breaks[0]).values] = 1
    res[(np.logical_and(households_for_estimation.income >= income_breaks[0], households_for_estimation.income < income_breaks[1])).values] = 2
    res[(np.logical_and(households_for_estimation.income >= income_breaks[1], households_for_estimation.income < income_breaks[2])).values] = 3
    res[(households_for_estimation.income >= income_breaks[2]).values] = 4
    return res

@orca.column('households_for_estimation', 'is_residence_mf', cache=True)
def is_residence_mf(households_for_estimation, buildings):
    return misc.reindex(buildings.multifamily_generic_type, households_for_estimation.building_id).fillna(-1)

@orca.column('households_for_estimation', 'parcel_id', cache=True)
def parcel_id(households_for_estimation, buildings):
    return misc.reindex(buildings.parcel_id, households_for_estimation.building_id)

@orca.column('households_for_estimation', 'residence_large_area', cache=True)
def residence_large_area(households_for_estimation, buildings):
    return misc.reindex(buildings.large_area_id, households_for_estimation.building_id).fillna(-1)

@orca.column('households_for_estimation', 'tractcity_id', cache=True)
def tractcity_id(households_for_estimation, parcels):
    return misc.reindex(parcels.tractcity_id, households_for_estimation.parcel_id)

@orca.column('households_for_estimation', 'worker1_zone_id', cache=True)
def worker1_zone_id(households_for_estimation, persons_for_estimation):
    return((persons_for_estimation.worker1)*(persons_for_estimation.workplace_zone_id)).\
            groupby(persons_for_estimation.household_id).sum().\
            reindex(households_for_estimation.index).fillna(-1).replace(0, -1)

@orca.column('households_for_estimation', 'worker2_zone_id', cache=True)
def worker2_zone_id(households_for_estimation, persons_for_estimation):
    return((persons_for_estimation.worker2)*(persons_for_estimation.workplace_zone_id)).\
            groupby(persons_for_estimation.household_id).sum().\
            reindex(households_for_estimation.index).fillna(-1).replace(0, -1)

@orca.column('households_for_estimation', 'work_zone_id', cache=True)
def work_zone_id(households_for_estimation, buildings):
    return misc.reindex(buildings.zone_id, households_for_estimation.building_id).fillna(-1)

@orca.column('households_for_estimation', 'zone_id', cache=True)
def zone_id(households_for_estimation, buildings):
    return misc.reindex(buildings.zone_id, households_for_estimation.building_id).fillna(-1)

@orca.column('households_for_estimation', 'prev_residence_is_mf', cache=True)
def prev_residence_is_mf(households_for_estimation, buildings_lag1):
    return misc.reindex(buildings_lag1.multifamily_generic_type, households_for_estimation.previous_building_id).fillna(-1)

@orca.column('households_for_estimation', 'prev_residence_large_area_id', cache=True)
def prev_residence_large_area_id(households_for_estimation, buildings_lag1):
    return misc.reindex(buildings_lag1.large_area_id, households_for_estimation.previous_building_id).fillna(-1)

@orca.column('households_for_estimation', 'persons_under_13', cache=True)
def persons_under_13(households_for_estimation, persons_for_estimation):
    df = persons_for_estimation.local[persons_for_estimation.local.age < 13]
    return df.groupby(df.household_id).age.count().reindex(households_for_estimation.index).fillna(0)

