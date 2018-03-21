import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# HOUSEHOLDS VARIABLES (in alphabetic order)
#####################

@orca.column('households', 'building_type_id', cache=True)
def building_type_id(households, buildings):
    return misc.reindex(buildings.building_type_id, households.building_id)

#@orca.column('households_lag1', 'building_type_id', cache=True)
#def building_type_id(households_lag1, buildings_lag1):
#    return misc.reindex(buildings_lag1.building_type_id, households_lag1.building_id)

@orca.column('households', 'city_id', cache=True)
def city_id(households, parcels):
    return misc.reindex(parcels.city_id, households.parcel_id)

@orca.column('households', 'faz_id', cache=True)
def faz_id(households, zones):
    return misc.reindex(zones.faz_id, households.zone_id)

@orca.column('households', 'grid_id', cache=True)
def grid_id(households, parcels):
    return misc.reindex(parcels.grid_id, households.parcel_id)

@orca.column('households', 'income_category', cache=True)
def income_category(households, settings):
    income_breaks = settings.get('income_breaks', [34000, 64000, 102000])
    res = households.income.values
    res[:] = np.nan
    res[(households.income < income_breaks[0]).values] = 1
    res[(np.logical_and(households.income >= income_breaks[0], households.income < income_breaks[1])).values] = 2
    res[(np.logical_and(households.income >= income_breaks[1], households.income < income_breaks[2])).values] = 3
    res[(households.income >= income_breaks[2]).values] = 4
    return res

@orca.column('households', 'is_inmigrant', cache=True)
def is_inmigrant(households, parcels):
    return (households.building_id < 0).reindex(households.index)

@orca.column('households', 'is_residence_mf', cache=True)
def is_residence_mf(households, buildings):
    return misc.reindex(buildings.multifamily_generic_type, households.building_id).fillna(-1)

@orca.column('households', 'parcel_id', cache=True)
def parcel_id(households, buildings):
    return misc.reindex(buildings.parcel_id, households.building_id)

@orca.column('households', 'residence_large_area', cache=True)
def residence_large_area(households, buildings):
    return misc.reindex(buildings.large_area_id, households.building_id).fillna(-1)


#@orca.column('households', 'same_building_type', cache=True)
#def same_building_type(households, households_lag1):
#    merged = households.building_type_id.to_frame('bt').join(households_lag1.building_type_id.to_frame('btlag'))
#    return merged.bt == merged.btlag

@orca.column('households', 'tractcity_id', cache=True)
def tractcity_id(households, parcels):
    return misc.reindex(parcels.tractcity_id, households.parcel_id)

@orca.column('households', 'worker1_zone_id', cache=True)
def worker1_zone_id(households, persons):
    return misc.reindex(persons.workplace_zone_id[persons.worker1==True], households.building_id).fillna(-1)

@orca.column('households', 'worker2_zone_id', cache=True)
def worker2_zone_id(households, persons):
    return misc.reindex(persons.workplace_zone_id[persons.worker2==True], households.building_id).fillna(-1)

@orca.column('households', 'work_zone_id', cache=True)
def work_zone_id(households, buildings):
    return misc.reindex(buildings.zone_id, households.building_id).fillna(-1)

@orca.column('households', 'zone_id', cache=True)
def zone_id(households, buildings):
    return misc.reindex(buildings.zone_id, households.building_id).fillna(-1)








