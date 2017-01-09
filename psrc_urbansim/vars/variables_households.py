import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# HOUSEHOLDS VARIABLES (in alphabetic order)
#####################

@orca.column('households', 'building_type_id', cache=True)
# needed for relocation model
def building_type_id(households, buildings):
    return misc.reindex(buildings.building_type_id, households.building_id)

@orca.column('households', 'city_id', cache=True)
def city_id(households, parcels):
    return misc.reindex(parcels.city_id, households.parcel_id)

@orca.column('households', 'faz_id', cache=True)
def faz_id(households, zones):
    return misc.reindex(zones.faz_id, households.zone_id)

@orca.column('households', 'grid_id', cache=True)
def grid_id(households, parcels):
    return misc.reindex(parcels.grid_id, households.parcel_id)

@orca.column('households', 'is_inmigrant', cache=True)
def is_inmigrant(households, parcels):
    return (households.building_id < 0).reindex(households.index)

@orca.column('households', 'parcel_id', cache=True)
def parcel_id(households, buildings):
    return misc.reindex(buildings.parcel_id, households.building_id)

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








