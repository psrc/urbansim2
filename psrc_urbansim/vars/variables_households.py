import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# HOUSEHOLDS VARIABLES
#####################

@orca.column('households', 'parcel_id', cache=True)
def parcel_id(households, buildings):
    return misc.reindex(buildings.parcel_id, households.building_id)

@orca.column('households', 'zone_id', cache=True)
def zone_id(households, buildings):
    return misc.reindex(buildings.zone_id, households.building_id)

@orca.column('households', 'faz_id', cache=True)
def faz_id(households, zones):
    return misc.reindex(zones.faz_id, households.zone_id)

@orca.column('households', 'tractcity_id', cache=True)
def tractcity_id(households, parcels):
    return misc.reindex(parcels.tractcity_id, households.parcel_id)

@orca.column('households', 'is_inmigrant', cache=True)
def is_inmigrant(households, parcels):
    return (households.building_id < 0).reindex(households.index)

@orca.column('households', 'building_type_id', cache=True)
# needed for relocation model
def building_type_id(households, buildings):
    return misc.reindex(buildings.building_type_id, households.building_id)
