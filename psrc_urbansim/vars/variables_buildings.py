import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# BUILDINGS VARIABLES
#####################

@orca.column('buildings', 'building_sqft', cache=True, cache_scope='iteration')
def building_sqft(buildings):
    results = np.zeros(buildings.local.shape[0],dtype=np.int32)
    where_res = np.where(buildings.residential_units > 0)[0]
    results[where_res] = buildings.residential_units[where_res] * buildings.sqft_per_unit[where_res]
    where_nonres = np.where(buildings.non_residential_sqft > 0)[0]
    results[where_nonres] = results[where_nonres] + buildings.non_residential_sqft[where_nonres]
    return pd.Series(results)

@orca.column('buildings', 'zone_id', cache=True)
def zone_id(buildings, parcels):
    return misc.reindex(parcels.zone_id, buildings.parcel_id)

@orca.column('buildings', 'faz_id', cache=True)
def faz_id(buildings, zones):
    return misc.reindex(zones.faz_id, buildings.zone_id)

@orca.column('buildings', 'tractcity_id', cache=True)
def tractcity_id(buildings, parcels):
    return misc.reindex(parcels.tractcity_id, buildings.parcel_id)

