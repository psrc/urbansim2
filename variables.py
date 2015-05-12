import pandas as pd
import numpy as np
from urbansim.utils import misc
import urbansim.sim.simulation as sim
import dataset
import urbansim_defaults.utils

#####################
# PARCELS VARIABLES
#####################

@sim.column('parcels', 'residential_units', cache=True, cache_scope='iteration')
def residential_units(parcels, buildings):
    return buildings.residential_units.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@sim.column('parcels', 'building_sqft', cache=True, cache_scope='iteration')
def building_sqft(parcels, buildings):
    return buildings.building_sqft.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@sim.column('parcels', 'total_improvement_value', cache=True, cache_scope='iteration')
def total_improvement_value(parcels, buildings):
    return buildings.improvement_value.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@sim.column('parcels', 'total_land_value_per_sqft', cache=True, cache_scope='iteration')
def total_land_value_per_sqft(parcels):
    return ((parcels.land_value + parcels.total_improvement_value)/parcels.parcel_sqft)

@sim.column('parcels', 'invfar', cache=True, cache_scope='iteration')
def invfar(parcels):
    return (parcels.parcel_sqft.astype(float)/parcels.building_sqft.astype(float)).replace(np.inf, 0)

@sim.column('parcels', 'average_income', cache=True, cache_scope='iteration')
def average_income(parcels, households):
    return households.income.groupby(households.parcel_id).mean().\
           reindex(parcels.index).fillna(0)


#####################
# BUILDINGS VARIABLES
#####################

@sim.column('buildings', 'building_sqft', cache=True, cache_scope='iteration')
def building_sqft(buildings):
    results = np.zeros(buildings.local.shape[0],dtype=np.int32)
    where_res = np.where(buildings.residential_units > 0)[0]
    results[where_res] = buildings.residential_units[where_res] * buildings.sqft_per_unit[where_res]
    where_nonres = np.where(buildings.non_residential_sqft > 0)[0]
    results[where_nonres] = results[where_nonres] + buildings.non_residential_sqft[where_nonres]
    return pd.Series(results)

@sim.column('buildings', 'zone_id', cache=True)
def zone_id(buildings, parcels):
    return misc.reindex(parcels.zone_id, buildings.parcel_id)

#####################
# HOUSEHOLDS VARIABLES
#####################

@sim.column('households', 'parcel_id', cache=True)
def parcel_id(households, buildings):
    return misc.reindex(buildings.parcel_id, households.building_id)

@sim.column('households', 'zone_id', cache=True)
def zone_id(households, buildings):
    return misc.reindex(buildings.zone_id, households.building_id)