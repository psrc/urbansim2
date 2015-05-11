import pandas as pd
import numpy as np
from urbansim.utils import misc
import urbansim.sim.simulation as sim
import dataset
import urbansim_defaults.utils

@sim.column('parcels', 'residential_units', cache=True, cache_scope='iteration')
def residential_units(parcels, buildings):
    return buildings.residential_units.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)


@sim.column('parcels', 'total_improvement_value', cache=True, cache_scope='iteration')
def total_improvement_value(parcels, buildings):
    return buildings.improvement_value.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)


@sim.column('parcels', 'total_land_value_per_sqft', cache=True, cache_scope='iteration')
def total_land_value_per_sqft(parcels, buildings):
    return ((parcels.land_value + parcels.total_improvement_value)/parcels.parcel_sqft).fillna(0)