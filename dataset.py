import pandas as pd
import assumptions
import urbansim_defaults.utils
import urbansim.sim.simulation as sim

import warnings
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)


@sim.table('parcels', cache=True)
def parcels(store):
    df = store['parcels']
    return df

@sim.table('buildings', cache=True)
def buildings(store):
    df = store['buildings']
    return df

@sim.table('households', cache=True)
def households(store):
    df = store['households']
    return df

@sim.table('zones', cache=True)
def zones(store):
    df = store['zones']
    return df

sim.broadcast('parcels', 'buildings', cast_index=True, onto_on='parcel_id')
sim.broadcast('households', 'buildings', cast_index=True, onto_on='building_id')
sim.broadcast('zones', 'parcels', cast_index=True, onto_on='zone_id')