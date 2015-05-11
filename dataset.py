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

sim.broadcast('parcels', 'buildings', cast_index=True, onto_on='parcel_id')