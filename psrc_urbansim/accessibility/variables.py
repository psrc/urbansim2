import urbansim.sim.simulation as sim
from psrc_urbansim.accessibility.utils import load_network, assign_nodes_to_dataset
import pandas as pd
import numpy as np
import os
from urbansim.utils import misc

distances = {"half.mile": 804.67, 
             "mile": 1609.34} 
max_dist = 4828 # 3 miles in meters
net = load_network(precompute=distances.values())
net.init_pois(10, max_dist, 1)

@sim.column('parcels', 'households_within_half_mile', cache=True, cache_scope='iteration')
def households_within_half_mile(parcels):
    if "node_ids" not in parcels.columns:
        assign_nodes_to_dataset(parcels, net)
    attr = "number_of_households"
    net.set(parcels.node_ids, variable=parcels[attr], name=attr)
    net_aggr = net.aggregate(distances["half.mile"], type="sum", decay="linear", name=attr)
    #df = pd.DataFrame({attr: net_aggr, "node_ids": net_aggr.index.values})
    #return pd.merge(parcels.to_frame(columns=["node_ids"]), df, on="node_ids")[attr]
    return net_aggr.loc[parcels.node_ids].values


@sim.column('parcels', 'distance_to_park', cache=True, cache_scope='iteration')
def distance_to_park(parcels):
    if "node_ids" not in parcels.columns:
        assign_nodes_to_dataset(parcels, net)    
    parcel_idx_park = np.where(parcels.is_park)[0]
    return process_dist_attribute("park", parcels.long[parcel_idx_park], parcels.lat[parcel_idx_park], node_ids=parcels.node_ids)
    
    
def process_dist_attribute(name, x, y, node_ids, default_value=9999, convert_to_miles=True):
    net.set_pois(name, x, y)
    res = net.nearest_pois(max_dist, name, num_pois=1, max_distance=default_value)
    if convert_to_miles:
        res[res <> default_value] = (res[res <> default_value]/1000. * 0.621371).astype(res.dtypes) # convert meters to miles
    return res.loc[node_ids].values.astype('float32')