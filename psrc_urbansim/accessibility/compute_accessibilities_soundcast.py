import pandana as pdna
from psrc_urbansim.accessibility.utils import load_network, load_network_addons, assign_nodes_to_dataset
import pandas as pd
import numpy as np
import os
import re
from urbansim.utils import misc
# this loads default datasets (default parcels needed since it has the right lat and long)
import psrc_urbansim.dataset
# this imports all the parcels variables we need
#import psrc_urbansim.variables
# this computes the variables
import urbansim.sim.simulation as sim

distances = { # in meters
    1: 804.67, # 0.5 mile
    2: 1609.34 # 1 mile
             }

# These are disaggregated to the network from the parcel data
parcel_attributes = {"sum": ["hh_p", "stugrd_p", "stuhgh_p", "stuuni_p", "empedu_p", "empfoo_p", "empgov_p", "empind_p", 
                      "empsvc_p", "empoth_p", "emptot_p", "parkdy_p", "parkhr_p", "nparks", "aparks"],
              "ave": [ "ppricdyp", "pprichrp"],
              }
# These are already on network (from add-ons)
network_attributes = {"tstops": "busstops"}
intersections = {"nodes1": "1-way", "nodes3": "3-way", "nodes4": "4-way"}

pois = {"lbus": "busstops", "ebus": "busstops", 
        "fry": "ferry", "crt": "railway", "lrt": "lightrail"} # will compute nearest distance to these

output_file = "parcel_accessibilities_soundcast.h5"

# get input parcel data
instore = pd.HDFStore('/Users/hana/workspace/data/soundcast/urbansim_outputs/2040/parcels.h5', "r")
parcels = instore["parcels"]
# merge in latitude and longitude columns 
parcels_with_lat_long = sim.get_table("parcels").to_frame(['lat', 'long'])
parcels = pd.merge(parcels, parcels_with_lat_long, left_index=True, right_index=True)

# load network and assign parcels to the network 
net = load_network(precompute=distances)
load_network_addons(network=net)
assign_nodes_to_dataset(parcels, net)

def process_net_attribute(network, attr, fun):
    print "Processing %s" % attr
    newdf = None
    for dist_index, dist in distances.iteritems():        
        res_name = "%s_%s" % (re.sub("_?p$", "", attr), dist_index) # remove '_p' if present
        aggr = network.aggregate(dist, type=fun, decay="linear", name=attr)
        if newdf is None:
            newdf = pd.DataFrame({res_name: aggr, "node_ids": aggr.index.values})
        else:
            newdf[res_name] = aggr
    return newdf
    
newdf = None
for fun, attrs in parcel_attributes.iteritems():    
    for attr in attrs:
        net.set(parcels["node_ids"], variable=parcels[attr], name=attr)
        res = process_net_attribute(net, attr, fun)
        if newdf is None:
            newdf = res
        else:
            newdf = pd.merge(newdf, res, on="node_ids", copy=False)


for new_name, attr in network_attributes.iteritems():    
    net.set(net.node_ids, variable=net.addons[attr]["has_poi"].values, name=new_name)
    newdf = pd.merge(newdf, process_net_attribute(net, new_name, "sum"), on="node_ids", copy=False)
    
for new_name, attr in intersections.iteritems():
    #tmp = pd.DataFrame({"node_ids": net.addons["intersections"][attr].index.values,
    #                    "has_poi": net.addons["intersections"][attr].values})
    #intersections_wparcels = pd.merge(parcels.loc[:,['parcelid', 'node_ids']], tmp, how='left', on="node_ids").set_index('parcelid')
    #net.set(intersections_wparcels["node_ids"], variable=intersections_wparcels["has_poi"], name=new_name)
    net.set(net.node_ids, variable=net.addons["intersections"][attr].values, name=new_name)
    newdf = pd.merge(newdf, process_net_attribute(net, new_name, "sum"), on="node_ids", copy=False)
   
parcels = pd.merge(parcels, newdf, on="node_ids", copy=False)

# nearest distance variables
max_dist = 4828 # 3 miles in meters
net.init_pois(len(pois)+1, max_dist, 1)

def process_dist_attribute(network, name, x, y):
    network.set_pois(name, x, y)
    res = network.nearest_pois(max_dist, name, num_pois=1, max_distance=999)
    res[res <> 999] = (res[res <> 999]/1000. * 0.621371).astype(res.dtypes) # convert to miles
    res_name = "dist_%s" % name
    parcels[res_name] = res.loc[parcels["node_ids"]].values
    
for new_name, attr in pois.iteritems():
    process_dist_attribute(net, new_name, net.addons[attr]["lon"], net.addons[attr]["lat"])
        
# distance to park
parcel_idx_park = np.where(parcels['nparks'] > 0)[0]
process_dist_attribute(net, "park", parcels["long"][parcel_idx_park], parcels["lat"][parcel_idx_park])


outstore = pd.HDFStore(output_file)        
outstore["parcels"] = parcels
outstore.close()

