import pandana as pdna
from psrc_urbansim.accessibility.utils import load_network, assign_nodes_to_dataset
import pandas as pd
import os
from urbansim.utils import misc
# this loads default datasets (default parcels needed since it has the right lat and long)
import psrc_urbansim.dataset
# this imports all the parcels variables we need
#import psrc_urbansim.variables
# this computes the variables
import urbansim.sim.simulation as sim

distances = [ # in meters
    804.67, # 0.5 mile
    1609.34 # 1 mile
             ]
#attributes = {"sum":["number_of_households", "number_of_jobs"]}
attributes = {"sum": ["hh_p", "stugrd_p", "stuhgh_p", "stuuni_p", "empedu_p", "empfoo_p", "empgov_p", "empind_p", "empsvc_p", "empoth_p", "emptot_p",
                      "parkdy_p", "parkhr_p"],
              "ave": [ "ppricdyp", "pprichrp"],
              "count":["node_ids"]
              }
output_file = "parcel_accessibilities_soundcast.h5"

# get input parcel data
instore = pd.HDFStore('/Users/hana/workspace/data/soundcast/urbansim_outputs/2040/parcels.h5', "r")
parcels = instore["parcels"]
# merge in latitude and longitude columns 
parcels_with_lat_long = sim.get_table("parcels").to_frame(['lat', 'long'])
parcels = pd.merge(parcels, parcels_with_lat_long, left_index=True, right_index=True)

# load network and assign parcels to the network 
net = load_network(precompute=distances)
assign_nodes_to_dataset(parcels, net)


for fun, attrs in attributes.iteritems():
    for attr in attrs:
        print "Processing %s" % attr
        net.set(parcels["node_ids"], variable=parcels[attr], name=attr)
        for dist in distances:        
            res_name = "%s_%s" % (attr, dist)
            parcels[res_name] = net.aggregate(dist, type=fun, decay="linear", name=attr)

outstore = pd.HDFStore(output_file)        
outstore["parcels"] = parcels
outstore.close()

