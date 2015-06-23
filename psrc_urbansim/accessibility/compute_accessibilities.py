from utils import load_network, assign_nodes_to_dataset
import pandas as pd
import os
from urbansim.utils import misc
# this imports all the parcels variables we need
import variables
# this computes the variables
import urbansim.sim.simulation as sim

distances = [500, 1000, 3000]
attributes = ["number_of_households", "number_of_jobs"]
output_file = "parcel_accessibilities.h5"

net = load_network(precompute=distances)
parcels = sim.get_table("parcels")

assign_nodes_to_dataset(parcels, net)

outstore = pd.HDFStore(output_file)

for attr in attributes:
    net.set(parcels["node_ids"], variable=parcels[attr], name=attr)
    for dist in distances:
        res_name = "%s_%s" % (attr, dist)
        sim.add_column("parcels", res_name, net.aggregate(dist, type="sum", decay="linear", name=attr))

outstore["parcels"] = parcels.to_frame()
outstore.close()