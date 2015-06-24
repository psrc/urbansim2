import pandana as pdna
import pandas as pd
import os
from urbansim.utils import misc
import psrc_urbansim.variables
import urbansim.sim.simulation as sim


# load from hdf5 file
store = pd.HDFStore(os.path.join(misc.data_dir(),'PugetSoundNetwork.h5'), "r")

nodes = store.nodes
edges = store.edges

print nodes.head(3)
print edges.head(3)

#nodes["x"].index.name = "index"
#nodes["y"].index.name = "index"
nodes.index.name = "index"

# create network
net=pdna.Network(nodes["x"], nodes["y"], edges["from"], edges["to"], edges[["distance"]])



# precompute aggregations, e.g. 3000m radius
net.precompute(3000)

#change_store('base_year.h5')
parcels = sim.get_table("parcels")
x, y = parcels["long"], parcels["lat"]

# set attributes to nodes
parcels["node_ids"] = net.get_node_ids(x, y)

net.set(parcels["node_ids"], variable=parcels.land_value, name="land_value")
net.set(parcels["node_ids"], variable=parcels.number_of_households, name="number_of_households")

# query results
lv = net.aggregate(500, type="ave", decay="linear", name="land_value")
du = net.aggregate(500, type="sum", decay="linear", name="number_of_households")

# plot results
bbox = [47.0, -122.0, 48.0,  -121.0]
p = net.plot(s, bbox=bbox)
import matplotlib.pyplot as plt
plt.show()