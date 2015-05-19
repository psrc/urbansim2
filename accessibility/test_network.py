import pandas as pd
import pandana as pdna
import os
from urbansim.utils import misc

# load from hdf5 file
store = pd.HDFStore(os.path.join(misc.data_dir(),'PugetSoundNetwork.h5'), "r")
storeUS = pd.HDFStore(os.path.join(misc.data_dir(),'base_year.h5'), "r")

nodes = store.nodes
edges = store.edges

print nodes.head(3)
print edges.head(3)

# create network
net=pdna.Network(nodes["x"], nodes["y"], edges["from"], edges["to"], edges[["distance"]])

# precompute aggregations, e.g. 3000m radius
net.precompute(3000)

#buildings = storeUS.buildings

parcels = storeUS.parcels
x, y = parcels.x_coord_sp, parcels.y_coord_sp
x.index.name = 'index'
y.index.name = 'index'

parcels["node_ids"] = net.get_node_ids(x, y)