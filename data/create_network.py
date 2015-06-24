from pandana.loaders import osm
import pandas as pd

filename = 'PugetSoundNetwork.h5'

#network = osm.network_from_bbox(46.73, -123.03, 48.3,  -120.95, network_type='drive')
#network.save_hdf5(filename)

all_nodes = []
for tag in ['"public_transport"="station"', '"amenity"="bus_station"', '"highway"="bus_stop"']:
    nodes_bus = osm.node_query(46.73, -123.03, 48.3,  -120.95, 
                             tags=tag)
    all_nodes = all_nodes + [nodes_bus.loc[:,['lat', 'lon']]]

all_nodes = pd.concat(all_nodes)    
all_nodes_unique = all_nodes.drop_duplicates(all_nodes) # returns 1836 stops

with pd.HDFStore(filename, mode='a') as store:
    store['stations'] = all_nodes_unique
    
