from pandana.loaders import osm
from psrc_urbansim.accessibility.utils import x_node_intersections, load_network
import pandas as pd 
import numpy as np

def create_osm_base_network(outputfile, bbox, network_type="drive"):
    network = osm.network_from_bbox(*bbox, network_type=network_type)
    network.save_hdf5(outputfile)
    return network

def add_feature(outputfile, bbox, network, tags, colname, mode='a'):
    all_nodes = []
    for tag in tags:
        nodes = osm.node_query(*bbox, tags=tag)
        all_nodes = all_nodes + [nodes.loc[:,['lat', 'lon']]]
    all_nodes = pd.concat(all_nodes)    
    all_nodes_unique = all_nodes.drop_duplicates()
    node_ids = network.get_node_ids(all_nodes_unique['lon'], all_nodes_unique['lat'])
    all_nodes_unique["node_id"] = node_ids.values    
    with pd.HDFStore(outputfile, mode=mode) as store:
        store[colname] = all_nodes_unique
        #store[colname] = pd.DataFrame({colname: np.ones(node_ids.size, dtype='bool8')}, index=node_ids.values)
        #store[colname] = pd.DataFrame({colname: np.ones(all_nodes_unique.index.size, dtype='bool8')}, index=all_nodes_unique.index.values)
        
def add_bus_stops(outputfile, bbox, network, mode='a'):
    add_feature(outputfile, bbox, network, 
                tags=['"public_transport"="station"', '"amenity"="bus_station"', '"highway"="bus_stop"',
                      ['"public_transport"="stop_position"', '"bus"="yes"']],
                colname="busstops", mode=mode) # finds 1836 stops

def add_railway(outputfile, bbox, network, mode='a'):
    add_feature(outputfile, bbox, network, 
                tags=[['"public_transport"="stop_position"', '"train"="yes"'], '"railway"="station"'],
                colname="railway", mode=mode)
    
def add_ferry(outputfile, bbox, network, mode='a'):
    add_feature(outputfile, bbox, network, 
                tags=[['"public_transport"="stop_position"', '"ferry"="yes"']],
                colname="ferry", mode=mode)
    
def add_lightrail(outputfile, bbox, network, mode='a'):
    add_feature(outputfile, bbox, network, 
                tags=[['"public_transport"="stop_position"', '"train"="yes"'], '"railway"="station"'],
                colname="lightrail", mode=mode)
    
def add_intersections(outputfile, bbox, network, network_type="drive", mode='a'):
    nodes, ways, waynodes = osm.ways_in_bbox(*bbox, network_type=network_type)
    waynodes = waynodes[np.in1d(waynodes["node_id"], network.node_ids)]
    intersections = x_node_intersections(waynodes, x=[1, 3, 4], last_open=True)
    intersections["node_id"] = intersections.index.values
    with pd.HDFStore(outputfile, mode=mode) as store:
        store['intersections'] = intersections
        

    
        
def create_network_addons(outputfile, bbox, network, network_type="drive"):
    add_bus_stops(outputfile, bbox, network=network, mode='w')
    add_intersections(outputfile, bbox, network=network, network_type=network_type)
    add_railway(outputfile, bbox, network=network)
    add_ferry(outputfile, bbox, network=network)
    add_lightrail(outputfile, bbox, network=network)
    
    
    
filename = 'PugetSoundNetwork'
network_type = "drive"
bbox = (46.73, -123.03, 48.3,  -120.95)
net = None
#net = create_osm_base_network("%s.h5" % filename, bbox, network_type=network_type)
if net is None:
    net = load_network()
create_network_addons("%sAddons.h5" % filename, bbox, network=net, network_type=network_type)
