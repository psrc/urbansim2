import pandana as pdna
from pandana.loaders import osm
import os
import pandas as pd
from urbansim.utils import misc

def load_network(precompute=None, file_name='PugetSoundNetwork.h5'):
    # load OSM from hdf5 file
    store = pd.HDFStore(os.path.join(misc.data_dir(), file_name), "r")
    nodes = store.nodes
    edges = store.edges
    nodes.index.name = "index" # something that Synthicity wanted to fix
    # create the network
    net=pdna.Network(nodes["x"], nodes["y"], edges["from"], edges["to"], edges[["distance"]])
    if precompute is not None:
        for dist in precompute:
            net.precompute(dist)
    return net

def assign_nodes_to_dataset(dataset, network, x_name="long", y_name="lat"):
    """Adds an attribute node_ids to the given dataset."""
    x, y = dataset["long"], dataset["lat"]   
    # set attributes to nodes
    dataset["node_ids"] = network.get_node_ids(x, y)    
    
def x_node_intersections(waynodes, x=1):
    """
    Returns a set of all the x-node intersections.

    Parameters
    ----------
    waynodes : pandas.DataFrame
        Mapping of way IDs to node IDs as returned by `ways_in_bbox`.

    Returns
    -------
    intersections : set
        Node IDs that appear in x ways.

    """
    counts = waynodes.node_id.value_counts()
    return set(counts[counts == x].index.values)