from pandana.loaders import osm
network = osm.network_from_bbox(47.139330, -122.729001, 48.054254,  -121.905026, network_type='drive')
network.save_hdf5('PugetSoundNetwork.h5')
