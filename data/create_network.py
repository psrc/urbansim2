from pandana.loaders import osm
network = osm.network_from_bbox(46.73, -123.03, 48.3,  -120.95, network_type='drive')
network.save_hdf5('PugetSoundNetwork.h5')
