import models
import urbansim.sim.simulation as sim
from urbansim.maps import dframe_explorer
from utils import change_store

data_file = "conversion/out2010run_113.h5"
#data_file = "data/conversion/out2040run_133.run_2015_05_08_10_27.h5"

geo = "zones"
#geo = "parcels" # does not work (probably too big)
geo = "fazes"

allgeo = {"zones": ("TAZ", "zone_id"),
          "parcels": ("NEW_USIMPI", "parcel_id"),
          "fazes": ("FAZ10", "faz_id")}

change_store(data_file)
common_tables = ['buildings', 'parcels', 'households', 'persons', 'jobs']
tables = {"zones": common_tables + ["zones"],
          "parcels": common_tables,
          "fazes": common_tables + ["zones", "fazes"]}

d = {tbl: sim.get_table(tbl).to_frame() for tbl in tables[geo]}
# add the id column since the joint does not work if the id is an index
d[geo][allgeo[geo][1]] = d[geo].index.values

dframe_explorer.start(d, 
                      center=[47.614848,-122.3359058],
                      zoom=11,
                      #shape_json='data/parcels.geojson', geom_name='PARCEL_ID', join_name='parcel_id',
                      shape_json='data/%s.geojson' % geo,
                      geom_name=allgeo[geo][0], # from JSON file
                      join_name=allgeo[geo][1], # from data frames
                      precision=2, 
                      port=8765
                      #port=8766
                      )
