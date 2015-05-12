import models
import urbansim.sim.simulation as sim
from urbansim.maps import dframe_explorer

geo = "zone"
geo = "parcel"
allgeo = {"zone": "TAZ",
          "parcel":"NEW_USIMPI"}

common_tables = ['buildings', 'parcels', 'households']
tables = {"zone": common_tables + ["zones"],
          "parcel": common_tables}
d = {tbl: sim.get_table(tbl).to_frame() for tbl in tables[geo]}
# add the id column since 
d['%ss' % geo]['%s_id' % geo] = d['%ss' % geo].index.values

dframe_explorer.start(d, 
                      center=[47.614848,-122.3359058],
                      zoom=11,
                      #shape_json='data/parcels.geojson', geom_name='PARCEL_ID', join_name='parcel_id',
                      shape_json='data/%ss.geojson' % geo,
                      geom_name=allgeo[geo], # from JSON file
                      join_name='%s_id'% geo, # from data frames
                      precision=2, 
                      port=8765
                      #port=8766
                      )
#df.groupby('zone_id')['income'].mean()
#df.groupby('zone_id')['income'].quantile(.5)