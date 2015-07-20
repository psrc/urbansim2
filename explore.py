import pandana
import psrc_urbansim.models
import urbansim.sim.simulation as sim
from urbansim.maps import dframe_explorer
from psrc_urbansim.utils import change_store

# set data file to explore
data_file = "conversion/out2010run_113.h5"
data_file = "conversion/run_133ref_with_school_models2041.h5"
data_file = "conversion/run_142.run_2015_07_15_13_392041.h5"

# geography for the display
geo = "zones" 
#geo = "parcels" # does not work (probably too big)
#geo = "fazes"

allgeo = {"zones": ("TAZ", "zone_id"),
          "parcels": ("NEW_USIMPI", "parcel_id"),
          "fazes": ("FAZ10", "faz_id")}

change_store(data_file)
common_tables = ['buildings', 'parcels', 'households', 'persons', 'jobs']
tables = {"zones": common_tables + ["zones"],
          "parcels": common_tables,
          "fazes": common_tables + ["zones", "fazes"]}

import psrc_urbansim.accessibility.variables


# create a dictionary of pandas frames
d = {tbl: sim.get_table(tbl).to_frame() for tbl in tables[geo]}
# add the id column since the join does not work if the id is an index
d[geo][allgeo[geo][1]] = d[geo].index.values

dframe_explorer.start(d, 
                      center=[47.614848,-122.3359058],
                      zoom=11,
                      shape_json='data/%s.geojson' % geo,
                      geom_name=allgeo[geo][0], # from JSON file
                      join_name=allgeo[geo][1], # from data frames
                      precision=2, 
                      port=8765
                      #port=8766
                      )

# In the browser, the fifth field is a filter, e.g. number_of_households > 1000, 
# and the sixth field is a simple computation, e.g. number_of_jobs / 1000 
# I believe the attribute must be on the displayed geography (i.e. geo above). 