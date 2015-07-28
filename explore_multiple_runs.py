import os
import pandana
import pandas as pd
from optparse import OptionParser
from urbansim.maps import dframe_explorer


# set data file to explore
#data_file = "conversion/out2010run_113.h5"
data_file = "conversion/run_142.run_2015_07_15_13_392041.h5"
data_file2 = "conversion/run_133ref_with_school_models2041.h5" # for computing differences between runs

# Correspondence between the data and the shape files.
allgeo = {"zones": ("TAZ", "zone_id"),
          "parcels": ("NEW_USIMPI", "parcel_id"),
          "fazes": ("FAZ10", "faz_id")}

# Which tables will appear in the menu in the upper right corner.
common_tables = ['buildings', 'parcels', 'households', 'persons', 'jobs']
tables = {"zones": common_tables + ["zones"],
          "parcels": common_tables,
          "fazes": common_tables + ["zones", "fazes"]}

# Which variables should be taken out of the second run
common_vars2 = ["number_of_households", "number_of_jobs"]
variables2 = {"parcels": common_vars2,
              "zones": common_vars2
              }   

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-g", "--geo", dest="geo",
                      help="Geography for the display, e.g. 'zones' (default) or 'fazes'", default="zones")
    parser.add_option("-p", "--port", dest="port", default=8765,
                      help="Port for the display. Default is 8765.")    
    (options, args) = parser.parse_args()    
    geo = options.geo
    port = options.port
       
    if "fazes" in tables[geo]:
        variables2["fazes"] = common_vars2  
        
    import psrc_urbansim.models
    import urbansim.sim.simulation as sim
    from psrc_urbansim.utils import change_store
    change_store(data_file)
    import psrc_urbansim.accessibility.variables
    
    # create a dictionary of pandas frames
    d = {tbl: sim.get_table(tbl).to_frame() for tbl in tables[geo]}
    change_store(data_file2)
    sim.clear_cache()
    #import urbansim.sim.simulation as sim
    #import psrc_urbansim.accessibility.variables
    for tbl in variables2.keys():
        d2 = sim.get_table(tbl).to_frame(columns=variables2[tbl])
        d2.columns = map(lambda x: x + "_2", d2.columns)
        d[tbl] = pd.merge(d[tbl], d2, left_index=True, right_index=True)

    # add the id column since the join does not work if the id is an index
    d[geo][allgeo[geo][1]] = d[geo].index.values

    dframe_explorer.start(d, 
                      center=[47.614848,-122.3359058],
                      zoom=11,
                      #shape_json=os.path.join(os.getenv("DATA_HOME", "."), 'data', '%s.geojson' % geo),
                      shape_json=os.path.join('data/', '%s.geojson' % geo),
                      geom_name=allgeo[geo][0], # from JSON file
                      join_name=allgeo[geo][1], # from data frames
                      precision=2, 
                      port=int(port)
                      )

# In the browser, the fifth field is a filter, e.g. number_of_households > 1000, 
# and the sixth field is a simple computation, e.g. number_of_jobs / 1000 
# I believe the attribute must be on the displayed geography (i.e. geo above). 