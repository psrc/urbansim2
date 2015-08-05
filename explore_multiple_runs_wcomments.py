import os
import pandana
import pandas as pd
from optparse import OptionParser
from urbansim.maps import dframe_explorer


# set data file to explore
#data_file = "conversion/out2010run_113.h5"
data_file = "conversion/run_142.run_2015_07_15_13_392041.h5"
#data_file2 = "conversion/run_133ref_with_school_models2041.h5" # for computing differences between runs
data_file2 = "conversion/run_138.run_2015_05_28_14_572041.h5"

comments_file = '/Users/hana/workspace/data/psrc_parcel/analyse_runs/Rreports/data/commentsLUV1tractcity.csv'

# Correspondence between the data and the shape files.
allgeo = {"zones": ("TAZ", "zone_id"),
          "parcels": ("NEW_USIMPI", "parcel_id"),
          "fazes": ("FAZ10", "faz_id"),
          "tractcity": ("ID", "tractcity_id")}

# Which tables will appear in the menu in the upper right corner.
common_tables = ['buildings', 'parcels', 'households', 'persons', 'jobs']
tables = {"zones": common_tables + ["zones"],
          "parcels": common_tables,
          "fazes": common_tables + ["zones", "fazes"],
          "tractcity": common_tables + ["tractcity"]}

# Which variables should be taken out of the second run
common_vars2 = ["number_of_households", "number_of_jobs"]
variables2 = {"parcels": common_vars2
              } 

# the eplorer will not work with text columns, therefore remove those
columns_to_remove = {
    "tractcity": ["juris_d", "tract", "trctjuris"]
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
       
    for tbl in ["zones", "fazes", "tractcity"]:
        if tbl in tables[geo]:
            variables2[tbl] = common_vars2 
        
    import psrc_urbansim.models
    import urbansim.sim.simulation as sim
    from psrc_urbansim.utils import change_store
    change_store(data_file)
    #import psrc_urbansim.accessibility.variables
    
    # create a dictionary of pandas frames
    d = {tbl: sim.get_table(tbl).to_frame() for tbl in tables[geo]}
    # remove unwanted columns
    for tbl, cols in columns_to_remove.iteritems():
        if tbl in d.keys():
            d[tbl].drop(cols, axis=1, inplace=True)
            
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

    # read comments and add them to the tractcity dataset
    if geo == 'tractcity':
        comments = pd.read_csv(comments_file, sep='\t', index_col="tractcity_id")
        d['tractcity']['comments_households'] = -1
        d['tractcity']['comments_jobs'] = -1
        where_hh_com = comments['HH_Comment_40'] != comments['HH_LUV_40']
        com_idx = comments.index[where_hh_com]
        d['tractcity']['comments_households'].loc[com_idx] = comments['HH_Comment_40'][where_hh_com]
        where_j_com = comments['Emp_Comment_40'] != comments['Emp_LUV_40']
        com_idx = comments.index[where_j_com] 
        d['tractcity']['comments_jobs'].loc[com_idx] = comments['Emp_Comment_40'][where_j_com]
        #d['tractcity']['comments_households'].loc[comments.index] = comments['HH_Comment_40']
        #d['tractcity']['comments_jobs'].loc[comments.index] = comments['Emp_Comment_40']
        
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