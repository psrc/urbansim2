# This script is obsolete. 
# To convert baseyear to hdf5, use conversion/cache_to_hdf5.py

import pandas as pd
import os

join_with_coordinates = True
direct = "/Users/hana/workspace/data/psrc_parcel/datasets"
tables = {"buildings": "building_id",
          "parcels": "parcel_id",
          "zones": "zone_id",
          "households": "household_id",
          "jobs": "job_id"}

parcels_with_coordinates = "parcels_for_google.csv"

store = pd.HDFStore('base_year.h5')
for table, tid in tables.iteritems():
    f = os.path.join(direct, table + ".csv")
    ds = pd.read_csv(f, index_col=tid)
    if join_with_coordinates and table == "parcels":
        ds_coor = pd.read_csv(os.path.join(direct, parcels_with_coordinates), index_col="parcel_id")
        ds = ds.merge(ds_coor.loc[:, ["lat", "long"]], how='left', left_index=True, right_index=True)
    store[table] = ds
store.close()
