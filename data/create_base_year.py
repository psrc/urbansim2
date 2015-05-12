import pandas as pd
import os

direct = "/Users/hana/workspace/data/psrc_parcel/datasets"
tables = {"buildings": "building_id",
          "parcels": "parcel_id",
          "zones": "zone_id",
          "households": "household_id"}


#store = pd.HDFStore('base_year.h5')
store = pd.HDFStore('base_year_with_ids.h5')
for table, tid in tables.iteritems():
    f = os.path.join(direct, table + ".csv")
    ds = pd.read_csv(f, index_col=tid)
    store[table] = ds
store.close()
