# Run this file with Python Opus settings.
# Set all the inputs in inputs.py

import os
import pandas as pd
from opus_core.datasets.pandas_dataset import PandasDataset
from opus_core.simulation_state import SimulationState
from opus_core.store.attribute_cache import AttributeCache


execfile('inputs.py')
output_file = "out%s%s.h5" % (year, run)
output_file = "%s%s.h5" % (run, year)
storage = AttributeCache(base_directory)
SimulationState().set_current_time(year)

first_table = True
# convert tables into h5 using opus
for table, tid in tables.iteritems():
    pds = PandasDataset(in_storage=storage, in_table_name=table, id_name=tid)
    if join_with_coordinates and table == "parcels":
        ds_coor = pd.read_csv(os.path.join(dir_with_coordinates, parcels_with_coordinates), index_col="parcel_id")
        pds.df = pds.df.merge(ds_coor.loc[:, ["lat", "long"]], how='left', left_index=True, right_index=True)    
    pds.df.to_hdf(output_file, table, append=first_table == False)
    first_table = False

