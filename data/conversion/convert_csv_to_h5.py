import os
import pandas as pd

execfile('inputs.py')
    
# convert csv into hdf5 using US2
store = pd.HDFStore(output_file)
for table, tid in tables.iteritems():
    f = os.path.join(out_directory, table + ".csv")
    ds = pd.read_csv(f, index_col=tid)
    store[table] = ds
store.close()

    
