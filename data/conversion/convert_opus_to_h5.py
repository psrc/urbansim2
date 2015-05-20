import os
from opus_core.datasets.pandas_dataset import PandasDataset
from opus_core.simulation_state import SimulationState
from opus_core.store.attribute_cache import AttributeCache

execfile('inputs.py')
output_file = "testout%s%s.h5" % (year, run)
storage = AttributeCache(base_directory)
SimulationState().set_current_time(year)

first_table = True
# convert tables into csv using opus
for table, tid in tables.iteritems():
    pds = PandasDataset(in_storage=storage, in_table_name=table, id_name=tid)
    pds.df.to_hdf(output_file, table, append=first_table == False)
    first_table = False

