import os
import orca
import pandas as pd
from urbansim.utils import misc

@orca.injectable('store', cache=True)
def store(settings):
    return pd.HDFStore(os.path.join(os.getenv('DATA_HOME'), settings["store"]), mode='r')

@orca.injectable()
def base_year(settings):
    return settings['base_year']

@orca.injectable()
def year(iter_var):
    return iter_var

@orca.injectable()
def fileyear(year, base_year):
    if year == base_year:
        return "base"
    return year

@orca.injectable('store_table_list', cache=True)
def store_table_list(store):
    return store.keys()

@orca.injectable()
def find_table_in_store(table, store, fileyear, base_year):
    searchyear = fileyear
    if searchyear == "base":
        return store['%s/%s' % (fileyear, table)]
    else:
        while searchyear > base_year:
            if (('%s/%s' % (searchyear, table)) in store_table_list):
                return store['%s/%s' % (searchyear, table)]
            else:
                searchyear = searchyear - 1
        else:
            return store['%s/%s' % ("base", table)]
    
@orca.table('buildings', cache=True)
def buildings(store, fileyear):
    return find_table_in_store('buildings', store, fileyear, base_year)

@orca.table('zones', cache=True)
def zones(store, fileyear):
    return store['%s/zones' % fileyear]

@orca.table('households', cache=True)
def households(store, fileyear):
    return store['%s/households' % fileyear]

@orca.table('jobs', cache=True)
def jobs(store, fileyear):
    return store['%s/jobs' % fileyear]

@orca.table('parcels', cache=True)
def parcels(store, fileyear):
    return store['%s/parcels' % fileyear]

@orca.table('fazes', cache=True)
def fazes(store, fileyear):
    return store['%s/fazes' % fileyear]
