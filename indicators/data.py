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

#@orca.injectable()
#def year(iter_var):
#    return iter_var
@orca.injectable()   
def year(base_year):
    if 'iter_var' in orca.list_injectables():
        year = orca.get_injectable('iter_var')
        if year is not None:
            return year
    # outside of a run, return the base/default
    return base_year

@orca.injectable()
def fileyear(year, base_year):
    if year == base_year:
        return "base"
    return year

def store_table_list(store):
    return store.keys()

def find_table_in_store(table, store, year, base_year):
    searchyear = year
    while searchyear > base_year:
        if (('%s/%s' % (searchyear, table)) in store_table_list(store)):
            return store['%s/%s' % (searchyear, table)]
        else:
            searchyear -= 1
    else:
        return store['%s/%s' % ("base", table)]
    
@orca.table('buildings', cache=True)
def buildings(store, year, base_year):
    return find_table_in_store('buildings', store, year, base_year)
    
@orca.table('zones', cache=True)
def zones(store, fileyear):
    return find_table_in_store('zones', store, year, base_year)

@orca.table('households', cache=True)
def households(store, fileyear):
    return find_table_in_store('households', store, year, base_year)

@orca.table('jobs', cache=True)
def jobs(store, fileyear):
    return find_table_in_store('jobs', store, year, base_year)

@orca.table('parcels', cache=True)
def parcels(store, year, base_year):
    return find_table_in_store('parcels', store, year, base_year)

@orca.table('fazes', cache=True)
def fazes(store, fileyear):
    return find_table_in_store('fazes', store, year, base_year)

@orca.table('building_types', cache=True)
def building_types(store, year):
    return find_table_in_store('building_types', store, year, base_year)

