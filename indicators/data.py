import os
import orca
import pandas as pd
from urbansim.utils import misc

@orca.injectable('store', cache=True)
def store(settings):
    #print os.path.join(os.getenv('DATA_HOME'), settings["store"])
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
        if (('/%s/%s' % (searchyear, table)) in store_table_list(store)):
            print 'returning /%s/%s' % (searchyear, table) #for debugging purposes only
            print store['/%s/%s' % (searchyear, table)].head()
            return store['/%s/%s' % (searchyear, table)]
        else:
            searchyear = searchyear - 1
    else:
        if year <> base_year:
            print 'Could not find table /%s/%s. Instead using the base table' % (year, table)
        print 'returning /%s/%s' % ("base", table) #for debugging purposes only
        print store['%s/%s' % ("base", table)].head()
        return store['/%s/%s' % ("base", table)]
    
@orca.table('buildings', cache=True, cache_scope='iteration')
def buildings(store, year, base_year):
    return find_table_in_store('buildings', store, year, base_year)
    
@orca.table('zones', cache=True, cache_scope='iteration')
def zones(store, year, base_year):
    return find_table_in_store('zones', store, year, base_year)

@orca.table('households', cache=True, cache_scope='iteration')
def households(store, year, base_year):
    return find_table_in_store('households', store, year, base_year)

@orca.table('jobs', cache=True, cache_scope='iteration')
def jobs(store, year, base_year):
    return find_table_in_store('jobs', store, year, base_year)

@orca.table('parcels', cache=True, cache_scope='iteration')
def parcels(store, year, base_year):
    return find_table_in_store('parcels', store, year, base_year)

@orca.table('fazes', cache=True, cache_scope='iteration')
def fazes(store, year, base_year):
    return find_table_in_store('fazes', store, year, base_year)

@orca.table('building_types', cache=True, cache_scope='iteration')
def building_types(store, year, base_year):
    return find_table_in_store('building_types', store, year, base_year)

@orca.table('land_use_types', cache=True, cache_scope='iteration')
def land_use_types(store, year, base_year):
    return find_table_in_store('land_use_types', store, year, base_year)

@orca.table('persons', cache=True, cache_scope='iteration')
def persons(store, year, base_year):
    return find_table_in_store('persons', store, year, base_year)

@orca.table('building_sqft_per_job', cache=True, cache_scope='iteration')
def building_sqft_per_job(store, year, base_year):
    return find_table_in_store('building_sqft_per_job', store, year, base_year)

@orca.table('employment_sectors', cache=True, cache_scope='iteration')
def employment_sectors(store, year, base_year):
    return find_table_in_store('employment_sectors', store, year, base_year)

@orca.table('employment_sector_groups', cache=True, cache_scope='iteration')
def employment_sector_groups(store, year, base_year):
    return find_table_in_store('employment_sector_groups', store, year, base_year)

@orca.table('schools', cache=True, cache_scope='iteration')
def schools(store, year, base_year):
    return find_table_in_store('schools', store, year, base_year)

@orca.table('travel_data', cache=True, cache_scope='iteration')
def travel_data(store, year, base_year):
    return find_table_in_store('travel_data', store, year, base_year)

@orca.table('gridcells', cache=True, cache_scope='iteration')
def gridcells(store, year, base_year):
    return find_table_in_store('gridcells', store, year, base_year)

@orca.table('cities', cache=True, cache_scope='iteration')
def cities(store, year, base_year):
    df_parcels = find_table_in_store('parcels', store, year, base_year)
    return df_parcels.groupby(df_parcels.city_id).first()

@orca.table('counties', cache=True, cache_scope='iteration')
def counties(store, year, base_year):
    df_parcels = find_table_in_store('parcels', store, year, base_year)
    return df_parcels.groupby(df_parcels.county_id).first()

@orca.table('alldata', cache=True)
def alldata():
    df = pd.DataFrame.from_dict({'alldata_id': [1]})
    df = df.set_index('alldata_id')
    return df