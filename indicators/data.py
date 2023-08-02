import os
import orca
import numpy as np
import pandas as pd
from urbansim.utils import misc

@orca.injectable('store', cache=True)
def store(settings):
    #print os.path.join(os.getenv('DATA_HOME'), settings["store"])
    return pd.HDFStore(os.path.join(os.getenv('DATA_HOME'), settings["store"]), mode='r')

@orca.injectable('csv_store', cache=True)
def csv_store():
    return os.path.join(os.getenv('DATA_HOME'), 'indicators/csv_store')
#    return pd.read_csv(os.path.join(os.getenv('DATA_HOME'), 'indicators/csv_store/growth_centers.csv'),
#                       index_col='growth_center_id')
    
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
#            print 'returning /%s/%s' % (searchyear, table) #for debugging purposes only
#            print store['/%s/%s' % (searchyear, table)].head()
            return store['/%s/%s' % (searchyear, table)]
        else:
            searchyear = searchyear - 1
    else:
#        if (table == 'cities') and (('/%s/%s' % ("base", table)) not in store_table_list(store)):
#            # Return table cities from look up table
#                df_parcels_geo = pd.read_csv(os.path.join(csv_store, 'parcels_geos.csv'),
#                       index_col='parcel_id')
#                return df_parcels_geo.groupby(df_parcels_geo.city_id).first()
#        else:
##        if year <> base_year:
##            print 'Could not find table /%s/%s. Instead using the base table' % (year, table)
#        print 'returning /%s/%s' % ("base", table) #for debugging purposes only
#        print store['%s/%s' % ("base", table)].head()
#            return store['/%s/%s' % ("base", table)]
        return store['/%s/%s' % ("base", table)]
    
def find_cities_in_store(table, store, year, base_year, csv_store):
    searchyear = year
    while searchyear > base_year:
        if (('/%s/%s' % (searchyear, table)) in store_table_list(store)):
#            print 'returning /%s/%s' % (searchyear, table) #for debugging purposes only
#            print store['/%s/%s' % (searchyear, table)].head()
            return store['/%s/%s' % (searchyear, table)]
        else:
            searchyear = searchyear - 1
    else:
        if (table == 'cities') and (('/%s/%s' % ("base", table)) not in store_table_list(store)):
            # Return table cities from look up table
            #print 'returning Cities table from parcels_geo'
            df_parcels_geo = pd.read_csv(os.path.join(csv_store, 'parcels_geos.csv'),
                   index_col='parcel_id')
            return df_parcels_geo.groupby(df_parcels_geo.city_id).first()
        else:
            #print 'returning cities from simresults'
#        if year <> base_year:
#            print 'Could not find table /%s/%s. Instead using the base table' % (year, table)
#        print 'returning /%s/%s' % ("base", table) #for debugging purposes only
#        print store['%s/%s' % ("base", table)].head()
            return store['/%s/%s' % ("base", table)]
    
@orca.table('employment_controls', cache=True)
def employment_controls(store, year, base_year):
    return find_table_in_store('employment_controls', store, year, base_year)

@orca.table('household_controls', cache=True)
def household_controls(store, year, base_year):
    return find_table_in_store('household_controls', store, year, base_year)
    
@orca.table('buildings', cache=True, cache_scope='iteration')
def buildings(store, year, base_year):
    return find_table_in_store('buildings', store, year, base_year)
    
@orca.table('zones', cache=True, cache_scope='iteration')
def zones(store, year, base_year):
    return find_table_in_store('zones', store, year, base_year)

@orca.table('zoning_heights', cache=True, cache_scope='iteration')
def zoning_heights(store, year, base_year):
#    zoning_heights_table = find_table_in_store('zoning_heights', store, year, base_year)
#    print zoning_heights_table.head()
    return find_table_in_store('zoning_heights', store, year, base_year)

@orca.table('households', cache=True, cache_scope='iteration')
def households(store, year, base_year):
    return find_table_in_store('households', store, year, base_year)

@orca.table('jobs', cache=True, cache_scope='iteration')
def jobs(store, year, base_year):
    return find_table_in_store('jobs', store, year, base_year)

@orca.table('parcels', cache=True, cache_scope='iteration')
def parcels(store, year, base_year):
    return find_table_in_store('parcels', store, year, base_year)

@orca.table('parcel_zoning', cache=True, cache_scope='iteration')
def parcel_zoning(store, parcels, zoning_heights):
    pcl = pd.DataFrame(parcels['plan_type_id'])
    pcl['parcel_id'] = pcl.index
    zh = pd.DataFrame(zoning_heights.local)
    zh['plan_type_id'] = zh.index
    #print zh['plan_type_id'].head()
    # merge parcels with zoning_heights
    #zoning = pd.merge(pcl, zh, how='left', on=['plan_type_id']) 
    zoning = pd.merge(pcl, zh, how='left', left_on=['plan_type_id'], right_index=True) 
#    print 'zoning table in parcel_zoning'
#    print zoning.head()
#    print zoning.info()
    # replace NaNs with 0 for records not found in zoning_heights (e.g. plan_type_id 1000) or constraints (e.g. plan_type_id 0)
    for col in zoning.columns:
        zoning[col] = np.nan_to_num(zoning[col])
    return zoning.set_index(['parcel_id'])

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
def cities(store, year, base_year, csv_store):
    #return pd.read_csv(os.path.join(csv_store, 'cities.csv'), index_col='city_id')
    return find_table_in_store('cities', store, year, base_year)

@orca.table('subregs', cache=True, cache_scope='iteration')
def subregs(store, year, base_year, csv_store):
    #return pd.read_csv(os.path.join(csv_store, 'subregs.csv'), index_col='subreg_id')
    return find_table_in_store('subregs', store, year, base_year)

#@orca.table('cities', cache=True, cache_scope='iteration')
#def cities(store, year, base_year, csv_store):
#    return find_cities_in_store('cities', store, year, base_year, csv_store)

#@orca.table('cities', cache=True, cache_scope='iteration')
#def cities(store, year, base_year):
#    df_parcels = find_table_in_store('parcels', store, year, base_year)
#    return df_parcels.groupby(df_parcels.city_id).first()


#@orca.table('counties', cache=True, cache_scope='iteration')
#def counties(store, year, base_year):
#    df_cities = find_table_in_store('cities', store, year, base_year)
#    return df_cities.groupby(df_cities.county_id).first()

@orca.table('counties', cache=True)
def counties():
    df = pd.DataFrame.from_dict({'county_id': [33, 35, 53, 61]})
    df = df.set_index('county_id')
    return df

@orca.table('alldata', cache=True, cache_scope='iteration')
def alldata():
    df = pd.DataFrame.from_dict({'alldata_id': [1]})
    df = df.set_index('alldata_id')
    return df

@orca.table('growth_centers', cache=True, cache_scope='iteration')
def growth_centers(csv_store):
    return pd.read_csv(os.path.join(csv_store, 'growth_centers.csv'),
                       index_col='growth_center_id')
    
@orca.table('parcels_geos', cache=True, cache_scope='iteration')
def parcels_geos(csv_store):
    return pd.read_csv(os.path.join(csv_store, 'parcels_geos.csv'),
                       index_col='parcel_id')

@orca.table('targets', cache=True, cache_scope='iteration')
def targets(csv_store, year, base_year):
    return find_table_in_store('targets', store, year, base_year)
    #return pd.read_csv(os.path.join(csv_store, 'targets.csv'), index_col='target_id')

@orca.table('controls', cache=True, cache_scope='iteration')
def controls(csv_store, year, base_year):
    #return pd.read_csv(os.path.join(csv_store, 'controls.csv'), index_col='control_id')
    return find_table_in_store('controls', store, year, base_year)

@orca.table('control_hcts', cache=True, cache_scope='iteration')
def control_hcts(csv_store, year, base_year):
    #return pd.read_csv(os.path.join(csv_store, 'control_hcts.csv'), index_col='control_hct_id')
    return find_table_in_store('control_hcts', store, year, base_year)
