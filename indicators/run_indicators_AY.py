import os
import numpy as np
import pandas as pd
import orca
from urbansim.utils import yamlio
import data
import psrc_urbansim.variables # import variables functions
from urbansim.utils import misc


# read h5 data into Orca
@orca.step()
def read_h5():
    store = pd.HDFStore(os.path.join(misc.data_dir(), 'simresult_demo_1212.h5'), "r")
    table_name_list = store.keys()
    for table_name in table_name_list: 
        orca.add_table(table_name, store[table_name])
orca.run(['read_h5'])

'''
Within simresult_demo.h5, table names are: 
table_name_List = store.keys()
>>>['/2015/fazes', '/2015/households', '/2015/jobs', '/2015/parcels', '/2015/persons', '/2015/zones', '/2016/fazes', '/2016/households', '/2016/jobs', '/2016/parcels', '/2016/persons', '/2016/zones', '/base/fazes', '/base/households', '/base/jobs', '/base/parcels', '/base/persons', '/base/zones']

if you want to check out the table:
orca.get_table('/2015/fazes').to_frame()
'''

## jobs by type 
def is_in_sector_group(group_name, jobs, employment_sectors, employment_sector_groups, employment_sector_group_definitions):
    group = employment_sector_groups.index[employment_sector_groups['name'] == group_name]
    idx = [jobs.sector_id.values, group[0]*np.ones(jobs.sector_id.size)]
    midx = pd.MultiIndex.from_arrays(idx, names=('sector_id', 'group_id'))
    res = np.logical_not(np.isnan(employment_sector_group_definitions.dummy[midx])).reset_index("group_id").dummy
    res.index = jobs.index
    return res 

@orca.column('/2015/jobs', 'is_in_sector_group_retail', cache=True)
def is_in_sector_group_retail(jobs, employment_sectors, employment_sector_groups, employment_sector_group_definitions):
    return is_in_sector_group("retail", jobs, employment_sectors, employment_sector_groups, employment_sector_group_definitions)

orca.get_table('/2015/jobs').to_frame().head()
# what is code number for other employment types? to be continued...
####################################################################

## population by year and geographic units 
# read population, household, parcel file, 
df1 = orca.get_table('/2015/persons').to_frame()
orca.add_table('my_person', df1)
df2 = orca.get_table('/2015/households').to_frame()
df2['household_id'] = df2.index
orca.add_table('my_household', df2)
df3 = orca.get_table('/2015/parcels').to_frame()
orca.add_table('my_parcel', df3)

# define the merging relationship, between person and household
orca.broadcast(cast='my_person', onto='my_household', cast_index=True, onto_on='household_id')
#orca.broadcast(cast='my_household', onto='my_parcel', cast_index=True, onto_on='census_2010_block_group_id')

my_col = ['population_2015']
#my_col = ['population_2015', 'faz_id']

@orca.step()
def get_person_geo():
    # join person data with hourshold data
    df4 = orca.merge_tables(target='my_household', tables=['my_person', 'my_household'])
    # geographic info to dictionary 
    faz_dict = dict(zip(df3['census_2010_block_group_id'], df3['faz_id']))
    zone_dict = dict(zip(df3['census_2010_block_group_id'], df3['zone_id']))
    city_dict = dict(zip(df3['census_2010_block_group_id'], df3['city_id']))
    # map geo info to person table
    df4['faz_id'] = df4['census_2010_block_group_id'].map(faz_dict)
    df4['zone_id'] = df4['census_2010_block_group_id'].map(zone_dict)
    df4['city_id'] = df4['census_2010_block_group_id'].map(city_dict)
    orca.add_table('my_person_geo', df4)
    #print df4.columns

# person count on diff geographic units
@orca.step()
def groupby_person(my_person_geo):
    df = my_person_geo.to_frame()
    df5 = df.groupby(by=geo_id)['persons'].count().to_frame()
    #df5[my_col[-1]] = df5.index
    df5.columns = my_col
    print df5.head()
    orca.add_table(file_name, df5)

# save person count file
@orca.step()
def output_table(my_person_faz_test):
    df = my_person_faz_test.to_frame()
    df.to_csv(file_csv)

# faz level summary 
geo_id = 'faz_id'
file_name = 'my_person_faz_test'
file_csv = 'my_person_faz_test.csv'
orca.run(['get_person_geo', 'groupby_person', 'output_table'])

# city level summery
geo_id = 'city_id'
file_name = 'my_person_city_test'
file_csv = 'my_person_city_test.csv'
orca.run(['get_person_geo', 'groupby_person', 'output_table'])

# all data summery
geo_id = 'city_id'
file_name = 'my_person_city_test_alldata'
file_csv = 'my_person_city_test_alldata.csv'
orca.run(['get_person_geo', 'groupby_person', 'output_table'])


