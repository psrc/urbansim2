import os
import numpy as np
import pandas as pd
import orca
from urbansim.utils import yamlio
import data
import psrc_urbansim.variables # import variables functions

from urbansim.utils import misc

################### Indicators script
################ read raw data ==================

# read h5 data into Orca
@orca.step()
def read_h5():
    store = pd.HDFStore(os.path.join(misc.data_dir(), 'simresult_demo.h5'), "r")
    table_name_list = store.keys()
    for table_name in table_name_list: 
        orca.add_table(table_name, store[table_name])
orca.run(['read_h5'])

'''
Within simresult_demo.h5, table names are: 
table_name_List = store.keys()
>>>['/2015/fazes', '/2015/households', '/2015/jobs', '/2015/parcels', '/2015/persons', '/2015/zones', '/2016/fazes', '/2016/households', '/2016/jobs', '/2016/parcels', '/2016/persons', '/2016/zones', '/base/fazes', '/base/households', '/base/jobs', '/base/parcels', '/base/persons', '/base/zones']

if you want to check the table:
orca.get_table('/2015/fazes').to_frame()
'''

# jobs 
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
# what is code number for other employment types? 
#---------------------------------8/9------------------- 



# person & age group: 0-5 and 5+ 
@orca.injectable()
def data_file(): 
    return '/2015/persons'

@orca.table()
def raw_data():
    return orca.get_table('/2015/persons').to_frame()

@orca.step()
def processed(raw_data):
    # calculate out age group and number of people in that group
    #df = orca.get_table('/2015/persons').to_frame()
    print raw_data
    number = len(raw_data[raw_data['age'] < 5])
    print number 
    return number

#@orca.step()
#def save_data(processed):
#    number.to_csv('processed' + '.csv')

orca.run(['processed'])








################ Define injectables
# 2015
@orca.injectable()
def data_file():
    return '/2015/persons'

@orca.injectable()
def column_name():
    return 'employment_status'

@orca.table()
def raw_data(data_file):
    df = orca.get_table(data_file).to_frame()
    return df

@orca.step()
def total_pop(column_name, raw_data):
    df = raw_data(columns = [column_name])
    return df.sum()

orca.run(['total_pop'])

# alldata__table__employment.csv
orca.add_injectable('c', 0)

@orca.injectable()
def times(s1, c):
    if s1 >= 0:
        return c += 1

#@orca.table()
#def total_emp(raw_data, 'employment_status', 'worker_num'):
#    worker = raw_data(columns = ['employment_status'])
#    raw_data['worker_num'] = 

#    return np.sum(raw_data['employment_status']*raw_data['worker_num'])

#alldata__table__employment.csv



@orca.injectable()
# do fancy function





############### replace this by passing yaml file name as argument
@orca.injectable()
def settings_file():
    return "indicators_settings.yaml"

############## Read yaml config
@orca.injectable()
def settings(settings_file):
    return yamlio.yaml_to_dict(str_or_buffer=settings_file)

@orca.step()
def compute_indicators(settings):
    # loop over indicators and dataests from settings and store into file
    for ind, value in settings['indicators'].iteritems():
        for ds in value['dataset']:
            print ds
            print orca.get_table(ds)[ind]
             

############## Compute indicators
orca.run(['compute_indicators'], iter_vars=settings(settings_file())['years'])

