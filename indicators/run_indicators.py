import os
import pandas as pd
import orca
from urbansim.utils import yamlio
import psrc_urbansim.variables
import data

# Indicators script
# ==================

# List of Indicator tables created during each iteration
ind_table_list = []


def DU_and_HH_by_bld_type_by_faz_by_year(ds, ind, settings, iter_var):
    # This will create the DU_and_HH_by_bld_type_by_faz_by_year dataset table
    # for the current year (iter_var).  It will query each column and create
    # the final output csv file.
    print 'Printing from DU_and_HH_by_bld_type_by_faz_by_year()'
    
    column_list = ['DU_SF_19']#, 'DU_MF_12', 'DU_CO_4', 'DU_MH_11', 'DU_Total', /
                   #'HH_SF_19', 'HH_MF_12', 'HH_CO_4', 'HH_MH_11', 'HH_Total']
    for column in column_list:
        #df = orca.get_table(ds)[column]
        print orca.get_table(ds)[column].to_frame().head()
    
    
# Define injectables

# replace this by passing yaml file name as argument
@orca.injectable()
def settings_file():
    return "indicators_settings.yaml"

# Read yaml config
@orca.injectable()
def settings(settings_file):
    return yamlio.yaml_to_dict(str_or_buffer=settings_file)
    

@orca.step()
def compute_indicators(settings, iter_var):
    # loop over indicators and datasets from settings and store into file
    for ind, value in settings['indicators'].iteritems():
        for ds in value['dataset']:
            ds_tablename = '%s_%s_%s' % (ds, ind, str(iter_var))
            print '%s_%s_%s' % (ds, ind, str(iter_var))
            df = orca.get_table(ds)[ind]
            print 'ds is %s and ind is %s' % (ds, ind)
            print orca.get_table(ds)[ind].to_frame().head()
            orca.add_table(ds_tablename, df)
            ind_table_list.append(ds_tablename)
    orca.clear_cache()      
             

@orca.step()
def compute_datasets(settings, iter_var):
    # Loops over dataset_tables and datasets from settings and store into file
    for ind, value in settings['dataset_tables'].iteritems():
        for ds in value['dataset']:
            print 'ds is %s and ind is %s' % (ds, ind)
            func = ind + "(ds, ind, settings, iter_var)"
            eval(func)
            

# Compute indicators
orca.run(['compute_indicators', 'compute_datasets'], iter_vars=settings(settings_file())['years'])

# Create CSV files
def create_csv_files():
    # Creating a unique list of indicators from the tables added in compute_indicators 
    print "ind_table_list"
    print ind_table_list
    unique_ind_table_list = []
    for table in ind_table_list:
        if table[:-5] not in unique_ind_table_list:
            unique_ind_table_list.append(table[:-5])
    print "unique_ind_table_list"
    print unique_ind_table_list    
    # create a CSV file for each indicator with a column per iterationn year
    for ind_table in unique_ind_table_list:
        ind_table_list_for_csv =[] 
        for table in ind_table_list:
            if ind_table in table:
                ind_table_list_for_csv.append(table)
        
        ind_df_list_for_csv = []
        column_labels = []
        for table in ind_table_list_for_csv:
            ind_df_list_for_csv.append(orca.get_table(table).to_frame())
            column_labels.append(table[table.find('_') + 1:])
            
        ind_csv = pd.concat(ind_df_list_for_csv, axis=1)
        ind_csv.columns = column_labels
        ind_csv.to_csv('%s.csv' % ind_table)

create_csv_files()

# test find_table_in_store()
#print orca.get_table('land_use_types').to_frame().head()