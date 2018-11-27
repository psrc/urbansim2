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

# Define injectables

@orca.injectable()
def alldata_ds_dictionary():
    return {'number_of_households': "households",
            'number_of_jobs': "jobs",
            'residential_units': "buildings",
            'population': "households"}

@orca.injectable()
def alldata_ind_dictionary():
    return{'number_of_jobs': "total_number_of_jobs",
            'residential_units': "total_residential_units"}

# replace this by passing yaml file name as argument
@orca.injectable()
def settings_file():
    return "indicators_settings.yaml"

# Read yaml config
@orca.injectable()
def settings(settings_file):
    return yamlio.yaml_to_dict(str_or_buffer=settings_file)
    

@orca.step()
def compute_indicators(settings, iter_var, alldata_ds_dictionary, alldata_ind_dictionary):
    # loop over indicators and datasets from settings and store into file
    for ind, value in settings['indicators'].iteritems():
        for ds in value['dataset']:
            if ds == "alldata":
                ds_tablename = '%s_%s_%s' % (ds, ind, str(iter_var))
                print ds_tablename
                print alldata_ds_dictionary[ind]
                if ind in alldata_ind_dictionary:
                    df = orca.get_table(alldata_ds_dictionary[ind])[alldata_ind_dictionary[ind]]
                    print 'ds is %s, the table called is %s, and ind is %s' % (ds, alldata_ds_dictionary[ind], alldata_ind_dictionary[ind])
                    print orca.get_table(alldata_ds_dictionary[ind])[alldata_ind_dictionary[ind]].to_frame().head()
                else:
                    df = orca.get_table(alldata_ds_dictionary[ind])[ind]
                    print 'ds is %s, the table called is %s, and ind is %s' % (ds, alldata_ds_dictionary[ind], ind)
                    print orca.get_table(alldata_ds_dictionary[ind])[ind].to_frame().head()
            else:
                ds_tablename = '%s_%s_%s' % (ds, ind, str(iter_var))
                print '%s_%s_%s' % (ds, ind, str(iter_var))
                df = orca.get_table(ds)[ind]
                print 'ds is %s and ind is %s' % (ds, ind)
                print orca.get_table(ds)[ind].to_frame().head()
            orca.add_table(ds_tablename, df)
            ind_table_list.append(ds_tablename)
    orca.clear_cache()      
             

# Compute indicators
orca.run(['compute_indicators'], iter_vars=settings(settings_file())['years'])

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