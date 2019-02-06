import os
import pandas as pd
import orca
from urbansim.utils import yamlio
import psrc_urbansim.variables
import data

# Indicators script
# ==================

datasets = {'DU_and_HH_by_bld_type_by_faz_by_year': ['DU_SF_19', 'DU_MF_12', 'DU_CO_4',
                                                      'DU_MH_11', 'DU_Total', 'HH_SF_19',
                                                      'HH_MF_12', 'HH_CO_4', 'HH_MH_11',
                                                      'HH_Total'],
            'employment_by_aggr_sector': ['Natural_resources', 'Construction',
                                         'Manuf', 'WTU', 'Retail', 'Business_Services',
                                         'Private_Ed', 'Healthcare', 'Food_Services',
                                         'Personal_Services', 'government', 'edu'],
            'persons_by_5year_age_groups': ['age_0_to_5', 'age_6_to_10', 'age_11_to_15',
                                            'age_16_to_20', 'age_21_to_25', 'age_26_to_30',
                                            'age_31_to_35', 'age_36_to_40', 'age_41_to_45',
                                            'age_46_to_50', 'age_51_to_55', 'age_56_to_60',
                                            'age_61_to_65', 'age_66_to_70', 'age_71_to_75',
                                            'age_76_to_80', 'age_81_to_85', 'age_86_to_90',
                                            'age_91_to_95', 'age_96_and_up'],
            'persons_by_age_groups_of_interest': ['Under5', 'Five_18', 'Nineteen_24',
                                                  'Twentyfive_60', 'Over_60'],
#            'pptyp': ['full_time_worker','part_time_worker', 'non_working_adult_age_65_plus',
#                                                   'non_working_adult_age_16_64',
#                                                   'university_student','hs_student_age_15_up',
#                                                   'child_age_5_15','child_age_0_4']
            'regional_total_hhs_by_30_60_90_in_14dollars_groups': ['Group1_Under36870K',
                                                                   'Group2_UpTo73700',
                                                                   'Group3_UpTo110600',
                                                                   'Group4_Over110600'],
            'regional_total_hhs_by_new_14incomegroups': ['Group1_Under50K','Group2_50_75K',
                                                         'Group3_75_100K','Group4_Over100K']
            }

# create_csv() will export the a .csv file from the given data and with the
# given file name.
def create_csv(column_list, column_list_headers, csv_file_name):
    ind_csv = pd.concat(column_list, axis=1)
    ind_csv.columns = column_list_headers
    ind_csv.to_csv(csv_file_name) 
    
def DU_and_HH_by_bld_type_by_faz_by_year(ds, ind, settings, iter_var):
    # This will create the DU_and_HH_by_bld_type_by_faz_by_year dataset table
    # for the current year (iter_var).  It will query each column and create
    # the final output csv file.
    print 'Printing from DU_and_HH_by_bld_type_by_faz_by_year()'
    column_list = ['DU_SF_19', 'DU_MF_12', 'DU_CO_4', 'DU_MH_11', 'DU_Total', \
                  'HH_SF_19', 'HH_MF_12', 'HH_CO_4', 'HH_MH_11', 'HH_Total']
    column_list_for_csv_table = []
    for column in column_list:
        df = orca.get_table(ds)[column]
        #print orca.get_table(ds)[column].to_frame().head()
        orca.add_table(column, df)
        column_list_for_csv_table.append(orca.get_table(column).to_frame())    
    create_csv(column_list_for_csv_table, column_list,
               '%s__dataset_table__DU_and_HH_by_bld_type_by_faz_by_year__%s.csv' % (ds, str(iter_var)))
#    ind_csv = pd.concat(column_list_for_csv_table, axis=1)
#    ind_csv.columns = column_list
#    ind_csv.to_csv('%s__dataset_table__DU_and_HH_by_bld_type_by_faz_by_year__%s.csv' % (ds, str(iter_var))) 
    


# List of Indicator tables created during each iteration used in compute_indicators()
ind_table_list = []
    
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
            #This is where my test start for using the dictionary of column headers
            column_list_for_csv_table = []
            for column in datasets[ind]:
                df = orca.get_table(ds)[column]
                #print orca.get_table(ds)[column].to_frame().head()
                orca.add_table(column, df)
                column_list_for_csv_table.append(orca.get_table(column).to_frame())    
            create_csv(column_list_for_csv_table, datasets[ind],
                       '%s__dataset_table__%s__%s.csv' % (ds, ind, str(iter_var)))
            #Test ends
            
#            func = ind + "(ds, ind, settings, iter_var)"
#            eval(func)
            

# Compute indicators
orca.run(['compute_indicators', 'compute_datasets'], iter_vars=settings(settings_file())['years'])

# Create CSV files
def create_table_csv_files():
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
         
        create_csv(ind_df_list_for_csv, column_labels, '%s.csv' % ind_table)
#        ind_csv = pd.concat(ind_df_list_for_csv, axis=1)
#        ind_csv.columns = column_labels
#        ind_csv.to_csv('%s.csv' % ind_table)

create_table_csv_files()

# test find_table_in_store()
#print orca.get_table('land_use_types').to_frame().head()