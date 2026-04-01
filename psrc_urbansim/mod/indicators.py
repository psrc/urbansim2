import os
import numpy as np
import pandas as pd
import orca
from urbansim.utils import misc


# =====================
# Data source definitions (formerly indicators/data.py)
# =====================

@orca.injectable('store', cache=True)
def store(settings):
    return pd.HDFStore(os.path.join(settings['output_dir'], settings['store']), mode='r')

@orca.injectable('csv_store', cache=True)
def csv_store(settings):
    return os.path.join(settings['data_dir'], 'csv_store')

@orca.injectable()
def base_year(settings):
    return settings['base_year']

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
            return store['/%s/%s' % (searchyear, table)]
        else:
            searchyear = searchyear - 1
    else:
        return store['/base/%s' % table]
    
def find_cities_in_store(table, store, year, base_year, csv_store):
    searchyear = year
    while searchyear > base_year:
        if (('/%s/%s' % (searchyear, table)) in store_table_list(store)):
            return store['/%s/%s' % (searchyear, table)]
        else:
            searchyear = searchyear - 1
    else:
        if (table == 'cities') and (('/%s/%s' % ("base", table)) not in store_table_list(store)):
            df_parcels_geo = pd.read_csv(os.path.join(csv_store, 'parcels_geos.csv'),
                   index_col='parcel_id')
            return df_parcels_geo.groupby(df_parcels_geo.city_id).first()
        else:
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
    zoning = pd.merge(pcl, zh, how='left', left_on=['plan_type_id'], right_index=True) 
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
    return find_table_in_store('cities', store, year, base_year)

@orca.table('subregs', cache=True, cache_scope='iteration')
def subregs(store, year, base_year, csv_store):
    return find_table_in_store('subregs', store, year, base_year)

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
def targets(store, year, base_year):
    return find_table_in_store('targets', store, year, base_year)

@orca.table('controls', cache=True, cache_scope='iteration')
def controls(store, year, base_year):
    return find_table_in_store('controls', store, year, base_year)

@orca.table('control_hcts', cache=True, cache_scope='iteration')
def control_hcts(store, year, base_year):
    return find_table_in_store('control_hcts', store, year, base_year)


# =====================
# Indicator definitions
# =====================# Table names and columns for creating Dataset CSV files
datasets = {'DU_and_HH_by_bld_type_by_faz_by_year': ['DU_SF_19', 'DU_MF_12', 'DU_CO_4',
                                                      'DU_MH_11', 'DU_Total', 'HH_SF_19',
                                                      'HH_MF_12', 'HH_CO_4', 'HH_MH_11',
                                                      'HH_Total'],
            'employment_by_sector': ['Natural_resources', 'Construction',
                                         'Manuf', 'WTU', 'Retail_only', 'Business_Services',
                                         'Private_Ed', 'Healthcare', 'Food_Services',
                                         'Personal_Services', 'government', 'edu'],
            'employment_by_aggr_sector': ['Con_Res', #1, 2
                                         'Manuf_WTU', # 3, 4
                                         'Retail', # 5, 10
                                         'FIRES', # 7, 9, 11
                                         'Gov', # 12
                                         'Edu' # 13, 8
                                         ],            
            'persons_by_5year_age_groups': ['age_0_to_5', 'age_6_to_10', 'age_11_to_15',
                                            'age_16_to_20', 'age_21_to_25', 'age_26_to_30',
                                            'age_31_to_35', 'age_36_to_40', 'age_41_to_45',
                                            'age_46_to_50', 'age_51_to_55', 'age_56_to_60',
                                            'age_61_to_65', 'age_66_to_70', 'age_71_to_75',
                                            'age_76_to_80', 'age_81_to_85', 'age_86_to_90',
                                            'age_91_to_95', 'age_96_and_up'],
            'persons_by_age_groups_of_interest': ['Under5', 'Five_18', 'Nineteen_24',
                                                  'Twentyfive_60', 'Over_60'],
            'pptyp': ['full_time_worker','part_time_worker', 'non_working_adult_age_65_plus',
                                                   'non_working_adult_age_16_64',
                                                   'university_student','hs_student_age_15_up',
                                                   'child_age_5_15','child_age_0_4'],
            'pwtyp': ['full_time', 'part_time', 'workers_no_job', 'non_workers_no_job'],
            'regional_total_hhs_by_30_60_90_in_14dollars_groups': ['Group1_Under36870K',
                                                                   'Group2_UpTo73700',
                                                                   'Group3_UpTo110600',
                                                                   'Group4_Over110600'],
            'regional_total_hhs_by_new_14incomegroups': ['Group1_Under50K','Group2_50_75K',
                                                         'Group3_75_100K','Group4_Over100K'],
            'eoy_vacancy_by_building_type': ['res_4_VR','res_12_VR','res_19_VR','nonres_3_VR',
                                             'nonres_8_VR','nonres_13_VR','nonres_20_VR',
                                             'nonres_21_VR'],
            'units_and_nonres_sqft_by_building_type': ['res_4_units','res_12_units','res_19_units',
                                                       'nonres_3_spaces','nonres_8_spaces','nonres_13_spaces',
                                                       'nonres_20_spaces','nonres_21_spaces','nonres_3_sqft',
                                                       'nonres_8_sqft','nonres_13_sqft','nonres_20_sqft',
                                                       'nonres_21_sqft'],
            'new_buildings': ['building_type_id', 'parcel_id', 'residential_units', 'non_residential_sqft', 'year_built', 'building_sqft', 'unit_price']
            }


geography_alias = {'cities': 'city', 'zones': 'zone', 'fazes': 'faz', 'subregs': 'subreg', 'targets': 'target',
                   'controls' : 'control', 'control_hcts': 'control_hct',
                   'counties': 'county', 'growth_centers': 'growth_center', 'buildings': 'building'}

table_alias = {'number_of_jobs': 'employment', 'number_of_households': 'households',
               'max_developable_capacity': 'max_dev_capacity',
               'max_developable_nonresidential_capacity': 'max_dev_nonresidential_capacity',
               'max_developable_residential_capacity': 'max_dev_residential_capacity'}


def create_csv(column_list, column_list_headers, csv_file_name):
    ind_csv = pd.concat(column_list, axis=1)
    ind_csv.columns = column_list_headers
    ind_csv.to_csv(csv_file_name)

def create_tab(column_list, column_list_headers, csv_file_name):
    ind_csv = pd.concat(column_list, axis=1)
    ind_csv.columns = column_list_headers
    ind_csv.to_csv(csv_file_name, sep='\t')


# List of Indicator tables created during each iteration used in compute_indicators()
ind_table_dic = {} 


@orca.injectable(cache=True)
def years_to_run(settings):
    if settings.get("years", None) is None:
        return range(settings["years_all"][0], settings["years_all"][1]+1)
    return settings["years"]
        
@orca.injectable(cache=True)
def is_annual(settings):
    return not "years" in settings.keys() and "years_all" in settings.keys()


@orca.step()
def add_new_datasets(settings, iter_var):
    # Add additional datasets (stored in csv files) to orca.
    # This model can be used when we need to change geography,
    # e.g. supply different city_id on parcel level and new cities table.
    # In the settings, use the node new_datasets with one subnode per dataset.
    # For each dataset, the node csv_file defines the name of the csv file with path 
    # relative to data_dir.
    # If the dataset should be merged with the existing dataset, 
    # set the node merge_with_existing to true.
    new_datasets = settings.get("new_datasets", {})
    if len(new_datasets) == 0:
        return
    for dsname, conf in new_datasets.items():
        orca_ds = orca.get_table(dsname).local
        ds = pd.read_csv(os.path.join(settings['data_dir'], conf.get("csv_file")), index_col=orca_ds.index.name)
        if conf.get("merge_with_existing", False):
            orca_ds[ds.columns] = ds
        else:
            orca_ds = ds
        orca.add_table(dsname, orca_ds)

    
@orca.step()
def compute_indicators(settings, iter_var, is_annual, years_to_run):
    # loop over indicators and datasets from settings and store into file
    suffix = ""
    if is_annual:
        suffix = "An"
    for ind, value in settings.get('indicators', {}).items():
        if iter_var not in value.get('year', years_to_run):
            continue
        for ds in value.get('dataset', {}):
            df = orca.get_table(ds)[ind].to_frame()
            if ds in geography_alias:
                ds = geography_alias[ds]
            if ind == 'nonres_sqft' and ds == 'alldata':
                ds_tablename = '%s__table__%s%s_%s' % (ds, 'non_residential_sqft', suffix, str(iter_var))
            elif ind in table_alias:
                ds_tablename = '%s__table__%s%s_%s' % (ds, table_alias[ind], suffix, str(iter_var))
            else:
                ds_tablename = '%s__table__%s%s_%s' % (ds, ind, suffix, str(iter_var))
            orca.add_table(ds_tablename, df)
            ind_table_dic[ds_tablename] = value['file_type']
    orca.clear_cache()      
             

def indicators_outdir(settings):
    """Return the output subdirectory for indicator files, derived from the store filename."""
    store_stem = os.path.splitext(settings['store'])[0]
    store_stem = store_stem.replace('results', 'indicators', 1)
    outdir = os.path.join(settings.get('output_dir', '.'), store_stem)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


@orca.step()
def compute_datasets(settings, iter_var, years_to_run):
    if not settings.get("compute_dataset_tables", True):
        return
    outdir = indicators_outdir(settings)
    # Loops over dataset_tables and datasets from settings and store into file
    for ind, value in settings.get('dataset_tables', {}).items():
        if iter_var not in value.get('year', years_to_run):
            continue
        for ds in value['dataset']:
            column_list_for_csv_table = []
            if value.get("include_condition", None) is not None:
                subset = orca.get_table(ds).local.query(value.get("include_condition"))
            dsobj = orca.get_table(ds)
            if not ind in datasets.keys():
                columns = dsobj.columns
            else:
                columns = datasets[ind]
            for column in columns:
                df = dsobj[column].to_frame()
                if value.get("include_condition", None) is not None:
                    df = df.loc[subset.index]
                orca.add_table(column, df)
                column_list_for_csv_table.append(orca.get_table(column).to_frame())
            if ds in geography_alias:
                ds = geography_alias[ds]
            fn = value.get("file_name", None)
            if value['file_type'] == 'csv':
                if fn is not None:
                    file_name = '%s.csv'% fn
                else:
                    file_name = '%s__dataset_table__%s__%s.csv' % (ds, ind, str(iter_var))
                create_csv(column_list_for_csv_table, columns, os.path.join(outdir, file_name))
            elif value['file_type'] == 'tab':
                if fn is not None:
                    file_name = '%s.tab'% fn
                else:
                    file_name = '%s__dataset_table__%s__%s.tab' % (ds, ind, str(iter_var))   
                create_tab(column_list_for_csv_table, datasets[ind], os.path.join(outdir, file_name))
    orca.clear_cache()


def create_tables(outdir):
    """Save collected indicator tables to CSV/tab files."""
    if not os.path.exists(outdir):
        os.makedirs(outdir)    
    unique_ind_table_dic = {}
    for table, filetype in ind_table_dic.items():
        if table[:-5] not in unique_ind_table_dic:
            unique_ind_table_dic[table[:-5]] = filetype
    for ind_table, filetype in unique_ind_table_dic.items():
        ind_table_list_for_csv = [] 
        for table in ind_table_dic:
            if ind_table in table:
                ind_table_list_for_csv.append(table)
        ind_table_list_for_csv.sort()
        ind_df_list_for_csv = []
        column_labels = []
        for table in ind_table_list_for_csv:
            ind_df_list_for_csv.append(orca.get_table(table).to_frame())
            column_labels.append(table[table.find('table') + 7:])
            
        if filetype == 'csv':
            create_csv(ind_df_list_for_csv, column_labels, os.path.join(outdir, '%s.csv' % ind_table))
        elif filetype == 'tab':
            create_tab(ind_df_list_for_csv, column_labels, os.path.join(outdir, '%s.tab' % ind_table))
