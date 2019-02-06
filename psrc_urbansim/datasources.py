import os
import orca
import pandas as pd
import numpy as np
import urbansim_defaults.utils as utils
from urbansim.utils import misc
from urbansim_defaults import datasources
import warnings
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

orca.add_injectable('base_year', 2014)
@orca.injectable()
def year(base_year):
    if 'iter_var' in orca.list_injectables():
        year = orca.get_injectable('iter_var')
        if year is not None:
            return year
    
    # outside of a run, return the base/default
    return base_year

@orca.injectable()
def store_table_names_dict(): 
    # Dictionary with pairs of orca.table name and the associated name in storage.
    # Only entries where the names differ.
    return {'employment_controls': "annual_employment_control_totals",
            'employment_sector_group_definitions': "employment_adhoc_sector_group_definitions", 
            'employment_sector_groups': "employment_adhoc_sector_groups",
            'household_controls': "annual_household_control_totals",
            'household_relocation_rates': 'annual_household_relocation_rates',
            'job_relocation_rates': 'annual_job_relocation_rates',
            'parcel_zoning': 'development_constraints'
            }

    
# datasets in alphabetical order

@orca.table('building_sqft_per_job', cache=True)
def building_sqft_per_job(store):
    df = store['building_sqft_per_job']
    return df

@orca.table('building_types', cache=True)
def building_types(store):
    df = store['building_types']
    return df

@orca.table('buildings', cache=True)
def buildings(store):
    df = store['buildings']
    #df = utils.fill_nas_from_config('buildings', df)
    return df

@orca.table('buildings_lag1', cache=True)
def buildings_lag1(store):
    dfname = 'buildings_lag1'
    if dfname not in [x[1:] for x in store.keys()]: 
        dfname = 'buildings'
    return store[dfname]

@orca.table('development_constraints', cache=True)
def development_constraints(store):
    df = store['development_constraints']#.drop_duplicates()
    return df

@orca.table('development_templates', cache=True)
def development_templates(store):
    df = store['development_templates']
    return df

@orca.table('development_template_components', cache=True)
def development_template_components(store):
    df = store['development_template_components']
    return df

@orca.table('employment_controls', cache=True)
def employment_controls(store):
    df = store["annual_employment_control_totals"]
    df[df < 0] = np.inf
    return df

@orca.table('employment_sector_group_definitions', cache=True)
def employment_sector_group_definitions(store):
    df = store["employment_adhoc_sector_group_definitions"]
    # Add a dummy because both columns (group_id, sector_id) are indices.
    # Otherwise it would be an empty dataframe
    df['dummy'] = 0 
    return df

@orca.table('employment_sector_groups', cache=True)
def employment_sector_groups(store):
    df = store["employment_adhoc_sector_groups"]
    return df

@orca.table('employment_sectors', cache=True)
def employment_sectors(store):
    df = store["employment_sectors"]
    return df

@orca.table('fazes', cache=True)
def fazes(store):
    df = store['fazes']
    return df

@orca.table('gridcells', cache=True)
def gridcells(store):
    df = store['gridcells']
    return df

@orca.table('household_controls', cache=True)
def household_controls(store):
    df = store["annual_household_control_totals"]
    df[df < 0] = np.inf
    return df

@orca.table('household_relocation_rates', cache=True)
def household_relocation_rates(store):
    df = store['annual_household_relocation_rates']
    df[df < 0] = np.nan 
    return df

@orca.table('households', cache=True)
def households(store):
    df = store['households']
    if not 'previous_building_id' in df.columns:
        df['previous_building_id'] = df['building_id']
    if not 'is_inmigrant' in df.columns:
        df['is_inmigrant'] = 0
    return df

@orca.table('households_lag1', cache=True)
def households_lag1(store):
    dfname = 'households_lag1'
    if dfname not in [x[1:] for x in store.keys()]: 
        dfname = 'households'
    return store[dfname]

@orca.table('households_for_estimation', cache=True)
def households_for_estimation(store):
    dfname = 'households_for_estimation'
    if dfname not in [x[1:] for x in store.keys()]: 
        dfname = 'households'
    df = store[dfname]
    if 'for_estimation' in df.columns:
        df = df[df.for_estimation == 1]
    return df

@orca.table('job_relocation_rates', cache=True)
def job_relocation_rates(store):
    df = store['annual_job_relocation_rates']  
    df[df < 0] = np.nan 
    return df

@orca.table('jobs', cache=True)
def jobs(store):
    df = store['jobs']
    # below needed to run urbansim_utilties.lcm_simulate
    df['number_of_jobs'] = 1
    df['vacant_jobs'] = 1
    return df

@orca.table('jobs_for_estimation', cache=True)
def jobs_for_estimation(store):
    dfname = 'jobs_for_estimation'
    if dfname not in [x[1:] for x in store.keys()]: 
        dfname = 'jobs'
    df = store[dfname]
    if 'for_estimation' in df.columns:
        df = df[df.for_estimation == 1]    
    return df

@orca.table('land_use_types', cache=True)
def land_use_types(store):
    df = store['land_use_types']
    return df

@orca.table('parcel_zoning', cache=True)
def parcel_zoning(store, development_constraints, parcels, zoning_heights):
    constr = development_constraints.local.drop_duplicates()
    # merge constraints with parcels
    pcl = pd.DataFrame(parcels['plan_type_id'])
    pcl['parcel_id'] = pcl.index
    constr['constraint_id'] = constr.index
    zoning = pd.merge(pcl, constr, how='left', on=['plan_type_id']) 
    # merge with zoning_heights
    zoning = pd.merge(zoning, zoning_heights.local, how='left', left_on=['plan_type_id'], right_index=True)
    # replace NaNs with 0 for records not found in zoning_heights (e.g. plan_type_id 1000) or constraints (e.g. plan_type_id 0)
    for col in zoning.columns:
        zoning[col] = np.nan_to_num(zoning[col])
    # index is a composition of three attributes
    return zoning.set_index(['parcel_id','generic_land_use_type_id', 'constraint_type'])

@orca.table('parcels', cache=True)
def parcels(store):
    df = store['parcels']
    return df

@orca.table('persons', cache=True)
def persons(store):
    df = store['persons']
    return df

@orca.table('persons_for_estimation', cache=True)
def persons(store):
    df = store['persons_for_estimation']
    # job_id = -1 is used for workers that have not been assigned a job, so coding non-workers to -2
    df.job_id = np.where(df.employment_status>0, df.job_id, -2)
    return df

@orca.table('schools', cache=True)
def schools(store):
    df = store['schools']
    return df

@orca.table('target_vacancies', cache=True)
def target_vacancies(store):
    df = store['target_vacancies']
    return df

@orca.table('target_vacancy', cache=True)
def target_vacancy(target_vacancies, year):
    df = target_vacancies.local[target_vacancies.index.get_level_values('year') == year]
    df.reset_index("year", drop=True, inplace=True) # remove year as index
    return df

@orca.table('tractcity', cache=True)
def tractcity(store):
    df = store['tractcity']
    return df

@orca.table('travel_data', cache=True)
def travel_data(store):
    df = store['travel_data']
    return df

@orca.table('zones', cache=True)
def zones(store):
    df = store['zones']
    return df

@orca.table('zoning_heights', cache=True)
def zoning_heights(store):
    df = store['zoning_heights']
    return df

orca.broadcast('buildings', 'households', cast_index=True, onto_on='building_id')
orca.broadcast('buildings', 'jobs', cast_index=True, onto_on='building_id')
orca.broadcast('fazes', 'zones', cast_index=True, onto_on='faz_id')
orca.broadcast('gridcells', 'parcels', cast_index=True, onto_on='grid_id')
orca.broadcast('households', 'persons', cast_index=True, onto_on='household_id')
#orca.broadcast('jobs', 'households', cast_index=True, onto_on='job_id')
orca.broadcast('jobs', 'persons', cast_index=True, onto_on='job_id')
orca.broadcast('parcels', 'buildings', cast_index=True, onto_on='parcel_id')
orca.broadcast('parcels', 'schools', cast_index=True, onto_on='parcel_id')
orca.broadcast('tractcity', 'parcels', cast_index=True, onto_on='tractcity_id')
orca.broadcast('zones', 'parcels', cast_index=True, onto_on='zone_id')
orca.broadcast('zones', 'persons_for_estimation', cast_index=True, onto_on='household_zone_id')
orca.broadcast('zones', 'persons', cast_index=True, onto_on='household_zone_id')
orca.broadcast('buildings', 'households_for_estimation', cast_index=True, onto_on='building_id')
orca.broadcast('buildings_lag1', 'households_for_estimation', cast_index=True, onto_on='building_id')
orca.broadcast('households_for_estimation', 'persons_for_estimation', cast_index=True, onto_on='household_id')
orca.broadcast('jobs', 'households_for_estimation', cast_index=True, onto_on='job_id')

# Assumptions

# this maps building "forms" from the developer model
# to building types so that when the developer builds a
# "form" this can be converted for storing as a type
# in the building table - in the long run, the developer
# forms and the building types should be the same and the
# developer model should account for the differences
orca.add_injectable("form_to_btype", {
    'residential': [1, 2, 3],
    'industrial': [7, 8, 9],
    'retail': [10, 11],
    'office': [4],
    'mixedresidential': [12],
    'mixedoffice': [14],
})