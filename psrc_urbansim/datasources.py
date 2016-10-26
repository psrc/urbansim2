import os
import orca
import pandas as pd
import numpy as np
import urbansim_defaults.utils as utils
from urbansim.utils import misc
from urbansim_defaults import datasources
import warnings
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)


@orca.table('parcels', cache=True)
def parcels(store):
    df = store['parcels']
    return df

@orca.table('buildings', cache=True)
def buildings(store):
    df = store['buildings']
    #df = utils.fill_nas_from_config('buildings', df)
    return df

@orca.table('households', cache=True)
def households(store):
    df = store['households']
    return df

@orca.table('jobs', cache=True)
def jobs(store):
    df = store['jobs']
    #df = utils.fill_nas_from_config('jobs', df)
    return df

@orca.table('persons', cache=True)
def persons(store):
    df = store['persons']
    return df

@orca.table('zones', cache=True)
def zones(store):
    df = store['zones']
    return df

@orca.table('fazes', cache=True)
def fazes(store):
    df = store['fazes']
    return df

@orca.table('tractcity', cache=True)
def tractcity(store):
    df = store['tractcity']
    return df

@orca.table('household_relocation_rates', cache=True)
def households_relocation_rates(store):
    df = store['annual_household_relocation_rates']
    # if the dataset was indexed by one of the columns, make it again a regular column
    # TODO: change it  in the conversion script
    if df.index.name is not None:
        df[df.index.name] = df.index
    #TODO: this is a hack! Update this in the input data. 
    df = df.rename(columns={"age_min": "age_of_head_min", "age_max": "age_of_head_max"})
    df[df < 0] = np.nan 
    return df

@orca.table('job_relocation_rates', cache=True)
def jobs_relocation_rates(store):
    df = store['annual_job_relocation_rates']
    # if the dataset was indexed by one of the columns, make it again a regular column
    if df.index.name is not None:
        df[df.index.name] = df.index    
    df[df < 0] = np.nan 
    return df

@orca.table('building_sqft_per_job', cache=True)
def buildings(store):
    df = store['building_sqft_per_job']
    return df

@orca.table('household_controls', cache=True)
def household_controls(store):
    df = store["annual_household_control_totals"]
    # TODO: This is a hack to overcome the fact that the transition model canot deal with -1s
    # remove 2015 controls and replace them with 2016
    del df['city_id']
    df = df.drop(2015)
    df2 = df.loc[2016]
    df2.index = df2.index-1
    df = df.append(df2)
    df[df < 0] = np.inf
    # TODO: This should be changed in the input data
    # add 0.5 to max of workers and persons, since an open right bracket is used, i.e. [min, max)
    df['persons_max'] = df['persons_max'] + 0.5
    df['workers_max'] = df['workers_max'] + 0.5
    return df

@orca.table('employment_controls', cache=True)
def employment_controls(store):
    df = store["annual_employment_control_totals"]
    # if the dataset was indexed by one of the columns, make it again a regular column
    # TODO: change it  in the conversion script
    if df.index.names is not None:
        for name in df.index.names:
            df[name] = df.index.get_level_values(name)
    df = df.set_index('year')
    # TODO: This is a hack to overcome the fact that the transition model canot deal with -1s
    # remove 2015 controls and replace them with 2016
    del df['city_id']
    df = df.drop(2015)
    df2 = df.loc[2016]
    df2.index = df2.index-1
    df = df.append(df2)
    return df



orca.broadcast('parcels', 'buildings', cast_index=True, onto_on='parcel_id')
orca.broadcast('buildings', 'households', cast_index=True, onto_on='building_id')
orca.broadcast('buildings', 'jobs', cast_index=True, onto_on='building_id')
orca.broadcast('zones', 'parcels', cast_index=True, onto_on='zone_id')
orca.broadcast('households', 'persons', cast_index=True, onto_on='household_id')
orca.broadcast('jobs', 'households', cast_index=True, onto_on='job_id')
orca.broadcast('fazes', 'zones', cast_index=True, onto_on='faz_id')
orca.broadcast('tractcity', 'parcels', cast_index=True, onto_on='tractcity_id')
