import pandas as pd
import numpy as np
import scipy.ndimage as ndi
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# FAZES VARIABLES
#####################
@orca.column('fazes', 'avg_school_score', cache=True, cache_scope='iteration')
def avg_school_score(fazes, schools):
    """
    Computes the average of the school's total score over FAZes, where missing values are removed 
    from the computation. Missing values are those that are less or equal zero.
    """    
    valid_idx = schools.total_score > 0
    res = ndi.mean(schools.total_score.values[valid_idx.values], labels=schools.faz_id.values[valid_idx.values], index=fazes.index)
    return pd.Series(res, index=fazes.index).fillna(0)
 
@orca.column('fazes', 'building_sqft', cache=True, cache_scope='iteration')
def building_sqft(fazes, buildings):
    return buildings.sqft_per_unit.groupby(buildings.faz_id).sum().\
           reindex(fazes.index).fillna(0)

@orca.column('fazes', 'DU_SF_19', cache=True, cache_scope='iteration')
def DU_SF_19(fazes, buildings):
   print 'In DU_SF_19'
   return buildings.query('building_type_id == 19').residential_units.\
          groupby((buildings.faz_id).sum().reindex(fazes.index).fillna(0)

@orca.column('fazes', 'nonres_sqft', cache=True, cache_scope='iteration')
def nonres_sqft(fazes, buildings):
    return buildings.non_residential_sqft.groupby(buildings.faz_id).sum().\
           reindex(fazes.index).fillna(0)

@orca.column('fazes', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(fazes, households):
    return households.persons.groupby(households.faz_id).size().\
           reindex(fazes.index).fillna(0)

@orca.column('fazes', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(fazes, jobs):
    return jobs.sector_id.groupby(jobs.faz_id).size().\
           reindex(fazes.index).fillna(0)
		   
@orca.column('fazes', 'population', cache=True, cache_scope='iteration')
def population(fazes, households):
    return households.persons.groupby(households.faz_id).sum().\
           reindex(fazes.index).fillna(0)
		   
@orca.column('fazes', 'residential_units', cache=True, cache_scope='iteration')
def residetial_units(fazes, buildings):
    return buildings.residential_units.groupby(buildings.faz_id).sum().\
           reindex(fazes.index).fillna(0)
