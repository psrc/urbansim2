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
    res = ndi.mean(schools.total_score.values[valid_idx], labels=schools.faz_id[valid_idx], index=fazes.index)
    return res.fillna(0)
    
@orca.column('fazes', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(fazes, households):
    return households.persons.groupby(households.faz_id).size().\
           reindex(fazes.index).fillna(0)

@orca.column('fazes', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(fazes, jobs):
    return jobs.sector_id.groupby(jobs.faz_id).size().\
           reindex(fazes.index).fillna(0)
