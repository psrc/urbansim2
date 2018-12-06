import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# ALLDATA VARIABLES (in alphabetic order)
#####################

@orca.column('alldata', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(alldata, households):
    print 'in variables_alldata.py, in number_of_households function'
    df = pd.DataFrame.from_dict({'households': households.persons.size,'alldata_id': [1]})
    df = df.set_index('alldata_id')
    return df.households

@orca.column('alldata', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(alldata, jobs):
    print 'in variables_alldata.py, in number_of_jobs function'
    df = pd.DataFrame.from_dict({'jobs': jobs.number_of_jobs.sum(),'alldata_id': [1]})
    df = df.set_index('alldata_id')
    return df.jobs

@orca.column('alldata', 'population', cache=True, cache_scope='iteration')
def number_of_households(households):
    print 'in variables_alldata.py, in number_of_households function'
    df = pd.DataFrame.from_dict({'population': households.persons.sum(),
    'alldata_id': [1]})
    df = df.set_index('alldata_id')
    return df.population

@orca.column('alldata', 'residential_units', cache=True, cache_scope='iteration')
def residetial_units(buildings):
    print 'in variables_alldata.py, residential_units function'
    df = pd.DataFrame.from_dict({'residential_units': buildings.residential_units.sum(),
    'alldata_id': [1]})
    df = df.set_index('alldata_id')
    return df.residential_units