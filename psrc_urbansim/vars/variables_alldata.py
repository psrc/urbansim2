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
    return pd.Series(households.persons.size, index = alldata.index)

@orca.column('alldata', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(alldata, jobs):
    print 'in variables_alldata.py, in number_of_jobs function'
    return pd.Series(jobs.number_of_jobs.sum(), index = alldata.index)

@orca.column('alldata', 'population', cache=True, cache_scope='iteration')
def number_of_households(alldata, households):
    print 'in variables_alldata.py, in number_of_households function'
    return  pd.Series(households.persons.sum(), index = alldata.index)

@orca.column('alldata', 'residential_units', cache=True, cache_scope='iteration')
def residetial_units(alldata, buildings):
    print 'in variables_alldata.py, residential_units function'
    return pd.Series(buildings.residential_units.sum(), index = alldata.index)