import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# CITIES VARIABLES (in alphabetic order)
#####################

@orca.column('cities', 'building_sqft', cache=True, cache_scope='iteration')
def building_sqft(cities, buildings):
    print 'in variables_city.py, in buildings_sqft function'
    return buildings.sqft_per_unit.groupby(buildings.city_id).sum().\
           reindex(cities.index).fillna(0)

@orca.column('cities', 'nonres_sqft', cache=True, cache_scope='iteration')
def nonres_sqft(cities, buildings):
    print 'in variables_city.py, in nonres_sqft function'
    return buildings.non_residential_sqft.groupby(buildings.city_id).sum().\
           reindex(cities.index).fillna(0)
 
@orca.column('cities', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(cities, households):
    print 'in variables_city.py, in number_of_households function'
    return households.persons.groupby(households.city_id).size().\
           reindex(cities.index).fillna(0)

@orca.column('cities', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(cities, jobs):
    print 'in variables_city.py, in number_of_jobs function'
    return jobs.sector_id.groupby(jobs.city_id).size().\
           reindex(cities.index).fillna(0)
		   
@orca.column('cities', 'population', cache=True, cache_scope='iteration')
def population(cities, households):
    print 'in variables_city.py, inpopulation function'
    return households.persons.groupby(households.city_id).sum().\
           reindex(cities.index).fillna(0)
		   
@orca.column('cities', 'residential_units', cache=True, cache_scope='iteration')
def residetial_units(cities, buildings):
    print 'in variables_city.py, residential_units function'
    return buildings.residential_units.groupby(buildings.city_id).sum().\
           reindex(cities.index).fillna(0)