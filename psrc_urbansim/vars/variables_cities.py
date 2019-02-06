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

@orca.column('cities', 'Business_Services', cache=True, cache_scope='iteration')
def Business_Services(cities, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 7)).groupby(jobs.city_id).sum().\
	        reindex(cities.index).fillna(0)

@orca.column('cities', 'Construction', cache=True, cache_scope='iteration')
def Construction(cities, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 2)).groupby(jobs.city_id).sum().\
	        reindex(cities.index).fillna(0)
			
@orca.column('cities', 'edu', cache=True, cache_scope='iteration')
def edu(cities, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 13)).groupby(jobs.city_id).sum().\
	        reindex(cities.index).fillna(0)

@orca.column('cities', 'Food_Services', cache=True, cache_scope='iteration')
def Food_Services(cities, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 10)).groupby(jobs.city_id).sum().\
	        reindex(cities.index).fillna(0)

@orca.column('cities', 'government', cache=True, cache_scope='iteration')
def government(cities, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 12)).groupby(jobs.city_id).sum().\
	        reindex(cities.index).fillna(0)

@orca.column('cities', 'Healthcare', cache=True, cache_scope='iteration')
def Healthcare(cities, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 9)).groupby(jobs.city_id).sum().\
	        reindex(cities.index).fillna(0)

@orca.column('cities', 'Manuf', cache=True, cache_scope='iteration')
def Manuf(cities, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 3)).groupby(jobs.city_id).sum().\
	        reindex(cities.index).fillna(0)


@orca.column('cities', 'Natural_resources', cache=True, cache_scope='iteration')
def Natural_resources(cities, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 1)).groupby(jobs.city_id).sum().\
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

@orca.column('cities', 'Personal_Services', cache=True, cache_scope='iteration')
def Personal_Services(cities, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 11)).groupby(jobs.city_id).sum().\
	        reindex(cities.index).fillna(0)
		   
@orca.column('cities', 'population', cache=True, cache_scope='iteration')
def population(cities, households):
    print 'in variables_city.py, inpopulation function'
    return households.persons.groupby(households.city_id).sum().\
           reindex(cities.index).fillna(0)

@orca.column('cities', 'Private_Ed', cache=True, cache_scope='iteration')
def Private_Ed(cities, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 8)).groupby(jobs.city_id).sum().\
	        reindex(cities.index).fillna(0)
		   
@orca.column('cities', 'residential_units', cache=True, cache_scope='iteration')
def residetial_units(cities, buildings):
    print 'in variables_city.py, residential_units function'
    return buildings.residential_units.groupby(buildings.city_id).sum().\
           reindex(cities.index).fillna(0)

@orca.column('cities', 'Retail', cache=True, cache_scope='iteration')
def Retail(cities, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 5)).groupby(jobs.city_id).sum().\
	        reindex(cities.index).fillna(0)

@orca.column('cities', 'WTU', cache=True, cache_scope='iteration')
def WTU(cities, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 4)).groupby(jobs.city_id).sum().\
	        reindex(cities.index).fillna(0)