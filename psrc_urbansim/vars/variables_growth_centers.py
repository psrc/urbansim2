import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# GROWTH CENTERS VARIABLES (in alphabetic order)
#####################

@orca.column('growth_centers', 'building_sqft', cache=True, cache_scope='iteration')
def building_sqft(growth_centers, buildings):
    return buildings.sqft_per_unit.groupby(buildings.growth_center_id).sum().\
           reindex(growth_centers.index).fillna(0)

@orca.column('growth_centers', 'nonres_sqft', cache=True, cache_scope='iteration')
def nonres_sqft(growth_centers, buildings):
    return buildings.non_residential_sqft.groupby(buildings.growth_center_id).sum().\
           reindex(growth_centers.index).fillna(0)

@orca.column('growth_centers', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(growth_centers, jobs):
    return jobs.sector_id.groupby(jobs.growth_center_id).size().\
           reindex(growth_centers.index).fillna(0)

@orca.column('growth_centers', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(growth_centers, households):
    return households.persons.groupby(households.growth_center_id).size().\
           reindex(growth_centers.index).fillna(0)
	   
@orca.column('growth_centers', 'population', cache=True, cache_scope='iteration')
def population(growth_centers, households):
    return households.persons.groupby(households.growth_center_id).sum().\
           reindex(growth_centers.index).fillna(0)
		   
@orca.column('growth_centers', 'residential_units', cache=True, cache_scope='iteration')
def residetial_units(growth_centers, buildings):
    return buildings.residential_units.groupby(buildings.growth_center_id).sum().\
           reindex(growth_centers.index).fillna(0)
