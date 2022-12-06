import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# SUBREGS VARIABLES (in alphabetic order)
#####################

@orca.column('subregs', 'building_sqft', cache=True, cache_scope='iteration')
def building_sqft(subregs, buildings):
    return buildings.sqft_per_unit.groupby(buildings.subreg_id).sum().\
           reindex(subregs.index).fillna(0)

@orca.column('subregs', 'Business_Services', cache=True, cache_scope='iteration')
def Business_Services(subregs, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 7)).groupby(jobs.subreg_id).sum().\
	        reindex(subregs.index).fillna(0)

@orca.column('subregs', 'Construction', cache=True, cache_scope='iteration')
def Construction(subregs, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 2)).groupby(jobs.subreg_id).sum().\
	        reindex(subregs.index).fillna(0)
			
@orca.column('subregs', 'edu', cache=True, cache_scope='iteration')
def edu(subregs, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 13)).groupby(jobs.subreg_id).sum().\
	        reindex(subregs.index).fillna(0)

@orca.column('subregs', 'Food_Services', cache=True, cache_scope='iteration')
def Food_Services(subregs, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 10)).groupby(jobs.subreg_id).sum().\
	        reindex(subregs.index).fillna(0)

@orca.column('subregs', 'government', cache=True, cache_scope='iteration')
def government(subregs, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 12)).groupby(jobs.subreg_id).sum().\
	        reindex(subregs.index).fillna(0)

@orca.column('subregs', 'Healthcare', cache=True, cache_scope='iteration')
def Healthcare(subregs, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 9)).groupby(jobs.subreg_id).sum().\
	        reindex(subregs.index).fillna(0)

@orca.column('subregs', 'Manuf', cache=True, cache_scope='iteration')
def Manuf(subregs, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 3)).groupby(jobs.subreg_id).sum().\
	        reindex(subregs.index).fillna(0)

@orca.column('subregs', 'max_developable_capacity', cache=True, cache_scope='iteration')
def max_developable_capacity(subregs, parcels):
    return parcels.max_developable_capacity.groupby(parcels.subreg_id).sum().\
	        reindex(subregs.index).fillna(0)

@orca.column('subregs', 'max_developable_nonresidential_capacity', cache=True, cache_scope='iteration')
def max_developable_nonresidential_capacity(subregs, parcels):
    return parcels.max_developable_nonresidential_capacity.groupby(parcels.subreg_id).sum().\
	        reindex(subregs.index).fillna(0)

@orca.column('subregs', 'max_developable_residential_capacity', cache=True, cache_scope='iteration')
def max_developable_residential_capacity(subregs, parcels):
    return parcels.max_developable_residential_capacity.groupby(parcels.subreg_id).sum().\
	        reindex(subregs.index).fillna(0)

@orca.column('subregs', 'Natural_resources', cache=True, cache_scope='iteration')
def Natural_resources(subregs, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 1)).groupby(jobs.subreg_id).sum().\
	        reindex(subregs.index).fillna(0)

@orca.column('subregs', 'nonres_sqft', cache=True, cache_scope='iteration')
def nonres_sqft(subregs, buildings):
    return buildings.non_residential_sqft.groupby(buildings.subreg_id).sum().\
           reindex(subregs.index).fillna(0)
 
@orca.column('subregs', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(subregs, households):
    return households.persons.groupby(households.subreg_id).size().\
           reindex(subregs.index).fillna(0)

@orca.column('subregs', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(subregs, jobs):
    return jobs.sector_id.groupby(jobs.subreg_id).size().\
           reindex(subregs.index).fillna(0)

@orca.column('subregs', 'Personal_Services', cache=True, cache_scope='iteration')
def Personal_Services(subregs, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 11)).groupby(jobs.subreg_id).sum().\
	        reindex(subregs.index).fillna(0)
		   
@orca.column('subregs', 'population', cache=True, cache_scope='iteration')
def population(subregs, households):
    return households.persons.groupby(households.subreg_id).sum().\
           reindex(subregs.index).fillna(0)

@orca.column('subregs', 'Private_Ed', cache=True, cache_scope='iteration')
def Private_Ed(subregs, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 8)).groupby(jobs.subreg_id).sum().\
	        reindex(subregs.index).fillna(0)
		   
@orca.column('subregs', 'residential_units', cache=True, cache_scope='iteration')
def residetial_units(subregs, buildings):
    return buildings.residential_units.groupby(buildings.subreg_id).sum().\
           reindex(subregs.index).fillna(0)

@orca.column('subregs', 'Retail', cache=True, cache_scope='iteration')
def Retail(subregs, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 5)).groupby(jobs.subreg_id).sum().\
	        reindex(subregs.index).fillna(0)

@orca.column('subregs', 'WTU', cache=True, cache_scope='iteration')
def WTU(subregs, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 4)).groupby(jobs.subreg_id).sum().\
	        reindex(subregs.index).fillna(0)