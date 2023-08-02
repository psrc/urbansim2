import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# COUNTIES VARIABLES 
#####################

@orca.column('counties', 'activity_units', cache=True, cache_scope='iteration')
def activity_units(counties):
    return counties.population + counties.number_of_jobs

@orca.column('counties', 'Business_Services', cache=True, cache_scope='iteration')
def Business_Services(counties, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 7)).groupby(jobs.county_id).sum().\
	        reindex(counties.index).fillna(0)

@orca.column('counties', 'Con_Res', cache=True, cache_scope='iteration')
def Con_Res(counties):
    return counties.Natural_resources + counties.Construction

@orca.column('counties', 'Construction', cache=True, cache_scope='iteration')
def Construction(counties, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 2)).groupby(jobs.county_id).sum().\
	        reindex(counties.index).fillna(0)

@orca.column('counties', 'Edu', cache=True, cache_scope='iteration')
def Edu(counties):
    return counties.edu + counties.Private_Ed
	
@orca.column('counties', 'edu', cache=True, cache_scope='iteration')
def edu(counties, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 13)).groupby(jobs.county_id).sum().\
	        reindex(counties.index).fillna(0)

@orca.column('counties', 'FIRES', cache=True, cache_scope='iteration')
def FIRES(counties):
    return counties.Business_Services + counties.Healthcare + counties.Personal_Services

@orca.column('counties', 'Food_Services', cache=True, cache_scope='iteration')
def Food_Services(counties, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 10)).groupby(jobs.county_id).sum().\
	        reindex(counties.index).fillna(0)

@orca.column('counties', 'Gov', cache=True, cache_scope='iteration')
def Gov(counties):
    return counties.government

@orca.column('counties', 'government', cache=True, cache_scope='iteration')
def government(counties, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 12)).groupby(jobs.county_id).sum().\
	        reindex(counties.index).fillna(0)

@orca.column('counties', 'Healthcare', cache=True, cache_scope='iteration')
def Healthcare(counties, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 9)).groupby(jobs.county_id).sum().\
	        reindex(counties.index).fillna(0)


@orca.column('counties', 'Manuf', cache=True, cache_scope='iteration')
def Manuf(counties, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 3)).groupby(jobs.county_id).sum().\
	        reindex(counties.index).fillna(0)

@orca.column('counties', 'Manuf_WTU', cache=True, cache_scope='iteration')
def Manuf_WTU(counties):
    return counties.Manuf + counties.WTU

@orca.column('counties', 'Natural_resources', cache=True, cache_scope='iteration')
def Natural_resources(counties, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 1)).groupby(jobs.county_id).sum().\
	        reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_sqft', cache=True, cache_scope='iteration')
def nonres_sqft(counties, buildings):
    return buildings.non_residential_sqft.groupby(buildings.county_id).sum().\
           reindex(counties.index).fillna(0)

@orca.column('counties', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(counties, households):
    return households.persons.groupby(households.control_id).size().\
           reindex(counties.index).fillna(0)

@orca.column('counties', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(counties, jobs):
    return jobs.sector_id.groupby(jobs.control_id).size().\
           reindex(counties.index).fillna(0)
		   
@orca.column('counties', 'Personal_Services', cache=True, cache_scope='iteration')
def Personal_Services(counties, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 11)).groupby(jobs.county_id).sum().\
	        reindex(counties.index).fillna(0)
		   
@orca.column('counties', 'population', cache=True, cache_scope='iteration')
def population(counties, households):
    return households.persons.groupby(households.control_id).sum().\
           reindex(counties.index).fillna(0)
		   
@orca.column('counties', 'Private_Ed', cache=True, cache_scope='iteration')
def Private_Ed(counties, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 8)).groupby(jobs.county_id).sum().\
	        reindex(counties.index).fillna(0)

@orca.column('counties', 'residential_units', cache=True, cache_scope='iteration')
def residetial_units(counties, buildings):
    return buildings.residential_units.groupby(buildings.control_id).sum().\
           reindex(counties.index).fillna(0)

@orca.column('counties', 'Retail_only', cache=True, cache_scope='iteration')
def Retail_only(counties, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 5)).groupby(jobs.county_id).sum().\
	        reindex(counties.index).fillna(0)

@orca.column('counties', 'Retail', cache=True, cache_scope='iteration')
def Retail(counties):
    return counties.Retail_only + counties.Food_Services

@orca.column('counties', 'WTU', cache=True, cache_scope='iteration')
def WTU(counties, jobs):
    return (jobs.number_of_jobs *(jobs.sector_id == 4)).groupby(jobs.county_id).sum().\
	        reindex(counties.index).fillna(0)


@orca.column('counties', 'nonres_3_all', cache=True, cache_scope='iteration')
def nonres_3_all(counties, buildings):
    return ((buildings.job_spaces) * (buildings.building_type_id == 3)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_3_spaces', cache=True, cache_scope='iteration')
def nonres_3_spaces(counties, buildings):
    return (buildings.job_spaces * (buildings.building_type_id == 3)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_3_sqft', cache=True, cache_scope='iteration')
def nonres_3_sqft(counties, buildings):
    return (buildings.non_residential_sqft * (buildings.building_type_id == 3)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_3_vac', cache=True, cache_scope='iteration')
def nonres_3_vac(counties, buildings):
    return ((buildings.vacant_job_spaces) * (buildings.building_type_id == 3)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_3_VR', cache=True, cache_scope='iteration')
def nonres_3_VR(counties):
    return counties.nonres_3_vac / counties.nonres_3_all


@orca.column('counties', 'nonres_8_all', cache=True, cache_scope='iteration')
def nonres_8_all(counties, buildings):
    return ((buildings.job_spaces) * (buildings.building_type_id == 8)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_8_spaces', cache=True, cache_scope='iteration')
def nonres_8_spaces(counties, buildings):
    return (buildings.job_spaces * (buildings.building_type_id == 8)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_8_sqft', cache=True, cache_scope='iteration')
def nonres_8_sqft(counties, buildings):
    return (buildings.non_residential_sqft * (buildings.building_type_id == 8)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_8_vac', cache=True, cache_scope='iteration')
def nonres_8_vac(counties, buildings):
    return ((buildings.vacant_job_spaces) * (buildings.building_type_id == 8)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_8_VR', cache=True, cache_scope='iteration')
def nonres_8_VR(counties):
    return counties.nonres_8_vac / counties.nonres_8_all


@orca.column('counties', 'nonres_13_all', cache=True, cache_scope='iteration')
def nonres_13_all(counties, buildings):
    return ((buildings.job_spaces) * (buildings.building_type_id == 13)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_13_spaces', cache=True, cache_scope='iteration')
def nonres_13_spaces(counties, buildings):
    return (buildings.job_spaces * (buildings.building_type_id == 13)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_13_sqft', cache=True, cache_scope='iteration')
def nonres_13_sqft(counties, buildings):
    return (buildings.non_residential_sqft * (buildings.building_type_id == 13)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_13_vac', cache=True, cache_scope='iteration')
def nonres_13_vac(counties, buildings):
    return ((buildings.vacant_job_spaces) * (buildings.building_type_id == 13)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_13_VR', cache=True, cache_scope='iteration')
def nonres_13_VR(counties):
    return counties.nonres_13_vac / counties.nonres_13_all


@orca.column('counties', 'nonres_20_all', cache=True, cache_scope='iteration')
def nonres_20_all(counties, buildings):
    return ((buildings.job_spaces) * (buildings.building_type_id == 20)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_20_spaces', cache=True, cache_scope='iteration')
def nonres_20_spaces(counties, buildings):
    return (buildings.job_spaces * (buildings.building_type_id == 20)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_20_sqft', cache=True, cache_scope='iteration')
def nonres_20_sqft(counties, buildings):
    return (buildings.non_residential_sqft * (buildings.building_type_id == 20)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_20_vac', cache=True, cache_scope='iteration')
def nonres_20_vac(counties, buildings):
    return ((buildings.vacant_job_spaces) * (buildings.building_type_id == 20)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_20_VR', cache=True, cache_scope='iteration')
def nonres_20_VR(counties):
    return counties.nonres_20_vac / counties.nonres_20_all


@orca.column('counties', 'nonres_21_all', cache=True, cache_scope='iteration')
def nonres_21_all(counties, buildings):
    return ((buildings.job_spaces) * (buildings.building_type_id == 21)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_21_spaces', cache=True, cache_scope='iteration')
def nonres_21_spaces(counties, buildings):
    return (buildings.job_spaces * (buildings.building_type_id == 21)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_21_sqft', cache=True, cache_scope='iteration')
def nonres_21_sqft(counties, buildings):
    return (buildings.non_residential_sqft * (buildings.building_type_id == 21)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_21_vac', cache=True, cache_scope='iteration')
def nonres_21_vac(counties, buildings):
    return ((buildings.vacant_job_spaces) * (buildings.building_type_id == 21)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_21_VR', cache=True, cache_scope='iteration')
def nonres_21_VR(counties):
    return counties.nonres_21_vac / counties.nonres_21_all


@orca.column('counties', 'res_4_all', cache=True, cache_scope='iteration')
def res_4_all(counties, buildings):
    return ((buildings.residential_units) * (buildings.building_type_id == 4)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'res_4_units', cache=True, cache_scope='iteration')
def res_4_units(counties, buildings):
    return (buildings.residential_units * (buildings.building_type_id == 4)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'res_4_vac', cache=True, cache_scope='iteration')
def res_4_vac(counties, buildings):
    return ((buildings.vacant_residential_units) * (buildings.building_type_id == 4)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'res_4_VR', cache=True, cache_scope='iteration')
def res_4_VR(counties, buildings):
    return counties.res_4_vac / counties.res_4_all


@orca.column('counties', 'res_12_all', cache=True, cache_scope='iteration')
def res_12_all(counties, buildings):
    return ((buildings.residential_units) * (buildings.building_type_id == 12)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'res_12_units', cache=True, cache_scope='iteration')
def res_12_units(counties, buildings):
    return (buildings.residential_units * (buildings.building_type_id == 12)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'res_12_vac', cache=True, cache_scope='iteration')
def res_12_vac(counties, buildings):
    return ((buildings.vacant_residential_units) * (buildings.building_type_id == 12)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'res_12_VR', cache=True, cache_scope='iteration')
def res_12_VR(counties, buildings):
    return counties.res_12_vac / counties.res_12_all


@orca.column('counties', 'res_19_all', cache=True, cache_scope='iteration')
def res_19_all(counties, buildings):
    return ((buildings.residential_units) * (buildings.building_type_id == 19)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'res_19_units', cache=True, cache_scope='iteration')
def res_19_units(counties, buildings):
    return (buildings.residential_units * (buildings.building_type_id == 19)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'res_19_vac', cache=True, cache_scope='iteration')
def res_19_vac(counties, buildings):
    return ((buildings.vacant_residential_units) * (buildings.building_type_id == 19)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'res_19_VR', cache=True, cache_scope='iteration')
def res_19_VR(counties, buildings):
    return counties.res_19_vac / counties.res_19_all