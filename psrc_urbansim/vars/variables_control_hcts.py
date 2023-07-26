import orca

#####################
# Control_hcts VARIABLES
#####################

@orca.column('control_hcts', 'activity_units', cache=True, cache_scope='iteration')
def activity_units(control_hcts, households, jobs):
    return control_hcts.population + control_hcts.number_of_jobs

@orca.column('control_hcts', 'nonres_sqft', cache=True, cache_scope='iteration')
def nonres_sqft(control_hcts, buildings):
    return buildings.non_residential_sqft.groupby(buildings.control_hct_id).sum().\
           reindex(control_hcts.index).fillna(0)

@orca.column('control_hcts', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(control_hcts, households):
    return households.persons.groupby(households.control_hct_id).size().\
           reindex(control_hcts.index).fillna(0)

@orca.column('control_hcts', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(control_hcts, jobs):
    return jobs.sector_id.groupby(jobs.control_hct_id).size().\
           reindex(control_hcts.index).fillna(0)
		   
@orca.column('control_hcts', 'population', cache=True, cache_scope='iteration')
def population(control_hcts, households):
    return households.persons.groupby(households.control_hct_id).sum().\
           reindex(control_hcts.index).fillna(0)
		   
@orca.column('control_hcts', 'residential_units', cache=True, cache_scope='iteration')
def residetial_units(control_hcts, buildings):
    return buildings.residential_units.groupby(buildings.control_hct_id).sum().\
           reindex(control_hcts.index).fillna(0)
