import orca

#####################
# Targets VARIABLES
#####################

@orca.column('targets', 'activity_units', cache=True, cache_scope='iteration')
def activity_units(targets, households, jobs):
    return targets.population + targets.number_of_jobs

@orca.column('targets', 'nonres_sqft', cache=True, cache_scope='iteration')
def nonres_sqft(targets, buildings):
    return buildings.non_residential_sqft.groupby(buildings.target_id).sum().\
           reindex(targets.index).fillna(0)

@orca.column('targets', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(targets, households):
    return households.persons.groupby(households.target_id).size().\
           reindex(targets.index).fillna(0)

@orca.column('targets', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(targets, jobs):
    return jobs.sector_id.groupby(jobs.target_id).size().\
           reindex(targets.index).fillna(0)
		   
@orca.column('targets', 'population', cache=True, cache_scope='iteration')
def population(targets, households):
    return households.persons.groupby(households.target_id).sum().\
           reindex(targets.index).fillna(0)
		   
@orca.column('targets', 'residential_units', cache=True, cache_scope='iteration')
def residetial_units(targets, buildings):
    return buildings.residential_units.groupby(buildings.target_id).sum().\
           reindex(targets.index).fillna(0)

@orca.column('targets', 'Con_Res', cache=True, cache_scope='iteration')
def Con_Res(targets, parcels):
    return parcels.Con_Res.groupby(parcels.target_id).sum().reindex(targets.index).fillna(0)
		
@orca.column('targets', 'Manuf_WTU', cache=True, cache_scope='iteration')
def Manuf_WTU(targets, parcels):
    return parcels.Manuf_WTU.groupby(parcels.target_id).sum().reindex(targets.index).fillna(0)

@orca.column('targets', 'Retail', cache=True, cache_scope='iteration')
def Retail(targets, parcels):
    return parcels.Retail.groupby(parcels.target_id).sum().reindex(targets.index).fillna(0)

@orca.column('targets', 'FIRES', cache=True, cache_scope='iteration')
def FIRES(targets, parcels):
    return parcels.FIRES.groupby(parcels.target_id).sum().reindex(targets.index).fillna(0)
	
@orca.column('targets', 'Gov', cache=True, cache_scope='iteration')
def Gov(targets, parcels):
    return parcels.Gov.groupby(parcels.target_id).sum().reindex(targets.index).fillna(0)
		
@orca.column('targets', 'Edu', cache=True, cache_scope='iteration')
def Edu(targets, parcels):
    return parcels.Edu.groupby(parcels.target_id).sum().reindex(targets.index).fillna(0)
	
