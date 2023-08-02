import orca

#####################
# Controls VARIABLES
#####################

@orca.column('controls', 'activity_units', cache=True, cache_scope='iteration')
def activity_units(controls, households, jobs):
    return controls.population + controls.number_of_jobs

@orca.column('controls', 'nonres_sqft', cache=True, cache_scope='iteration')
def nonres_sqft(controls, buildings):
    return buildings.non_residential_sqft.groupby(buildings.control_id).sum().\
           reindex(controls.index).fillna(0)

@orca.column('controls', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(controls, households):
    return households.persons.groupby(households.control_id).size().\
           reindex(controls.index).fillna(0)

@orca.column('controls', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(controls, jobs):
    return jobs.sector_id.groupby(jobs.control_id).size().\
           reindex(controls.index).fillna(0)
		   
@orca.column('controls', 'population', cache=True, cache_scope='iteration')
def population(controls, households):
    return households.persons.groupby(households.control_id).sum().\
           reindex(controls.index).fillna(0)
		   
@orca.column('controls', 'residential_units', cache=True, cache_scope='iteration')
def residetial_units(controls, buildings):
    return buildings.residential_units.groupby(buildings.control_id).sum().\
           reindex(controls.index).fillna(0)


@orca.column('controls', 'Con_Res', cache=True, cache_scope='iteration')
def Con_Res(controls, parcels):
    return parcels.Con_Res.groupby(parcels.control_id).sum().reindex(controls.index).fillna(0)
		
@orca.column('controls', 'Manuf_WTU', cache=True, cache_scope='iteration')
def Manuf_WTU(controls, parcels):
    return parcels.Manuf_WTU.groupby(parcels.control_id).sum().reindex(controls.index).fillna(0)

@orca.column('controls', 'Retail', cache=True, cache_scope='iteration')
def Retail(controls, parcels):
    return parcels.Retail.groupby(parcels.control_id).sum().reindex(controls.index).fillna(0)

@orca.column('controls', 'FIRES', cache=True, cache_scope='iteration')
def FIRES(controls, parcels):
    return parcels.FIRES.groupby(parcels.control_id).sum().reindex(controls.index).fillna(0)
	
@orca.column('controls', 'Gov', cache=True, cache_scope='iteration')
def Gov(controls, parcels):
    return parcels.Gov.groupby(parcels.control_id).sum().reindex(controls.index).fillna(0)
		
@orca.column('controls', 'Edu', cache=True, cache_scope='iteration')
def Edu(controls, parcels):
    return parcels.Edu.groupby(parcels.control_id).sum().reindex(controls.index).fillna(0)
	
