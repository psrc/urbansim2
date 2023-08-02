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


@orca.column('control_hcts', 'Con_Res', cache=True, cache_scope='iteration')
def Con_Res(control_hcts, parcels):
    return parcels.Con_Res.groupby(parcels.control_hct_id).sum().reindex(control_hcts.index).fillna(0)
		
@orca.column('control_hcts', 'Manuf_WTU', cache=True, cache_scope='iteration')
def Manuf_WTU(control_hcts, parcels):
    return parcels.Manuf_WTU.groupby(parcels.control_hct_id).sum().reindex(control_hcts.index).fillna(0)

@orca.column('control_hcts', 'Retail', cache=True, cache_scope='iteration')
def Retail(control_hcts, parcels):
    return parcels.Retail.groupby(parcels.control_hct_id).sum().reindex(control_hcts.index).fillna(0)

@orca.column('control_hcts', 'FIRES', cache=True, cache_scope='iteration')
def FIRES(control_hcts, parcels):
    return parcels.FIRES.groupby(parcels.control_hct_id).sum().reindex(control_hcts.index).fillna(0)
	
@orca.column('control_hcts', 'Gov', cache=True, cache_scope='iteration')
def Gov(control_hcts, parcels):
    return parcels.Gov.groupby(parcels.control_hct_id).sum().reindex(control_hcts.index).fillna(0)
		
@orca.column('control_hcts', 'Edu', cache=True, cache_scope='iteration')
def Edu(control_hcts, parcels):
    return parcels.Edu.groupby(parcels.control_hct_id).sum().reindex(control_hcts.index).fillna(0)
	
