import pandas as pd
import numpy as np
import scipy.ndimage as ndi
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# FAZES VARIABLES
#####################
@orca.column('fazes', 'avg_school_score', cache=True, cache_scope='iteration')
def avg_school_score(fazes, schools):
    """
    Computes the average of the school's total score over FAZes, where missing values are removed 
    from the computation. Missing values are those that are less or equal zero.
    """    
    valid_idx = schools.total_score > 0
    if valid_idx.sum() > 0:
        res = ndi.mean(schools.total_score.values[valid_idx.values], labels=schools.faz_id.values[valid_idx.values], index=fazes.index)
    else:
        res = 0.0
    return pd.Series(res, index=fazes.index).fillna(0)
 
@orca.column('fazes', 'building_sqft', cache=True, cache_scope='iteration')
def building_sqft(fazes, buildings):
    return buildings.sqft_per_unit.groupby(buildings.faz_id).sum().\
           reindex(fazes.index).fillna(0)

@orca.column('fazes', 'DU_CO_4', cache=True, cache_scope='iteration')
def DU_CO_4(fazes, buildings):
    return (buildings.residential_units *(buildings.building_type_id == 4)).\
          groupby(buildings.faz_id).sum().reindex(fazes.index).fillna(0)

@orca.column('fazes', 'DU_MH_11', cache=True, cache_scope='iteration')
def DU_MH_11(fazes, buildings):
    return (buildings.residential_units *(buildings.building_type_id == 11)).\
          groupby(buildings.faz_id).sum().reindex(fazes.index).fillna(0)

@orca.column('fazes', 'DU_MF_12', cache=True, cache_scope='iteration')
def DU_MF_12(fazes, buildings):
    return (buildings.residential_units *(buildings.building_type_id == 12)).\
          groupby(buildings.faz_id).sum().reindex(fazes.index).fillna(0)

@orca.column('fazes', 'DU_SF_19', cache=True, cache_scope='iteration')
def DU_SF_19(fazes, buildings):
    return (buildings.residential_units *(buildings.building_type_id == 19)).\
          groupby(buildings.faz_id).sum().reindex(fazes.index).fillna(0)

@orca.column('fazes', 'DU_Total', cache=True, cache_scope='iteration')
def DU_Total(fazes):
    return fazes.residential_units

@orca.column('fazes', 'HH_CO_4', cache=True, cache_scope='iteration')
def HH_CO_4(fazes, households):
    return ((households.persons > 0) *(households.building_type_id == 4)).\
          groupby(households.faz_id).sum().reindex(fazes.index).fillna(0)   

@orca.column('fazes', 'HH_MF_12', cache=True, cache_scope='iteration')
def HH_MF_12(fazes, households):
    return ((households.persons > 0) *(households.building_type_id == 12)).\
          groupby(households.faz_id).sum().reindex(fazes.index).fillna(0)

@orca.column('fazes', 'HH_SF_19', cache=True, cache_scope='iteration')
def HH_SF_19(fazes, households):
    return ((households.persons > 0) *(households.building_type_id == 19)).\
          groupby(households.faz_id).sum().reindex(fazes.index).fillna(0)

@orca.column('fazes', 'HH_MH_11', cache=True, cache_scope='iteration')
def HH_MH_11(fazes, households):
    return ((households.persons > 0) *(households.building_type_id == 11)).\
          groupby(households.faz_id).sum().reindex(fazes.index).fillna(0)

@orca.column('fazes', 'HH_Total', cache=True, cache_scope='iteration')
def HH_Total(fazes):
    return fazes.number_of_households

@orca.column('fazes', 'max_developable_capacity', cache=True, cache_scope='iteration')
def max_developable_capacity(fazes, parcels):
    return parcels.max_developable_capacity.groupby(parcels.faz_id).sum().\
	        reindex(fazes.index).fillna(0)

@orca.column('fazes', 'max_developable_nonresidential_capacity', cache=True, cache_scope='iteration')
def max_developable_nonresidential_capacity(fazes, parcels):
    return parcels.max_developable_nonresidential_capacity.groupby(parcels.faz_id).sum().\
	        reindex(fazes.index).fillna(0)

@orca.column('fazes', 'max_developable_residential_capacity', cache=True, cache_scope='iteration')
def max_developable_residential_capacity(fazes, parcels):
    return parcels.max_developable_residential_capacity.groupby(parcels.faz_id).sum().\
	        reindex(fazes.index).fillna(0)

@orca.column('fazes', 'nonres_sqft', cache=True, cache_scope='iteration')
def nonres_sqft(fazes, buildings):
    return buildings.non_residential_sqft.groupby(buildings.faz_id).sum().\
           reindex(fazes.index).fillna(0)

@orca.column('fazes', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(fazes, households):
    return households.persons.groupby(households.faz_id).size().\
           reindex(fazes.index).fillna(0)

@orca.column('fazes', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(fazes, jobs):
    return jobs.sector_id.groupby(jobs.faz_id).size().\
           reindex(fazes.index).fillna(0)
		   
@orca.column('fazes', 'population', cache=True, cache_scope='iteration')
def population(fazes, households):
    return households.persons.groupby(households.faz_id).sum().\
           reindex(fazes.index).fillna(0)
		   
@orca.column('fazes', 'residential_units', cache=True, cache_scope='iteration')
def residetial_units(fazes, buildings):
    return buildings.residential_units.groupby(buildings.faz_id).sum().\
           reindex(fazes.index).fillna(0)

@orca.column('fazes', 'Con_Res', cache=True, cache_scope='iteration')
def Con_Res(fazes, zones):
    return zones.Con_Res.groupby(zones.faz_id).sum().reindex(fazes.index).fillna(0)
		
@orca.column('fazes', 'Manuf_WTU', cache=True, cache_scope='iteration')
def Manuf_WTU(fazes, zones):
    return zones.Manuf_WTU.groupby(zones.faz_id).sum().reindex(fazes.index).fillna(0)

@orca.column('fazes', 'Retail', cache=True, cache_scope='iteration')
def Retail(fazes, zones):
    return zones.Retail.groupby(zones.faz_id).sum().reindex(fazes.index).fillna(0)

@orca.column('fazes', 'FIRES', cache=True, cache_scope='iteration')
def FIRES(fazes, zones):
    return zones.FIRES.groupby(zones.faz_id).sum().reindex(fazes.index).fillna(0)
	
@orca.column('fazes', 'Gov', cache=True, cache_scope='iteration')
def Gov(fazes, zones):
    return zones.Gov.groupby(zones.faz_id).sum().reindex(fazes.index).fillna(0)
		
@orca.column('fazes', 'Edu', cache=True, cache_scope='iteration')
def Edu(fazes, zones):
    return zones.Edu.groupby(zones.faz_id).sum().reindex(fazes.index).fillna(0)
	
