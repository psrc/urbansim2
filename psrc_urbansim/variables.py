import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import psrc_urbansim.dataset
import urbansim_defaults.utils

#####################
# PARCELS VARIABLES
#####################

@orca.column('parcels', 'residential_units', cache=True, cache_scope='iteration')
def residential_units(parcels, buildings):
    return buildings.residential_units.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'building_sqft', cache=True, cache_scope='iteration')
def building_sqft(parcels, buildings):
    return buildings.building_sqft.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'total_improvement_value', cache=True, cache_scope='iteration')
def total_improvement_value(parcels, buildings):
    return buildings.improvement_value.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'total_land_value_per_sqft', cache=True, cache_scope='iteration')
def total_land_value_per_sqft(parcels):
    return ((parcels.land_value + parcels.total_improvement_value)/parcels.parcel_sqft).replace(np.inf, 0).fillna(0)

@orca.column('parcels', 'invfar', cache=True, cache_scope='iteration')
def invfar(parcels):
    return (parcels.parcel_sqft.astype(float)/parcels.building_sqft.astype(float)).replace(np.inf, 0).fillna(0)

@orca.column('parcels', 'is_park', cache=True, cache_scope='iteration')
def is_park(parcels):
    return (parcels.land_use_type_id == 19)

@orca.column('parcels', 'park_area', cache=True, cache_scope='iteration')
def park_area(parcels):
    return ((parcels.land_use_type_id == 19) * parcels.parcel_sqft)

@orca.column('parcels', 'average_income', cache=True, cache_scope='iteration')
def average_income(parcels, households):
    return households.income.groupby(households.parcel_id).mean().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(parcels, households):
    return households.persons.groupby(households.parcel_id).size().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(parcels, jobs):
    return jobs.sector_id.groupby(jobs.parcel_id).size().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'faz_id', cache=True)
def faz_id(parcels, zones):
    return misc.reindex(zones.faz_id, parcels.zone_id)

#####################
# BUILDINGS VARIABLES
#####################

@orca.column('buildings', 'building_sqft', cache=True, cache_scope='iteration')
def building_sqft(buildings):
    results = np.zeros(buildings.local.shape[0],dtype=np.int32)
    where_res = np.where(buildings.residential_units > 0)[0]
    results[where_res] = buildings.residential_units[where_res] * buildings.sqft_per_unit[where_res]
    where_nonres = np.where(buildings.non_residential_sqft > 0)[0]
    results[where_nonres] = results[where_nonres] + buildings.non_residential_sqft[where_nonres]
    return pd.Series(results)

@orca.column('buildings', 'zone_id', cache=True)
def zone_id(buildings, parcels):
    return misc.reindex(parcels.zone_id, buildings.parcel_id)

@orca.column('buildings', 'faz_id', cache=True)
def faz_id(buildings, zones):
    return misc.reindex(zones.faz_id, buildings.zone_id)

@orca.column('buildings', 'tractcity_id', cache=True)
def tractcity_id(buildings, parcels):
    return misc.reindex(parcels.tractcity_id, buildings.parcel_id)

#####################
# HOUSEHOLDS VARIABLES
#####################

@orca.column('households', 'parcel_id', cache=True)
def parcel_id(households, buildings):
    return misc.reindex(buildings.parcel_id, households.building_id)

@orca.column('households', 'zone_id', cache=True)
def zone_id(households, buildings):
    return misc.reindex(buildings.zone_id, households.building_id)

@orca.column('households', 'faz_id', cache=True)
def faz_id(households, zones):
    return misc.reindex(zones.faz_id, households.zone_id)

@orca.column('households', 'tractcity_id', cache=True)
def tractcity_id(households, parcels):
    return misc.reindex(parcels.tractcity_id, households.parcel_id)

#####################
# PERSONS VARIABLES
#####################

@orca.column('persons', 'parcel_id', cache=True)
def parcel_id(persons, households):
    return misc.reindex(households.parcel_id, persons.household_id)

@orca.column('persons', 'zone_id', cache=True)
def zone_id(persons, households):
    return misc.reindex(households.zone_id, persons.household_id)

@orca.column('persons', 'faz_id', cache=True)
def faz_id(persons, zones):
    return misc.reindex(zones.faz_id, persons.zone_id)

@orca.column('persons', 'tractcity_id', cache=True)
def tractcity_id(persons, households):
    return misc.reindex(households.tractcity_id, persons.household_id)

#####################
# JOBS VARIABLES
#####################

@orca.column('jobs', 'parcel_id', cache=True)
def parcel_id(jobs, buildings):
    return misc.reindex(buildings.parcel_id, jobs.building_id)

@orca.column('jobs', 'zone_id', cache=True)
def zone_id(jobs, buildings):
    return misc.reindex(buildings.zone_id, jobs.building_id)

@orca.column('jobs', 'faz_id', cache=True)
def faz_id(jobs, zones):
    return misc.reindex(zones.faz_id, jobs.zone_id)

@orca.column('jobs', 'tractcity_id', cache=True)
def tractcity_id(jobs, parcels):
    return misc.reindex(parcels.tractcity_id, jobs.parcel_id)

#####################
# ZONES VARIABLES
#####################

@orca.column('zones', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(zones, households):
    return households.persons.groupby(households.zone_id).size().\
           reindex(zones.index).fillna(0)

@orca.column('zones', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(zones, jobs):
    return jobs.sector_id.groupby(jobs.zone_id).size().\
           reindex(zones.index).fillna(0)


#####################
# FAZES VARIABLES
#####################

@orca.column('fazes', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(fazes, households):
    return households.persons.groupby(households.faz_id).size().\
           reindex(fazes.index).fillna(0)

@orca.column('fazes', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(fazes, jobs):
    return jobs.sector_id.groupby(jobs.faz_id).size().\
           reindex(fazes.index).fillna(0)


#####################
# TRACT-CITY VARIABLES
#####################

@orca.column('tractcity', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(tractcity, parcels):
    return parcels.number_of_households.groupby(parcels.tractcity_id).sum().\
           reindex(tractcity.index).fillna(0)

@orca.column('tractcity', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(tractcity, parcels):
    return parcels.number_of_jobs.groupby(parcels.tractcity_id).sum().\
           reindex(tractcity.index).fillna(0)
