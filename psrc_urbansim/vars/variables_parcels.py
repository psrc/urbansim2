import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# PARCELS VARIABLES (in alphabetic order)
#####################

@orca.column('parcels', 'acres_wwd', cache=True, cache_scope='iteration')
def acres_wwd(parcels):
    return parcels.parcel_sqft_wwd / 43560.0

@orca.column('parcels', 'average_income', cache=True, cache_scope='iteration')
def average_income(parcels, households):
    return households.income.groupby(households.parcel_id).mean().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'building_sqft', cache=True, cache_scope='iteration')
def building_sqft(parcels, buildings):
    return buildings.building_sqft.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'employment_density_wwd', cache=True, cache_scope='step')
def employment_density_wwd(parcels):
    return (parcels.number_of_jobs_wwd / parcels.acres_wwd).replace(np.inf, 0).fillna(0)

@orca.column('parcels', 'existing_units', cache=True, cache_scope='iteration')
def existing_units(parcels):
    results = np.zeros(parcels.local.shape[0], dtype=np.int32)
    for name in ["building_sqft", "parcel_sqft", "residential_units"]:
        w = np.where(parcels.unit_name == name)[0]
        results[w] = parcels[name].iloc[w].astype(np.int32)
    return pd.Series(results, index=parcels.index)

@orca.column('parcels', 'faz_id', cache=True)
def faz_id(parcels, zones):
    return misc.reindex(zones.faz_id, parcels.zone_id)

@orca.column('parcels', 'invfar', cache=True, cache_scope='iteration')
def invfar(parcels):
    return (parcels.parcel_sqft.astype(float)/parcels.building_sqft.astype(float)).replace(np.inf, 0).fillna(0)

@orca.column('parcels', 'is_park', cache=True, cache_scope='iteration')
def is_park(parcels):
    return (parcels.land_use_type_id == 19)

@orca.column('parcels', 'lnemp20da', cache=True, cache_scope='iteration')
def lnemp20da(parcels, zones):
    return np.log1p(misc.reindex(zones.jobs_within_20_min_tt_hbw_am_drive_alone, parcels.zone_id))

@orca.column('parcels', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(parcels, households):
    return households.persons.groupby(households.parcel_id).size().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(parcels, jobs):
    return jobs.parcel_id.groupby(jobs.parcel_id).size().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'number_of_jobs_wwd', cache=True, cache_scope='iteration')
def number_of_jobs_wwd(parcels, gridcells, settings):
    from abstract_variables import abstract_within_walking_distance_parcels
    return abstract_within_walking_distance_parcels("number_of_jobs", parcels, gridcells, settings)

@orca.column('parcels', 'parcel_sqft_wwd', cache=True, cache_scope='iteration')
def parcel_sqft_wwd(parcels, gridcells, settings):
    from abstract_variables import abstract_within_walking_distance_parcels
    return abstract_within_walking_distance_parcels("parcel_sqft", parcels, gridcells, settings)

@orca.column('parcels', 'park_area', cache=True, cache_scope='iteration')
def park_area(parcels):
    return ((parcels.land_use_type_id == 19) * parcels.parcel_sqft)

@orca.column('parcels', 'residential_units', cache=True, cache_scope='iteration')
def residential_units(parcels, buildings):
    return buildings.residential_units.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'total_improvement_value', cache=True, cache_scope='iteration')
def total_improvement_value(parcels, buildings):
    return buildings.improvement_value.groupby(buildings.parcel_id).sum().\
           reindex(parcels.index).fillna(0)

@orca.column('parcels', 'total_land_value_per_sqft', cache=True, cache_scope='iteration')
def total_land_value_per_sqft(parcels):
    return ((parcels.land_value + parcels.total_improvement_value)/parcels.parcel_sqft).replace(np.inf, 0).fillna(0)

@orca.column('parcels', 'unit_name', cache=True)
def unit_name(parcels, land_use_types):
    return misc.reindex(land_use_types.unit_name, parcels.land_use_type_id)

@orca.column('parcels', 'unit_price', cache=True, cache_scope='iteration')
def unit_price(parcels):
    return ((parcels.land_value + parcels.total_improvement_value)/parcels.existing_units).replace(np.inf, 0).fillna(0)
