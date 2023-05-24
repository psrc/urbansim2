import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils
from psrc_urbansim.vars.variables_interactions import avg_network_distance_from_home_to_work
from psrc_urbansim.vars.variables_interactions import max_network_distance_from_home_to_work

#####################
# HOUSEHOLDS VARIABLES (in alphabetic order)
#####################

@orca.column('households', 'building_type_id', cache=True, cache_scope='step')
def building_type_id(households, buildings):
    return misc.reindex(buildings.building_type_id, households.building_id)

#@orca.column('households_lag1', 'building_type_id', cache=True)
#def building_type_id(households_lag1, buildings_lag1):
#    return misc.reindex(buildings_lag1.building_type_id, households_lag1.building_id)

@orca.column('households', 'city_id', cache=True, cache_scope='step')
def city_id(households, parcels):
    if "city_id" in households.local_columns:
        # this hack is needed for allocation mode, since orca 
        # gives priority to computed columns instead of local columns
        return households.local.city_id
    return misc.reindex(parcels.city_id, households.parcel_id)

@orca.column('households', 'county_id', cache=True, cache_scope='step')
def county_id(households, parcels):
    if "county_id" in households.local_columns:    
        return households.local.county_id 
    return misc.reindex(parcels.county_id, households.parcel_id)

@orca.column('households', 'subreg_id', cache=True, cache_scope='step')
def subreg_id(households, parcels):
    if "subreg_id" in households.local_columns:
        # this hack is needed for allocation mode, since orca 
        # gives priority to computed columns instead of local columns
        return households.local.subreg_id
    return misc.reindex(parcels.subreg_id, households.parcel_id)

@orca.column('households', 'faz_id', cache=True, cache_scope='step')
def faz_id(households, zones):
    return misc.reindex(zones.faz_id, households.zone_id)

@orca.column('households', 'grid_id', cache=True, cache_scope='step')
def grid_id(households, parcels):
    return misc.reindex(parcels.grid_id, households.parcel_id)

@orca.column('households', 'growth_center_id', cache=True)
def growth_center_id(households, parcels, parcels_geos):
    if "growth_center_id" in parcels.columns:
        return misc.reindex(parcels.growth_center_id, households.parcel_id)	
    return misc.reindex(parcels_geos.growth_center_id, households.parcel_id)	

@orca.column('households', 'target_id', cache=True, cache_scope='step')
def target_id(households, parcels):
    return misc.reindex(parcels.target_id, households.parcel_id)

@orca.column('households', 'control_id', cache=True, cache_scope='step')
def control_id(households, parcels):
    return misc.reindex(parcels.control_id, households.parcel_id)

@orca.column('households', 'control_hct_id', cache=True, cache_scope='step')
def control_id(households, parcels):
    return misc.reindex(parcels.control_hct_id, households.parcel_id)


@orca.column('households', 'income_category', cache=True, cache_scope='step')
def income_category(households, settings):
    income_breaks = settings.get('income_breaks', [34000, 64000, 102000])
    res = households.income.values
    res[:] = 0
    res[(households.income < income_breaks[0]).values] = 1
    res[(np.logical_and(households.income >= income_breaks[0], households.income < income_breaks[1])).values] = 2
    res[(np.logical_and(households.income >= income_breaks[1], households.income < income_breaks[2])).values] = 3
    res[(households.income >= income_breaks[2]).values] = 4
    return res

#@orca.column('households', 'is_inmigrant', cache=True)
#def is_inmigrant(households, parcels):
#    return (households.building_id < 0).reindex(households.index)

@orca.column('households', 'is_residence_mf', cache=True, cache_scope='step')
def is_residence_mf(households, buildings):
    return misc.reindex(buildings.multifamily_generic_type, households.building_id).fillna(-1)

@orca.column('households', 'parcel_id', cache=True, cache_scope='step')
def parcel_id(households, buildings):
    return misc.reindex(buildings.parcel_id, households.building_id)

@orca.column('households', 'residence_large_area', cache=True, cache_scope='step')
def residence_large_area(households, buildings):
    return misc.reindex(buildings.large_area_id, households.building_id).fillna(-1)


#@orca.column('households', 'same_building_type', cache=True)
#def same_building_type(households, households_lag1):
#    merged = households.building_type_id.to_frame('bt').join(households_lag1.building_type_id.to_frame('btlag'))
#    return merged.bt == merged.btlag

@orca.column('households', 'tractcity_id', cache=True, cache_scope='step')
def tractcity_id(households, parcels):
    return misc.reindex(parcels.tractcity_id, households.parcel_id)

@orca.column('households', 'worker1_zone_id', cache=True, cache_scope='step')
def worker1_zone_id(households, persons):
    return ((persons.worker1)*(persons.workplace_zone_id)).\
        groupby(persons.household_id).sum().\
        reindex(households.index).fillna(-1).replace(0, -1)

@orca.column('households', 'worker2_zone_id', cache=True, cache_scope='step')
def worker2_zone_id(households, persons):
    return((persons.worker2)*(persons.workplace_zone_id)).\
        groupby(persons.household_id).sum().\
        reindex(households.index).fillna(-1).replace(0, -1)

@orca.column('households', 'work_zone_id', cache=True, cache_scope='step')
def work_zone_id(households, buildings):
    return misc.reindex(buildings.zone_id, households.building_id).fillna(-1)

@orca.column('households', 'zone_id', cache=True, cache_scope='step')
def zone_id(households, buildings):
    return misc.reindex(buildings.zone_id, households.building_id).fillna(-1)

@orca.column('households', 'prev_residence_is_mf', cache=True, cache_scope='step')
def prev_residence_is_mf(households, buildings_lag1):
    return misc.reindex(buildings_lag1.multifamily_generic_type, households.previous_building_id).fillna(-1)

@orca.column('households', 'prev_residence_large_area_id', cache=True, cache_scope='step')
def prev_residence_large_area_id(households, buildings_lag1):
    return misc.reindex(buildings_lag1.large_area_id, households.previous_building_id).fillna(-1)

@orca.column('households', 'persons_under_13', cache=True, cache_scope='step')
def persons_under_13(households, persons):
    df = persons.local[persons.local.age < 13]
    return df.groupby(df.household_id).age.count().reindex(households.index).fillna(0)

#@orca.column('households', 'max_distance_to_work', cache=True)
#def max_distance_to_work(households):
#    return pd.Series(max_network_distance_from_home_to_work(households.worker1_zone_id, households.worker1_zone_id, households.zone_id))

@orca.column('households', 'prev_zone_id', cache=True, cache_scope='step')
def prev_zone_id(households, buildings_lag1):
    return misc.reindex(buildings_lag1.zone_id, households.previous_building_id).fillna(-1)


#@orca.column('households', 'prev_max_distance_to_work', cache=True)
#def prev_max_distance_to_work(households):
#    return pd.Series(max_network_distance_from_home_to_work(households.worker1_zone_id, households.worker1_zone_id, households.prev_zone_id))

                                     






