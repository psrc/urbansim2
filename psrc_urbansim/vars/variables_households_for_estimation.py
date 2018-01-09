import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils
import psrc_urbansim.vars.variables_interactions 
#import avg_network_distance_from_home_to_work
#####################
# households_for_estimation VARIABLES (in alphabetic order)
#####################

@orca.column('households_for_estimation', 'building_type_id', cache=True)
def building_type_id(households_for_estimation, buildings):
    return misc.reindex(buildings.building_type_id, households_for_estimation.building_id)

#@orca.column('households_for_estimation_lag1', 'building_type_id', cache=True)
#def building_type_id(households_for_estimation_lag1, buildings_lag1):
#    return misc.reindex(buildings_lag1.building_type_id, households_for_estimation_lag1.building_id)

@orca.column('households_for_estimation', 'city_id', cache=True)
def city_id(households_for_estimation, parcels):
    return misc.reindex(parcels.city_id, households_for_estimation.parcel_id)

@orca.column('households_for_estimation', 'faz_id', cache=True)
def faz_id(households_for_estimation, zones):
    return misc.reindex(zones.faz_id, households_for_estimation.zone_id)

@orca.column('households_for_estimation', 'grid_id', cache=True)
def grid_id(households_for_estimation, parcels):
    return misc.reindex(parcels.grid_id, households_for_estimation.parcel_id)

@orca.column('households_for_estimation', 'income_category', cache=True)
def income_category(households_for_estimation, settings):
    income_breaks = settings.get('income_breaks', [34000, 64000, 102000])
    res = households_for_estimation.income.values
    res[:] = np.nan
    res[(households_for_estimation.income < income_breaks[0]).values] = 1
    res[(np.logical_and(households_for_estimation.income >= income_breaks[0], households_for_estimation.income < income_breaks[1])).values] = 2
    res[(np.logical_and(households_for_estimation.income >= income_breaks[1], households_for_estimation.income < income_breaks[2])).values] = 3
    res[(households_for_estimation.income >= income_breaks[2]).values] = 4
    return res

#@orca.column('households_for_estimation', 'is_inmigrant', cache=True)
#def is_inmigrant(households_for_estimation, parcels):
#    return return (households_for_estimation.building_id < 0).reindex(households_for_estimation.index)

@orca.column('households_for_estimation', 'is_residence_mf', cache=True)
def is_residence_mf(households_for_estimation, buildings):
    return misc.reindex(buildings.multifamily_generic_type, households_for_estimation.building_id).fillna(-1)

@orca.column('households_for_estimation', 'parcel_id', cache=True)
def parcel_id(households_for_estimation, buildings):
    return misc.reindex(buildings.parcel_id, households_for_estimation.building_id)

@orca.column('households_for_estimation', 'residence_large_area', cache=True)
def residence_large_area(households_for_estimation, buildings):
    return misc.reindex(buildings.large_area_id, households_for_estimation.building_id).fillna(-1)


#@orca.column('households_for_estimation', 'same_building_type', cache=True)
#def same_building_type(households_for_estimation, households_for_estimation_lag1):
#    merged = households_for_estimation.building_type_id.to_frame('bt').join(households_for_estimation_lag1.building_type_id.to_frame('btlag'))
#    return merged.bt == merged.btlag

@orca.column('households_for_estimation', 'tractcity_id', cache=True)
def tractcity_id(households_for_estimation, parcels):
    return misc.reindex(parcels.tractcity_id, households_for_estimation.parcel_id)

#@orca.column('households_for_estimation', 'worker1_zone_id', cache=True)
#def worker1_zone_id(households_for_estimation, persons_for_estimation):
#    return misc.reindex(persons_for_estimation.workplace_zone_id[persons_for_estimation.worker1==True], households_for_estimation.building_id).fillna(-1)

@orca.column('households_for_estimation', 'worker1_zone_id', cache=True)
def worker1_zone_id(households_for_estimation, persons_for_estimation):
    df = persons_for_estimation.to_frame()
    df = df[df.worker1==True]
    df = df[['workplace_zone_id', 'household_id']]
    df2 = pd.DataFrame(households_for_estimation.index)
    df2 = df2.merge(df, how='left', on ='household_id').fillna(-1)
    df2.set_index(df2.household_id, inplace = True)
    return df2.workplace_zone_id 
    
    #df = persons_for_estimation.to_frame()
    #df = df[df.worker1==True]
    #df = df[['workplace_zone_id', 'household_id']]
    #df.set_index(df.household_id, inplace = True)
    #df2 = pd.DataFrame(households_for_estimation.index)
    #return misc.reindex(df.workplace_zone_id, df2.household_id).fillna(-1)
    

#return misc.reindex(df.workplace_zone_id, pd.DataFrame(households_for_estimation.index).household_id).fillna(-1)
    #misc.reindex(df.workplace_zone_id, df2.household_id).fillna(-1)
    #return misc.reindex(df2.household_id, df2.).fillna(-1)
    #test = pd.merge(pd.DataFrame({"left": df2.household_id}),
    #              pd.DataFrame({"right": df.workplace_zone_id}),
    #              left_on="left",
    #              right_index=True,
    #              how="left")

@orca.column('households_for_estimation', 'worker2_zone_id', cache=True)
def worker2_zone_id(households_for_estimation, persons_for_estimation):
    df = persons_for_estimation.to_frame()
    df = df[df.worker2==True]
    df = df[['workplace_zone_id', 'household_id']]
    df2 = pd.DataFrame(households_for_estimation.index)
    df2 = df2.merge(df, how='left', on ='household_id').fillna(-1)
    df2.set_index(df2.household_id, inplace = True)
    return df2.workplace_zone_id 
    #return misc.reindex(persons_for_estimation.workplace_zone_id[persons_for_estimation.worker2==True], households_for_estimation.building_id).fillna(-1)

@orca.column('households_for_estimation', 'work_zone_id', cache=True)
def work_zone_id(households_for_estimation, buildings):
    return misc.reindex(buildings.zone_id, households_for_estimation.building_id).fillna(-1)

@orca.column('households_for_estimation', 'zone_id', cache=True)
def zone_id(households_for_estimation, buildings):
    return misc.reindex(buildings.zone_id, households_for_estimation.building_id).fillna(-1)

@orca.column('households_for_estimation', 'prev_residence_is_mf', cache=True)
def prev_residence_is_mf(households_for_estimation, buildings_lag1):
    return misc.reindex(buildings_lag1.multifamily_generic_type, households_for_estimation.previous_building_id).fillna(-1)
#@orca.column('households_for_estimation', 'previous_building_id', cache=True)
#def previous_building_id(households_for_estimation, buildings):
#    return misc.reindex(buildings.zone_id, households_for_estimation.building_id).fillna(-1)


#@orca.column('households_for_estimation', 'avg_net_dist_from_home_to_work', cache=True)
#def avg_net_dist_from_home_to_work(households_for_estimation, buildings):
#    return avg_network_distance_from_home_to_work(households_for_estimation.worker1_zone_id, households_for_estimation.worker2_zone_id, households_for_estimation.zone_id)
