import pandas as pd
import numpy as np
import scipy.ndimage as ndi
import orca
from urbansim.utils import misc
import urbansim_defaults.utils


def travel_data_2d(travel_data_column, fill=0):
    """Create a matrix of values from travel_data_column index by (from_zone_id, to_zone_id)
    """
    mat = travel_data_column.unstack()
    mat[np.isnan(mat)] = fill
    return mat

def abstract_access_within_threshold_variable_from_origin(travel_data_attribute, zone_attribute, threshold):
    """Accessibility variable within a threshold given by a zone_attribute (e.g. number_of_jobs), 
    measured using the travel_data_attribute (e.g. am_single_vehicle_to_work_travel_time).
    """
    mat = travel_data_2d(travel_data_attribute, fill=threshold+1)
    zone_ids = zone_attribute.index
    vv = (mat <= threshold)[zone_ids] * zone_attribute
    result = vv.sum(axis=1)[zone_ids]
    result[np.isnan(result)] = 0
    return result

def abstract_logsum_interaction_variable(travel_data_attribute_dict, agent_categories, agent_zone_id, location_zone_id, 
                                              direction_from_home = True):
    if direction_from_home:
        home_zone = agent_zone_id
        work_zone = location_zone_id
    else:
        home_zone = location_zone_id
        work_zone = agent_zone_id

    idx = pd.MultiIndex.from_arrays([home_zone.values, work_zone.values], names=["from_zone_id", "to_zone_id"])    
    max_choices = agent_categories.values.max() + 1
    tmlist = [np.nan] * max_choices
    for i in range(max_choices): # iterate ofve income categories
        if i in travel_data_attribute_dict:
            tmlist[i] = travel_data_attribute_dict[i][idx].reset_index(drop=True)
    return agent_categories.values.choose(tmlist)

def abstract_weighted_access(travel_data_attribute, zone_attribute, 
                             aggregate_by_origin = True, function="sum"):
    """
    Summarizes zone attribute weighted by a travel_data attribute, e.g. 
    generalized_cost_weighted_access_to_employment_hbw_am_drive_alone = 
    sum of number of jobs in zone j divided by generalized cost from zone i to j.
    The weight is the home-based-work am generalized cost by auto drive-alone.
    """    
    if aggregate_by_origin:
        attribute_zone_id_name = 'from_zone_id'
        summary_zone_id_name = 'to_zone_id'
    else:
        attribute_zone_id_name = 'to_zone_id'
        summary_zone_id_name = 'from_zone_id'
    attribute = zone_attribute[travel_data_attribute.index.get_level_values(attribute_zone_id_name)]
    weighted_attribute = attribute.values * np.power(travel_data_attribute, -2).values
    f = getattr(ndi, function)
    results = np.array(f(weighted_attribute, labels = travel_data_attribute.index.get_level_values(summary_zone_id_name), 
                                 index=zone_attribute.index))
    return results

def abstract_travel_time_variable_to_destination(travel_data_attribute, destination):
    return travel_data_attribute.xs(destination, level='to_zone_id')

def abstract_travel_time_interaction_variable(travel_data_attribute, agent_zone_id, location_zone_id, 
                                              direction_from_home = True):
    if direction_from_home:
        home_zone = agent_zone_id
        work_zone = location_zone_id
    else:
        home_zone = location_zone_id
        work_zone = agent_zone_id
    idx = pd.MultiIndex.from_arrays([home_zone.values, work_zone.values], names=["from_zone_id", "to_zone_id"])
    return travel_data_attribute[idx].reset_index(drop=True)
        
def abstract_iv_residual(dependent_var, iv, filter):
    """Abstract variable for constructing an instrumental variable, such as price residuals."""
    ifilter = np.where(filter)
    y = dependent_var.iloc[ifilter]
    z = iv.iloc[ifilter]
    zdf = pd.concat([pd.Series(1,index=z.index), z], axis=1)
    zt = np.transpose(zdf.values)
    est = np.dot( np.dot( np.linalg.inv(np.dot(zt, zdf.values)), zt), y )
    r =  y - np.dot( zdf, est )
    results = pd.Series(0, index=dependent_var.index)
    results.iloc[ifilter] = r
    return results
    
def abstract_within_given_radius(radius, quantity, x, y, filter=None):
    from scipy.spatial import cKDTree
    if filter is not None:
        index = np.where(filter > 0)[0]
    else:
        index = np.arange(quantity.size)
    arr = quantity.iloc[index].values
    coords = np.column_stack((x.iloc[index], y.iloc[index]))
    kd_tree = cKDTree(coords, 100)
    KDTresults = kd_tree.query_ball_tree(kd_tree, radius)
    result = np.zeros(quantity.size, dtype=arr.dtype)
    # This solution can lead to an out-of-memory crash when executed on all parcels
    #akd = pd.DataFrame(KDTresults)
    #def get_quant_sum(x):
    #    return arr[x[~np.isnan(x)]].sum()
    #tmp = akd.apply(get_quant_sum, axis=1, raw=True)
    # This solution is very slow when executed on all parcels
    tmp = np.array([arr[l].sum() for l in KDTresults])
    result[index] = tmp
    return result    

def set_walking_distance_footprint(cell_size=150, walking_distance_circle_radius=600):
    wd_gc = int(2*walking_distance_circle_radius/float(cell_size)+1)
    center = int((wd_gc-1)/2)
    distance = np.ones((wd_gc,wd_gc), dtype="float32")
    distance[center,center]=0.0
    distance = ndi.distance_transform_edt(distance)
    return np.where(distance*cell_size <= walking_distance_circle_radius, 1, 0)

def get2Dattribute(attribute, dataset, x_name='relative_x', y_name='relative_y'):
    df = pd.DataFrame({"value": attribute, "x": dataset[x_name], "y": dataset[y_name]})
    df = df.set_index(["x", "y"])
    return (df.unstack(), df.index)
    
def abstract_within_walking_distance_gridcells(attribute, gridcells, filled_value=0, 
                        mode="reflect", x_name='relative_x', y_name='relative_y', **kwargs):
    wd_footprint = set_walking_distance_footprint(**kwargs)
    attr2d, index2d = get2Dattribute( attribute, gridcells, x_name=x_name, y_name=y_name)
    summed = ndi.correlate( np.ma.filled(attr2d, filled_value ), wd_footprint, mode=mode)
    attr2d[:] = summed
    res = pd.Series(attr2d.stack().reindex(index2d)["value"].values, index=gridcells.index)
    res[np.isnan(res)] = filled_value
    return res
    
    
def abstract_within_walking_distance_parcels(attribute_name, parcels, gridcells, settings, walking_radius=None, **kwargs):
    gcl_values = parcels[attribute_name].groupby(parcels.grid_id).sum().reindex(gridcells.index).fillna(0) 
    res = misc.reindex(abstract_within_walking_distance_gridcells(gcl_values, gridcells, 
                cell_size=settings.get('cell_size', 150), walking_distance_circle_radius=(walking_radius or settings.get('cell_walking_radius', 600)), 
                mode=settings.get("wwd_correlate_mode", "reflect"), **kwargs), 
                        parcels.grid_id)
    #TODO: this step should not be needed if all parcels have an exisitng gridcell assigned
    res[np.isnan(res)] = 0
    return res    

def abstract_trip_weighted_average_from_home(time_attribute, trips_attribute, from_zone_id, zones, missing_value=999):
    """Trip-weighted averaging for zone dataset."""
    non_missing_idx = np.where(np.logical_and(time_attribute != missing_value, trips_attribute != missing_value))
    numerator = np.array(ndi.sum(time_attribute.iloc[non_missing_idx] * trips_attribute.iloc[non_missing_idx],
                            labels = from_zone_id[non_missing_idx], index=zones.index))
    denominator = np.array(ndi.sum(trips_attribute.iloc[non_missing_idx],
                            labels = from_zone_id[non_missing_idx], index=zones.index), dtype="float32")
    # if there is a divide by zero then substitute the values from the next zone below 
    # if there are contiguous places of zero division the values should propagate upon iteration
    no_trips_from_here = np.where(denominator == 0)[0]
    if no_trips_from_here.size == denominator.size:
        print("%s attribute of travel_data is all zeros; trip_weighted_average_from_home returns all zeros" % trips_attribute.name)
        return np.zeros(numerator.size)
    while no_trips_from_here.size != 0:             
        substitute_locations = no_trips_from_here - 1    # a mapping, what zone the new data will come from
        if substitute_locations[0] < 0: substitute_locations[0] = 1
        numerator[no_trips_from_here] = numerator[substitute_locations]
        denominator[no_trips_from_here] = denominator[substitute_locations] 
        no_trips_from_here = np.where(denominator == 0)[0]
    return pd.Series(numerator/denominator, index=zones.index)