import pandas as pd
import numpy as np
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

    