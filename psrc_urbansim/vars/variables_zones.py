import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils


#####################
# ZONES VARIABLES (in alphabetic order)
#####################

@orca.column('zones', 'avg_income')
def ave_income(zones, households):
    s = households.income.groupby(households.zone_id).quantile().apply(np.log1p)
    return s.reindex(zones.index).fillna(s.quantile())

@orca.column('zones', 'jobs_within_20_min_tt_hbw_am_drive_alone')
def jobs_within_20_min_tt_hbw_am_drive_alone(zones, travel_data):
    from abstract_variables import abstract_access_within_threshold_variable_from_origin
    return abstract_access_within_threshold_variable_from_origin(travel_data.am_single_vehicle_to_work_travel_time, zones.number_of_jobs, 20)

@orca.column('zones', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(zones, households):
    return households.persons.groupby(households.zone_id).size().\
           reindex(zones.index).fillna(0)

@orca.column('zones', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(zones, jobs):
    return jobs.zone_id.groupby(jobs.zone_id).size().\
           reindex(zones.index).fillna(0)

def trip_weighted_average_logsum_hbw_am_income_category(zones, travel_data, income_category):
    from abstract_variables import abstract_trip_weighted_average_from_home
    return abstract_trip_weighted_average_from_home(travel_data["logsum_hbw_am_income_%s" % income_category], 
                                                    travel_data["am_pk_period_drive_alone_vehicle_trips"],
                                                    travel_data.index.get_level_values('from_zone_id'), zones)

@orca.column('zones', 'trip_weighted_average_logsum_hbw_am_income_1', cache=True, cache_scope='iteration')
def trip_weighted_average_logsum_hbw_am_income_1(zones, travel_data):
    return trip_weighted_average_logsum_hbw_am_income_category(zones, travel_data, 1)

@orca.column('zones', 'trip_weighted_average_logsum_hbw_am_income_2', cache=True, cache_scope='iteration')
def trip_weighted_average_logsum_hbw_am_income_2(zones, travel_data):
    return trip_weighted_average_logsum_hbw_am_income_category(zones, travel_data, 2)

@orca.column('zones', 'trip_weighted_average_logsum_hbw_am_income_3', cache=True, cache_scope='iteration')
def trip_weighted_average_logsum_hbw_am_income_3(zones, travel_data):
    return trip_weighted_average_logsum_hbw_am_income_category(zones, travel_data, 3)

@orca.column('zones', 'trip_weighted_average_logsum_hbw_am_income_4', cache=True, cache_scope='iteration')
def trip_weighted_average_logsum_hbw_am_income_4(zones, travel_data):
    return trip_weighted_average_logsum_hbw_am_income_category(zones, travel_data, 4)