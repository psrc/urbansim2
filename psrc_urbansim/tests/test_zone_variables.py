import pandas as pd
import numpy as np
import pytest
import orca

def setup_function(func):
    orca.clear_all()

def teardown_function(func):
    orca.clear_all()


@pytest.fixture
def dummy():
    return 0

def test_dummy(dummy):
    assert dummy == 0
    
    
@pytest.fixture
def expected_tm_variables():
    return pd.DataFrame(
        {'jobs_within_20_min_tt_hbw_am_drive_alone' : np.array([0, 10, 0], dtype=np.int32)})

@pytest.fixture
def input_travel_data():
    df = pd.DataFrame(
            {"from_zone_id": np.array([3,3,1,1,1]),
             "to_zone_id": np.array([1,3,1,3,4]),
             "am_single_vehicle_to_work_travel_time": np.array([10.1, 20.2, 30.3, 40.4, 99])})
    return df.set_index(['from_zone_id', 'to_zone_id'])

@pytest.fixture
def input_zone():
    df = pd.DataFrame({"zone_id": np.array([1,3,4])})
    return df.set_index("zone_id")

@pytest.fixture
def input_job():
    df = pd.DataFrame({
               "job_id": np.arange(1,12),
               "zone_id": np.array(10*[1] + [3])})
    return df.set_index("job_id")

@pytest.fixture
def inputs(input_zone, input_travel_data, input_job):
    orca.add_table('zones', input_zone)
    orca.add_table('travel_data', input_travel_data)
    orca.add_table('jobs', input_job)
    import psrc_urbansim.vars.variables_zones

def test_jobs_within_20_min_tt(inputs, expected_tm_variables):
    zones = orca.get_table('zones')
    td = orca.get_table('travel_data')
    #pytest.set_trace()
    assert (zones.jobs_within_20_min_tt_hbw_am_drive_alone.values == expected_tm_variables['jobs_within_20_min_tt_hbw_am_drive_alone'].values).all()


# pytest.main('-k test_jobs')
