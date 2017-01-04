import pandas as pd
import pytest
import orca

def setup_function(func):
    orca.clear_all()

def teardown_function(func):
    orca.clear_all()
    
@pytest.fixture
def input_pcl():
    df = pd.DataFrame(
        {
            "parcel_id":  [1,   2,    3,   4,   5],
            "x_coord_sp": [1000,   3000,    3000, 3000, 4000],
            "y_coord_sp": [2000,   1000,    2000, 2000, 2000],
            "grid_id":    [5,   3,    7,   7,   8]
        })
    return df.set_index("parcel_id")

@pytest.fixture
def input_gcl():
    df = pd.DataFrame(
        {
            "grid_id":    [1,  2,  3,  4,  5,  6,  7,  8],
            "relative_x": [1,  2,  3,  4,  1,  2,  3,  4],
            "relative_y": [1,  1,  1,  1,  2,  2,  2,  2]           
        })
    return df.set_index("grid_id")


@pytest.fixture
def input_bld():
    df = pd.DataFrame(
        {
            "building_id":  [1,2,3,4,5,6],
            "parcel_id":    [1,1,2,1,4,5]
        })
    return df.set_index("building_id")

@pytest.fixture
def input_jobs():
    df = pd.DataFrame(
            {
                "job_id":     [1,2,3,4,5,6],
                "building_id":[1,2,3,4,5,6]
            })
    return df.set_index("job_id")

@pytest.fixture
def input_settings():
    return {"cell_size": 150, 
            "cell_walking_radius": 150,
            "wwd_correlate_mode": "constant"}

@pytest.fixture
def expected_pcl():
    return pd.DataFrame(
        {'number_of_jobs_within_radius' : [4, 3, 6, 6, 3],
         'number_of_jobs_wwd' : [3, 2, 3, 3, 2]})
    
@pytest.fixture
def inputs(input_pcl, input_bld, input_jobs, input_gcl, input_settings):
    orca.add_table('parcels', input_pcl)
    orca.add_table('buildings', input_bld)
    orca.add_table('jobs', input_jobs)
    orca.add_table('gridcells', input_gcl)
    orca.add_injectable('settings', input_settings)
    import psrc_urbansim.vars.variables_jobs
    import psrc_urbansim.vars.variables_parcels

def test_number_of_jobs_within_radius(inputs, expected_pcl):
    import psrc_urbansim.vars.abstract_variables as av
    pcl = orca.get_table('parcels')
    #pytest.set_trace()
    assert (av.abstract_within_given_radius(2000, pcl.number_of_jobs, pcl.x_coord_sp, pcl.y_coord_sp) == expected_pcl['number_of_jobs_within_radius']).all()
    
def test_number_of_jobs_within_walking_distance(inputs, expected_pcl):
    pcl = orca.get_table('parcels')
    #pytest.set_trace()
    assert (pcl.number_of_jobs_wwd == expected_pcl['number_of_jobs_wwd']).all()
    
#pytest.main(['test_parcel5_variables.py'])
#pytest.main(['-k test_number_of_jobs_within_walking'])