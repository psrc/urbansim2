import pandas as pd
import pytest
import orca

def setup_function(func):
    orca.clear_all()

def teardown_function(func):
    orca.clear_all()
    
@pytest.fixture
def expected_pcl():
    return pd.DataFrame(
        {'residential_units' : [8, 9, 11]})

@pytest.fixture
def input_pcl():
    return pd.DataFrame(
            {'parcel_id': [1,2,3]})

@pytest.fixture
def input_bld():
    return pd.DataFrame(
            {'building_id':       [1,2, 3, 4,5,6],
             'residential_units': [3,5,10,0,1,9],
             'parcel_id':         [0,0, 2,2,2,1]})   

@pytest.fixture
def inputs(input_pcl, input_bld):
    orca.add_table('parcels', input_pcl)
    orca.add_table('buildings', input_bld)
    #orca.broadcast('parcels', 'buildings', cast_index=True, onto_on='parcel_id')
    import psrc_urbansim.vars.variables_parcels

def test_residential_units(inputs, expected_pcl):
    pcl = orca.get_table('parcels')
    #pytest.set_trace()
    assert (pcl['residential_units'].values == expected_pcl['residential_units'].values).all()
    
    
#pytest.main("-k test_residential_units")
