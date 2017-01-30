import pandas as pd
import pytest
import orca
    
@pytest.fixture(scope='function')
def expected_pcl():
    return pd.DataFrame(
        {'residential_units' : [8, 9, 11]})

@pytest.fixture(scope='function')
def input_pcl():
    return pd.DataFrame(
            {'parcel_id': [1,2,3]})

@pytest.fixture(scope='function')
def input_bld():
    return pd.DataFrame(
            {'building_id':       [1,2, 3, 4,5,6],
             'residential_units': [3,5,10,0,1,9],
             'parcel_id':         [0,0, 2,2,2,1]})   

@pytest.fixture(scope='function')
def inputs(input_pcl, input_bld):
    orca.clear_all()
    orca.add_table('parcels', input_pcl)
    orca.add_table('buildings', input_bld)
    import psrc_urbansim.vars.variables_parcels

def xtest_residential_units(inputs, expected_pcl):
    pcl = orca.get_table('parcels')
    #pytest.set_trace()
    assert (pcl['residential_units'].values == expected_pcl['residential_units'].values).all()
    
    
#pytest.main("-k test_residential_units")
