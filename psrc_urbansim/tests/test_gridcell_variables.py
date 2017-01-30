import pandas as pd
import pytest
import orca
    
@pytest.fixture
def input_gcl():
    df = pd.DataFrame(
        {
            'grid_id': [1,2,3,4],
            'relative_x': [1,2,1,2],
            'relative_y': [1,1,2,2],
            'hhs': [100, 500, 1000, 1500],     
        })
    return df.set_index("grid_id")

@pytest.fixture
def expected_gcl():
    return pd.DataFrame(
        {'hhs_wwd' : [1800.0, 3100.0, 4600.0, 6000.0]})

@pytest.fixture(scope='function')
def inputs(input_gcl):
    orca.clear_all()
    orca.add_table('gridcells', input_gcl)
    

def test_hhs_wwd(inputs, expected_gcl):
    import psrc_urbansim.vars.abstract_variables as av
    gcl = orca.get_table('gridcells')
    #pytest.set_trace()
    assert (av.abstract_within_walking_distance_gridcells(gcl.hhs, gcl, cell_size=150, walking_distance_circle_radius=150).values == expected_gcl['hhs_wwd'].values).all()
    
#pytest.main(["test_gridcell_variables.py"])
