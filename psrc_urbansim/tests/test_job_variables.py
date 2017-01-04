import pandas as pd
import numpy as np
import pytest
import orca

def setup_function(func):
    orca.clear_all()

def teardown_function(func):
    orca.clear_all()

@pytest.fixture
def input_jobs():
    df = pd.DataFrame({
               "job_id": range(1,5),
               "sector_id": [1, 3, 2, 3]
    })
    return df.set_index("job_id")

@pytest.fixture
def input_sectors():
    df = pd.DataFrame({
        "sector_id": [1, 2],
        "name": ["Retail Trade", "Basic"]
    })
    return df.set_index("sector_id")

@pytest.fixture
def input_groups():
    df = pd.DataFrame({
        "group_id": [10, 11],
        "name": ["retail", "basic"]
    })
    return df.set_index("sector_id")

@pytest.fixture
def input_groups():
    df = pd.DataFrame({
        "group_id": [10, 11],
        "name": ["retail", "basic"]
    })
    return df.set_index("group_id")

@pytest.fixture
def input_group_def():
    df = pd.DataFrame({
        "sector_id": [1, 2, 3, 4],
        "group_id": [10, 11, 10, 11],
        "dummy": [0,0,0,0]
    })
    return df.set_index(["sector_id", "group_id"])

@pytest.fixture
def expected_sector_group_variables():
    return pd.DataFrame(
        {'is_in_sector_group_retail' : [True, True, False, True]})


@pytest.fixture
def inputs(input_jobs, input_sectors, input_groups, input_group_def):
    orca.add_table('jobs', input_jobs)
    orca.add_table('employment_sectors', input_sectors)
    orca.add_table('employment_sector_groups', input_groups)
    orca.add_table('employment_sector_group_definitions', input_group_def)    
    import psrc_urbansim.vars.variables_jobs
    
def test_retail_sector(inputs, expected_sector_group_variables):
    jobs = orca.get_table('jobs')
    #pytest.set_trace()
    assert (jobs.is_in_sector_group_retail.values == expected_sector_group_variables['is_in_sector_group_retail'].values).all()

pytest.main('-k test_retail_sector')
