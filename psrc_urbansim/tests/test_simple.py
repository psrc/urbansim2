import pandas as pd
import pytest

@pytest.fixture
def dummy():
    return 0

def test_dummy(dummy):
    assert dummy == 0