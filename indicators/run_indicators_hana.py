import os
import pandas as pd
import orca
from urbansim.utils import yamlio
import data

# Indicators script
# ==================

# Define injectables

@orca.injectable()
def settings_file():
    return "indicators_settings.yaml"

# Read yaml config
@orca.injectable()
def settings(settings_file):
    return yamlio.yaml_to_dict(str_or_buffer=settings_file)

@orca.step()
def compute_indicators():
    zones = orca.get_table('zones')
    print zones['number_of_households']
    print orca.get_table('fazes')['number_of_households']


# Compute indicators
orca.run(['compute_indicators'], iter_vars=settings(settings_file())['years'])


# Export
