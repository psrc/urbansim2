import os
import pandas as pd
import orca
from urbansim.utils import yamlio
import data

# Indicators script
# ==================

# Define injectables

# replace this by passing yaml file name as argument
@orca.injectable()
def settings_file():
    return "indicators_settings.yaml"

# Read yaml config
@orca.injectable()
def settings(settings_file):
    return yamlio.yaml_to_dict(str_or_buffer=settings_file)

@orca.step()
def compute_indicators(settings):
    # loop over indicators from settings and store
    for ind, value in settings['indicators'].iteritems():
        ds = value['dataset']
        for d in ds:
            print orca.get_table(d)[ind]
             

# Compute indicators
orca.run(['compute_indicators'], iter_vars=settings(settings_file())['years'])

