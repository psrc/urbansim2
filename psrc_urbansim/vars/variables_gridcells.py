import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_parcels.utils

#####################
# GRIDCELLS VARIABLES
#####################

@orca.column('gridcells', 'is_near_arterial', cache=True, cache_scope='iteration')
def is_near_arterial(gridcells, settings):
    return (gridcells.distance_to_arterial <= settings.get("near_arterial_threshold", 300)).astype("int16")

@orca.column('gridcells', 'is_near_highway', cache=True, cache_scope='iteration')
def is_near_highway(gridcells, settings):
    return (gridcells.distance_to_highway <= settings.get("near_highway_threshold", 300)).astype("int16")


