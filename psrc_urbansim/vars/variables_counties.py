import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# COUNTIES VARIABLES (in alphabetic order)
#####################


@orca.column('counties', 'res_4_VR', cache=True, cache_scope='iteration')
def res_4_VR(counties, buildings):
    return ((buildings.building_type_id == 4) * (buildings.target_vacancy_rate)).\
	        groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)