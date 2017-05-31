import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_parcels.utils

#####################
# SCHOOLS VARIABLES (in alphabetic order)
#####################

@orca.column('schools', 'faz_id', cache=True)
def faz_id(schools, parcels):
    return misc.reindex(parcels.faz_id, schools.parcel_id)