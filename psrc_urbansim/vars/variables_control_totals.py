import orca
from urbansim.utils import misc
import numpy as np

#####################
# CONTROL TOTALS VARIABLES
#####################

@orca.column('household_controls', 'county_id', cache=True)
def county_id(household_controls, subregs):
    # add subreg_id column if does not exist.
    # needed only for alocation mode, but not 
    # availble in simulation input dataset, so
    # need to add it here. 

    if not 'subreg_id' in household_controls.columns:
        arr = np.full(household_controls.__len__(), -1, dtype="int32")
        orca.add_column("household_controls", "subreg_id", arr)
    return misc.reindex(subregs.county_id, household_controls.subreg_id)

@orca.column('employment_controls', 'county_id', cache=True)
def county_id(employment_controls, subregs):
    # add subreg_id column if does not exist.
    # needed only for alocation mode, but not 
    # availble in simulation input dataset, so
    # need to add it here. 

    if not 'subreg_id' in employment_controls.columns:
        arr = np.full(employment_controls.__len__(), -1, dtype="int32")
        orca.add_column("employment_controls", "subreg_id", arr)
    return misc.reindex(subregs.county_id, employment_controls.subreg_id)