import orca
from urbansim.utils import misc

#####################
# CONTROL TOTALS VARIABLES
#####################

@orca.column('household_controls', 'county_id', cache=True)
def county_id(household_controls, subregs):
    return misc.reindex(subregs.county_id, household_controls.subreg_id)

@orca.column('employment_controls', 'county_id', cache=True)
def county_id(employment_controls, subregs):
    return misc.reindex(subregs.county_id, employment_controls.subreg_id)