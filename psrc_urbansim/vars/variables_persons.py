import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# PERSONS VARIABLES
#####################

@orca.column('persons', 'parcel_id', cache=True)
def parcel_id(persons, households):
    return misc.reindex(households.parcel_id, persons.household_id)

@orca.column('persons', 'zone_id', cache=True)
def zone_id(persons, households):
    return misc.reindex(households.zone_id, persons.household_id)

@orca.column('persons', 'faz_id', cache=True)
def faz_id(persons, zones):
    return misc.reindex(zones.faz_id, persons.zone_id)

@orca.column('persons', 'tractcity_id', cache=True)
def tractcity_id(persons, households):
    return misc.reindex(households.tractcity_id, persons.household_id)

