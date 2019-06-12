import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# COUNTIES VARIABLES (in alphabetic order)
#####################


@orca.column('counties', 'nonres_3_all', cache=True, cache_scope='iteration')
def nonres_3_all(counties, buildings):
    return ((buildings.job_spaces) * (buildings.building_type_id == 3)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_3_vac', cache=True, cache_scope='iteration')
def nonres_3_vac(counties, buildings):
    return ((buildings.vacant_job_spaces) * (buildings.building_type_id == 3)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_3_VR', cache=True, cache_scope='iteration')
def nonres_3_VR(counties):
    return counties.nonres_3_vac / counties.nonres_3_all


@orca.column('counties', 'nonres_8_all', cache=True, cache_scope='iteration')
def nonres_8_all(counties, buildings):
    return ((buildings.job_spaces) * (buildings.building_type_id == 8)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_8_vac', cache=True, cache_scope='iteration')
def nonres_8_vac(counties, buildings):
    return ((buildings.vacant_job_spaces) * (buildings.building_type_id == 8)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_8_VR', cache=True, cache_scope='iteration')
def nonres_8_VR(counties):
    return counties.nonres_8_vac / counties.nonres_8_all


@orca.column('counties', 'nonres_13_all', cache=True, cache_scope='iteration')
def nonres_13_all(counties, buildings):
    return ((buildings.job_spaces) * (buildings.building_type_id == 13)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_13_vac', cache=True, cache_scope='iteration')
def nonres_13_vac(counties, buildings):
    return ((buildings.vacant_job_spaces) * (buildings.building_type_id == 13)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_13_VR', cache=True, cache_scope='iteration')
def nonres_13_VR(counties):
    return counties.nonres_13_vac / counties.nonres_13_all


@orca.column('counties', 'nonres_20_all', cache=True, cache_scope='iteration')
def nonres_20_all(counties, buildings):
    return ((buildings.job_spaces) * (buildings.building_type_id == 20)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_20_vac', cache=True, cache_scope='iteration')
def nonres_20_vac(counties, buildings):
    return ((buildings.vacant_job_spaces) * (buildings.building_type_id == 20)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_20_VR', cache=True, cache_scope='iteration')
def nonres_20_VR(counties):
    return counties.nonres_20_vac / counties.nonres_20_all


@orca.column('counties', 'nonres_21_all', cache=True, cache_scope='iteration')
def nonres_21_all(counties, buildings):
    return ((buildings.job_spaces) * (buildings.building_type_id == 21)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_21_vac', cache=True, cache_scope='iteration')
def nonres_21_vac(counties, buildings):
    return ((buildings.vacant_job_spaces) * (buildings.building_type_id == 21)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'nonres_21_VR', cache=True, cache_scope='iteration')
def nonres_21_VR(counties):
    return counties.nonres_21_vac / counties.nonres_21_all


@orca.column('counties', 'res_4_all', cache=True, cache_scope='iteration')
def res_4_all(counties, buildings):
    return ((buildings.residential_units) * (buildings.building_type_id == 4)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'res_4_vac', cache=True, cache_scope='iteration')
def res_4_vac(counties, buildings):
    return ((buildings.vacant_residential_units) * (buildings.building_type_id == 4)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'res_4_VR', cache=True, cache_scope='iteration')
def res_4_VR(counties, buildings):
    return counties.res_4_vac / counties.res_4_all


@orca.column('counties', 'res_12_all', cache=True, cache_scope='iteration')
def res_12_all(counties, buildings):
    return ((buildings.residential_units) * (buildings.building_type_id == 12)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'res_12_vac', cache=True, cache_scope='iteration')
def res_12_vac(counties, buildings):
    return ((buildings.vacant_residential_units) * (buildings.building_type_id == 12)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'res_12_VR', cache=True, cache_scope='iteration')
def res_12_VR(counties, buildings):
    return counties.res_12_vac / counties.res_12_all


@orca.column('counties', 'res_19_all', cache=True, cache_scope='iteration')
def res_19_all(counties, buildings):
    return ((buildings.residential_units) * (buildings.building_type_id == 19)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'res_19_vac', cache=True, cache_scope='iteration')
def res_19_vac(counties, buildings):
    return ((buildings.vacant_residential_units) * (buildings.building_type_id == 19)).\
	groupby(buildings.county_id).sum().reindex(counties.index).fillna(0)

@orca.column('counties', 'res_19_VR', cache=True, cache_scope='iteration')
def res_19_VR(counties, buildings):
    return counties.res_19_vac / counties.res_19_all