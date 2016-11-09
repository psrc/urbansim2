import orca
import random
import urbansim_defaults.utils as utils
import psrc_urbansim.utils as psrcutils
#import urbansim_defaults.models
import datasources
import variables
import numpy as np
import pandas as pd
from psrc_urbansim.mod.allocation import AgentAllocationModel

# Residential REPM
@orca.step('repmres_estimate')
def repmres_estimate(parcels):
    return utils.hedonic_estimate("repmres.yaml", parcels, None)

@orca.step('repmres_simulate')
def repmres_simulate(parcels):
    return utils.hedonic_simulate("repmres.yaml", parcels, None, "land_value", cast=True)

# Non-Residential REPM
@orca.step('repmnr_estimate')
def repmnr_estimate(parcels):
    return utils.hedonic_estimate("repmnr.yaml", parcels, None)

@orca.step('repmnr_simulate')
def repmnr_simulate(parcels):
    return utils.hedonic_simulate("repmnr.yaml", parcels, None, "land_value", cast=True)


# HLCM
@orca.step('hlcm_estimate')
def hlcm_estimate(households, buildings, parcels, zones):
    return utils.lcm_estimate("hlcm.yaml", households, "building_id",
                              buildings, [parcels, zones])

@orca.step('hlcm_simulate')
def hlcm_simulate(households, buildings, parcels, zones):
    return utils.lcm_simulate("hlcm.yaml", households, buildings, [parcels, zones],
                              "building_id", "residential_units", "vacant_residential_units", cast=True)

# ELCM
@orca.step('elcm_estimate')
def elcm_estimate(jobs, buildings, parcels, zones):
    return utils.lcm_estimate("elcm.yaml", jobs, "building_id",
                              buildings, [parcels, zones])


@orca.step('elcm_simulate')
def elcm_simulate(jobs, buildings, parcels, zones):
    return utils.lcm_simulate("elcm.yaml", jobs, buildings, [parcels, zones],
                              "building_id", "job_spaces", "vacant_job_spaces", cast=True)


@orca.step('households_relocation')
def households_relocation(households, household_relocation_rates):
    return utils.simple_relocation(households, .05, "building_id", cast=True)
    #from urbansim.models import relocation as relo
    #rm = relo.RelocationModel(household_relocation_rates.to_frame())
    #movers = rm.find_movers(households.to_frame())
    #print "%s households selected to move." % movers.size
    #households.update_col_from_series("building_id",
                            #pd.Series(-1, index=movers), cast=True)    

@orca.step('jobs_relocation')
def jobs_relocation(jobs, job_relocation_rates):
    from urbansim.models import relocation as relo
    rm = relo.RelocationModel(job_relocation_rates.to_frame(), rate_column='job_relocation_probability')
    movers = rm.find_movers(jobs.to_frame())
    print "%s jobs selected to move." % movers.size
    jobs.update_col_from_series("building_id",
                            pd.Series(-1, index=movers), cast=True) 


@orca.step('households_transition')
def households_transition(households, household_controls, year, settings, persons):
    orig_size_hh = households.local.shape[0]
    orig_size_pers = persons.local.shape[0]
    res = utils.full_transition(households, household_controls, year, 
                                 settings['households_transition'], "building_id", linked_tables={"persons": (persons.local, 'household_id')})
    print "Net change: %s households" % (orca.get_table("households").local.shape[0] - orig_size_hh)
    print "Net change: %s persons" % (orca.get_table("persons").local.shape[0] - orig_size_pers)
    return res


@orca.step('jobs_transition')
def jobs_transition(jobs, employment_controls, year, settings):
    orig_size = jobs.local.shape[0]
    res = utils.full_transition(jobs,
                                 employment_controls,
                                 year,
                                 settings['jobs_transition'],
                                 "building_id")
    print "Net change: %s jobs" % (orca.get_table("jobs").local.shape[0] - orig_size)
    return res    


@orca.step('governmental_jobs_scaling')
def governmental_jobs_scaling(jobs, buildings, year):
    orca.add_column('buildings', 'existing', np.zeros(len(buildings), dtype="int32"))
    alloc = AgentAllocationModel('existing', 'number_of_governmental_jobs', as_delta=False)
    jobs_to_place = jobs.local[np.logical_and(np.in1d(jobs.sector_id, [18, 19]), jobs.building_id < 0)]
    print "Locating %s governmental jobs" % len(jobs_to_place)
    loc_ids, loc_allo = alloc.locate_agents(orca.get_table("buildings").to_frame(buildings.local_columns + ['number_of_governmental_jobs', 'existing']), 
                                            jobs_to_place, year=year)
    jobs.local.loc[loc_ids.index, buildings.index.name] = loc_ids
    print "Number of unplaced governmental jobs: %s" % np.logical_or(np.isnan(loc_ids), loc_ids < 0).sum()
    orca.add_table(jobs.name, jobs.local)





