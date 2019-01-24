import os
import sys
import orca
import random
import urbansim_defaults.utils as utils
import psrc_urbansim.utils as psrcutils
import datasources
import variables
import psrc_urbansim.vars.variables_persons as variables_persons
import psrc_urbansim.vars.variables_households as variables_households
import numpy as np
import pandas as pd
from psrc_urbansim.mod.allocation import AgentAllocationModel
import urbansim.developer as dev
import developer_models as psrcdev
import dcm_weighted_sampling as psrc_dcm
import sqftproforma
from urbansim.utils import misc, yamlio
import os
from psrc_urbansim.vars.variables_interactions import network_distance_from_home_to_work


# Residential REPM
@orca.step('repmres_estimate')
def repmres_estimate(parcels, zones, gridcells):
    return utils.hedonic_estimate("repmres.yaml", parcels, [zones, gridcells],
                                  out_cfg="repmrescoef.yaml")


@orca.step('repmres_simulate')
def repmres_simulate(parcels, zones, gridcells):
    return utils.hedonic_simulate("repmrescoef.yaml", parcels,
                                  [zones, gridcells], "land_value", cast=True)


# Non-Residential REPM
@orca.step('repmnr_estimate')
def repmnr_estimate(parcels, zones, gridcells):
    return utils.hedonic_estimate("repmnr.yaml", parcels,
                                  [zones, gridcells],
                                  out_cfg="repmnrcoef.yaml")


@orca.step('repmnr_simulate')
def repmnr_simulate(parcels, zones, gridcells):
    return utils.hedonic_simulate("repmnrcoef.yaml",
                                  parcels, [zones, gridcells],
                                  "land_value", cast=True)


# HLCM
@orca.step('hlcm_estimate')
def hlcm_estimate(households_for_estimation, buildings, parcels, zones):
    return utils.lcm_estimate("hlcm.yaml", households_for_estimation,
                              "building_id", buildings, None,
                              out_cfg="hlcmcoef.yaml")


@orca.step('hlcm_simulate')
def hlcm_simulate(households, buildings, persons, settings):
    col = households.building_id
    test = pd.Series(-1, index = np.random.choice(households.index, 100000, False))
    col.update(test)

    households.update_col_from_series('building_id', col, households.index)

    movers = households.to_frame()
    movers = movers[movers.building_id == -1]
    relocated = movers[movers.is_inmigrant < 1]
    res = utils.lcm_simulate("hlcmcoef.yaml", households, buildings,
                             None, "building_id", "residential_units",
                             "vacant_residential_units", cast=True)
    #orca.clear_cache()

    # Determine which relocated persons get disconnected from their job
    if settings.get('remove_jobs_from_workers', False):
        persons_df = persons.to_frame()
        relocated_workers = persons_df.loc[(persons_df.employment_status > 0) &
                                       (persons_df.household_id.isin
                                       (relocated.index))]
        relocated_workers['new_dist_to_work'] = network_distance_from_home_to_work(
                                        relocated_workers.workplace_zone_id,
                                        relocated_workers.household_zone_id)
        relocated_workers['prev_dist_to_work'] = network_distance_from_home_to_work(
                                        relocated_workers.workplace_zone_id,
                                        relocated_workers.prev_household_zone_id)

        # if new distance to work is greater than old, disconnect person from job
        relocated_workers.job_id = np.where(relocated_workers.new_dist_to_work >
                                        relocated_workers.prev_dist_to_work,
                                        -1, relocated_workers.job_id)
        persons.update_col_from_series("job_id", relocated_workers.job_id,
                                   cast=True)

        # Update is_inmigrant- I think this it is ok to do this now,
        # but perhaps this should be part of a clean up step
        # at the end of the sim year.

        households.update_col_from_series("is_inmigrant", pd.Series(0,
                                      index=households.index), cast=True)

    #orca.clear_cache()

    return res

@orca.step('hlcm_simulate_sample')
def hlcm_simulate_sample(households, buildings, persons, settings):

    #df = households.to_frame()
    #df['building_id'] = -1

    col = households.building_id
    test = pd.Series(-1, index = np.random.choice(households.index, 100000, False))
    col.update(test)

    households.update_col_from_series('building_id', col, households.index)
    #col.update(test)
    #households.update_col("building_id", -1)

    res = psrc_dcm.lcm_simulate_sample("hlcmcoef.yaml", households, 'prev_residence_large_area_id', buildings,
                             None, "building_id", "residential_units",
                             "vacant_residential_units", cast=True)
    #orca.clear_cache()

@orca.step('hlcm_estimate_sample')
def hlcm_estimate_sample(households_for_estimation, buildings, persons, settings):

    res = psrc_dcm.lcm_estimate_sample("hlcm.yaml", households_for_estimation, 'prev_residence_large_area_id',
                              "building_id", buildings, None,
                              out_cfg="hlcmcoef.yaml")
    #orca.clear_cache()

# WPLCM
@orca.step('wplcm_estimate')
def wplcm_estimate(persons_for_estimation, jobs):
    return utils.lcm_estimate("wplcm.yaml", persons_for_estimation, "job_id",
                              jobs, None, out_cfg="wplcmcoef.yaml")


# ELCM
@orca.step('elcm_estimate')
def elcm_estimate(jobs, buildings, parcels, zones, gridcells):
    return utils.lcm_estimate("elcm.yaml", jobs, "building_id",
                              buildings, [parcels, zones, gridcells],
                              out_cfg="elcmcoef.yaml")


@orca.step('elcm_simulate')
def elcm_simulate(jobs, buildings, parcels, zones, gridcells):
    res = utils.lcm_simulate("elcmcoef.yaml", jobs, buildings,
                             [parcels, zones, gridcells],
                             "building_id", "job_spaces", "vacant_job_spaces",
                             cast=True)
    #orca.clear_cache()


@orca.step('households_relocation')
def households_relocation(households, household_relocation_rates):
    from urbansim.models import relocation as relo
    rm = relo.RelocationModel(household_relocation_rates.to_frame(),
                              rate_column='probability_of_relocating')
    movers = rm.find_movers(households.to_frame()
                            [households.building_id > 0])  # relocate only residents
    print "%s households selected to move." % movers.size
    households.update_col_from_series("building_id",
                                      pd.Series(-1, index=movers), cast=True)
    print "%s households are unplaced in total." % ((households.local["building_id"] <= 0).sum())

@orca.step('jobs_relocation')
def jobs_relocation(jobs, job_relocation_rates):
    from urbansim.models import relocation as relo
    rm = relo.RelocationModel(job_relocation_rates.to_frame(), rate_column='job_relocation_probability')
    movers = rm.find_movers(jobs.to_frame()[jobs.building_id > 0])  # relocate only placed jobs
    print "%s jobs selected to move." % movers.size
    jobs.update_col_from_series("building_id",
                                pd.Series(-1, index=movers), cast=True)
    print "%s jobs are unplaced in total." % ((jobs.local["building_id"] <= 0).sum())


@orca.step('update_persons_jobs')
def update_persons_jobs(jobs, persons):

    # Persons whoose jobs have relocated no longer have those jobs
    persons.update_col_from_series("job_id",
                                   pd.Series(np.where(persons.job_id.isin
                                             (jobs.building_id[jobs.building_id == -1].
                                              index), -1, persons.job_id),
                                             index=persons.index),
                                             cast=True)

    # Their home-based status should be set to 0 for now.
    # Because their job_id is -1, they will be run through the wahlcm later:
    persons.update_col_from_series("work_at_home",
                                   pd.Series(np.where(np.logical_or
                                             (persons.job_id.isin
                                              (jobs.building_id
                                               [jobs.building_id == -1].index),
                                              persons.job_id == -1), 0,
                                               persons.work_at_home),
                                             index=persons.index), cast=True)

    # Update jobs available column to reflect which jobs are taken, available:
    jobs.update_col_from_series("vacant_jobs",
                                pd.Series(np.where(jobs.index.isin
                                          (persons.job_id), 0, 1),
                                          index=jobs.index), cast=True)


@orca.step('households_transition')
def households_transition(households, household_controls,
                          year, settings, persons):
    orig_size_hh = households.local.shape[0]
    orig_size_pers = persons.local.shape[0]
    orig_pers_index = persons.index
    orig_hh_index = households.index
    res = utils.full_transition(households, household_controls, year,
                                settings['households_transition'],
                                "building_id",
                                linked_tables={"persons":
                                               (persons.local,
                                                'household_id')})

    print "Net change: %s households" % (orca.get_table("households").
                                         local.shape[0] - orig_size_hh)
    print "Net change: %s persons" % (orca.get_table("persons").
                                      local.shape[0] - orig_size_pers)

    # changes to households/persons table are not reflected in local scope
    # need to reset vars to get changes.
    households = orca.get_table('households')
    persons = orca.get_table("persons")

    # need to make some updates to the persons & households table
    households.update_col_from_series("is_inmigrant",
                                      pd.Series(np.where
                                                (~households.index.isin
                                                 (orig_hh_index), 1, 0),
                                                index=households.index),
                                      cast=True)

    households.update_col_from_series("previous_building_id",
                                      pd.Series(np.where
                                                (~households.index.isin
                                                 (orig_hh_index), -1, households.previous_building_id),
                                                index=households.index),
                                      cast=True)

    # new workers dont have jobs yet, set job_id to -1
    persons.update_col_from_series("job_id",
                                   pd.Series(np.where(~persons.index.isin
                                             (orig_pers_index), -1,
                                             persons.job_id),
                                             index=persons.index), cast=True)

    # dont know their work at home status yet, set to 0:
    persons.update_col_from_series("work_at_home",
                                   pd.Series(np.where
                                             (~persons.index.isin
                                              (orig_pers_index), 0,
                                              persons.work_at_home),
                                             index=persons.index), cast=True)
    # set non-worker job_id to -2
    persons.update_col_from_series("job_id",
                                   pd.Series(np.where
                                             (persons.employment_status > 0,
                                              persons.job_id, -2),
                                             index=persons.index), cast=True)
    #orca.clear_cache()
    return res


@orca.step('jobs_transition')
def jobs_transition(jobs, employment_controls, year, settings):
    orig_size = jobs.local.shape[0]
    res = utils.full_transition(jobs,
                                employment_controls,
                                year,
                                settings['jobs_transition'],
                                "building_id")
    print "Net change: %s jobs" % (orca.get_table("jobs").local.shape[0]-
                                   orig_size)
    return res


@orca.step('governmental_jobs_scaling')
def governmental_jobs_scaling(jobs, buildings, year):
    orca.add_column('buildings', 'existing', np.zeros(len(buildings),
                    dtype="int32"))
    alloc = AgentAllocationModel('existing', 'number_of_governmental_jobs',
                                 as_delta=False)
    jobs_to_place = jobs.local[np.logical_and(np.in1d(jobs.sector_id,
                                              [18, 19]), jobs.building_id < 0)]
    print "Locating %s governmental jobs" % len(jobs_to_place)
    loc_ids, loc_allo = alloc.locate_agents(orca.get_table
                                            ("buildings").to_frame
                                            (buildings.local_columns +
                                             ['number_of_governmental_jobs',
                                              'existing']), jobs_to_place,
                                            year=year)
    jobs.local.loc[loc_ids.index, buildings.index.name] = loc_ids
    print "Number of unplaced governmental jobs: %s" % np.logical_or(np.isnan(loc_ids), loc_ids < 0).sum()
    orca.add_table(jobs.name, jobs.local)

@orca.step('create_proforma_config')
def create_proforma_config(proforma_settings):
    yaml_file = misc.config("proforma_user.yaml")
    user_cfg = yamlio.yaml_to_dict(str_or_buffer=yaml_file)
    config = psrcdev.update_sqftproforma(user_cfg, proforma_settings)
    yamlio.convert_to_yaml(config, "proforma.yaml")
    
@orca.step('proforma_feasibility')
def proforma_feasibility(parcels, proforma_settings, parcel_price_placeholder, parcel_sales_price_sqft_func, 
                         parcel_is_allowed_func):

    development_filter = "capacity_opportunity_non_gov" # includes empty parcels
    pcl = parcels.to_frame(parcels.local_columns + ['max_far', 'max_dua', 'max_height', 'ave_unit_size', 'parcel_size', 'land_cost'])
    # reduce parcel dataset to those that can be developed
    if development_filter is not None:
        pcl = pcl.loc[parcels[development_filter] == True]
    df = orca.DataFrameWrapper("parcels", pcl, copy_col=False)
    # create a feasibility dataset
    sqftproforma.run_feasibility(df, parcel_sales_price_sqft_func,
                                 #parcel_price_placeholder, 
                                 parcel_is_allowed_func, cfg = "proforma.yaml",
                                parcel_custom_callback = parcel_sales_price_sqft_func,
                                proforma_uses = proforma_settings)
    #projects = orca.get_table("feasibility")
    #p = projects.local.stack(level=0)
    #pp = p.reset_index()
    #pp.rename(columns = {'level_1':'form'}, inplace=True)
    #pp.to_csv("proforma_projects.csv")
    return

@orca.step('developer_picker')
def developer_picker(feasibility, buildings, parcels, year, target_vacancy, proposal_selection, building_sqft_per_job):
    target_units = psrcdev.compute_target_units(target_vacancy)
    new_buildings = psrcdev.run_developer(forms = [],
                        agents = None,
                        buildings = buildings,
                        supply_fname = ["residential_units", "job_spaces"],
                        feasibility = feasibility,
                        parcel_size = parcels.parcel_size,
                        ave_unit_size = {"single_family_residential": parcels.ave_unit_size_sf, 
                                         "multi_family_residential": parcels.ave_unit_size_mf,
                                         "condo_residential": parcels.ave_unit_size_condo},
                        cfg = 'developer.yaml',
                        year = year,
                        num_units_to_build = target_units,
                        add_more_columns_callback = add_extra_columns,
                        custom_selection_func = proposal_selection,
                        building_sqft_per_job = building_sqft_per_job
                        )

def random_type(form):
    form_to_btype = orca.get_injectable("form_to_btype")
    return random.choice(form_to_btype[form])


def add_extra_columns(df, new_cols = {}):
    bldgs = orca.get_table('buildings')
    for col in bldgs.local_columns:
        if col in new_cols.keys():
            df[col] = new_cols[col]
        elif col not in df.columns:
            df[col] = 0
    return df


@orca.step('add_lag1_tables')
def add_lag1_tables(year, simfile, settings):
    add_lag_tables(1, year, settings['base_year'],
                   simfile, ["households", "buildings"])


def add_lag_tables(lag, year, base_year, filename, table_names):
    store = pd.HDFStore(filename, mode="r")
    prefix = max(year-lag, base_year)
    if prefix == base_year:
        prefix = "base"
    key_template = '{}/{{}}'.format(prefix)
    for table in table_names:
        orca.add_table("{}_lag1".format(table),
                       store[key_template.format(table)], cache=True)
    store.close()


@orca.step('update_household_previous_building_id')
def update_household_previous_building_id(households):
    households.update_col_from_series("previous_building_id",
                                      households.building_id, cast=True)


@orca.step('update_buildings_lag1')
def update_buildings_lag1(buildings):
    df = buildings.to_frame()
    orca.add_table('buildings_lag1', df, cache=True)


#@orca.step('clear_cache')
##def clear_cache():
#    orca.clear_cache()