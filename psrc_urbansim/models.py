import os
import sys
import orca
import random
import numpy as np
import pandas as pd
import logging
import time
import urbansim_defaults.utils as utils
import urbansim.developer as dev
from urbansim.utils import misc, yamlio
import psrc_urbansim.utils as psrcutils
import psrc_urbansim.datasources
import psrc_urbansim.variables
import psrc_urbansim.vars.variables_persons as variables_persons
import psrc_urbansim.vars.variables_households as variables_households
from psrc_urbansim.vars.variables_interactions import network_distance_from_home_to_work
from psrc_urbansim.mod.allocation import AgentAllocationModel
import psrc_urbansim.developer_models as psrcdev
import psrc_urbansim.dcm_weighted_sampling as psrc_dcm
import psrc_urbansim.sqftproforma as sqftproforma
from psrc_urbansim.binary_discrete_choice import BinaryDiscreteChoiceModel

logger = logging.getLogger(__name__)

# dummy steps for logging and timing the current year
@orca.step('start_year')
def start_year(year):
    logger.info("{} **** START YEAR {} **** \n============".format(time.strftime("%Y-%m-%d"), year))
    orca.add_injectable('iter_start_time', time.time())
    if not orca.is_injectable('iter_start_simulation_time'):
        orca.add_injectable('iter_start_simulation_time', time.time())
        
    
@orca.step('end_year')
def end_year(year):
    tend = time.time()    
    logger.info("{} **** END YEAR {} ****. Processing year: {} Processing simulation: {}".format(
        time.strftime("%Y-%m-%d"), year, 
        time.strftime("%H:%M:%S", time.gmtime(tend - orca.get_injectable("iter_start_time"))),
        time.strftime("%H:%M:%S", time.gmtime(tend - orca.get_injectable("iter_start_simulation_time")))
    ))
    

# Residential REPM
@orca.step('repmres_estimate')
def repmres_estimate(parcels, zones, gridcells):
    return utils.hedonic_estimate("repmres.yaml", parcels, [zones, gridcells],
                                  out_cfg="repmrescoef.yaml")


@orca.step('repmres_simulate')
def repmres_simulate(parcels, zones, gridcells, year, settings):
    return psrcutils.hedonic_simulate("repmrescoef.yaml", parcels,
                                  [zones, gridcells], "land_value", cast=True,
                                  # compute residuals only in the first simulation year
                                  compute_residuals = year == (settings['base_year'] + 1), 
                                  add_residuals = True, 
                                  settings = settings.get("real_estate_price_model", {}))


# Non-Residential REPM
@orca.step('repmnr_estimate')
def repmnr_estimate(parcels, zones, gridcells):
    return utils.hedonic_estimate("repmnr.yaml", parcels,
                                  [zones, gridcells],
                                  out_cfg="repmnrcoef.yaml")


@orca.step('repmnr_simulate')
def repmnr_simulate(parcels, zones, gridcells, year, settings):
    return psrcutils.hedonic_simulate("repmnrcoef.yaml", parcels, 
                                      [zones, gridcells], "land_value", cast=True,
                                      # compute residuals only in the first simulation year
                                      compute_residuals = year == (settings['base_year'] + 1),
                                      add_residuals = True, 
                                      settings = settings.get("real_estate_price_model", {}))


# HLCM
@orca.step('hlcm_estimate')
def hlcm_estimate(households_for_estimation, buildings, parcels, zones):
    return utils.lcm_estimate("hlcm.yaml", households_for_estimation,
                              "building_id", buildings, None,
                              out_cfg="hlcmcoef.yaml")


@orca.step('hlcm_simulate')
def hlcm_simulate(households, buildings, persons, settings):
    movers = households.to_frame(households.local_columns)
    movers = movers[movers.building_id == -1]
    relocated = movers[movers.is_inmigrant < 1]
    res = psrc_dcm.lcm_simulate("hlcmcoef.yaml", households, buildings,
                                settings['min_overfull_buildings'],
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

    res = psrc_dcm.lcm_simulate_sample(settings.get("hlcmcoef_file", "hlcmcoef.yaml"), households, 'prev_residence_large_area_id', buildings, 
                                       settings['min_overfull_buildings'], settings['large_area_sample'], None, "building_id", "residential_units",
                             "vacant_residential_units", cast=True)
    
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
    res = psrc_dcm.lcm_simulate("elcmcoef.yaml", jobs, buildings, 0,
                             [parcels, zones, gridcells],
                             "building_id", "job_spaces", "vacant_job_spaces",
                             cast=True)


# Relocation models
@orca.step('households_relocation')
def households_relocation(households, household_relocation_rates):
    from urbansim.models import relocation as relo
    rm = relo.RelocationModel(household_relocation_rates.to_frame(),
                              rate_column='probability_of_relocating')
    movers = rm.find_movers(households.to_frame(households.local_columns + ['building_type_id'])
                            [households.building_id > 0])  # relocate only residents
    households.update_col_from_series("building_id",
                                      pd.Series(-1, index=movers), cast=True)
    logger.info("{} households selected to move. {} households are unplaced in total.".format(
        (households.local["building_id"] <= 0).sum(), movers.size))

@orca.step('household_logit_relocation_estimate')
def household_logit_relocation_estimate(households_for_estimation):
    from workplace_models import to_frame
    cfg_file = misc.config("hhreloc.yaml")
    cfg = yamlio.yaml_to_dict(str_or_buffer=cfg_file)
    choice_attr = cfg.get("choice_column", "move")
    choosers = to_frame(households_for_estimation, [], cfg_file, additional_columns = [choice_attr])
    out_cfg_file = misc.config("hhreloccoef.yaml")
    return BinaryDiscreteChoiceModel.fit_from_cfg(choosers, choice_attr, 
                                                  cfg_file, outcfgname=out_cfg_file)

@orca.step('household_logit_relocation_simulate')
def household_logit_relocation_simulate(households):
    # TODO: test this model
    cfg = misc.config("hhreloccoef.yaml")
    choosers = to_frame(households, [], cfg)
    return BinaryDiscreteChoiceModel.predict_from_cfg(choosers, cfg)



@orca.step('jobs_relocation')
def jobs_relocation(jobs, job_relocation_rates):
    from urbansim.models import relocation as relo
    rm = relo.RelocationModel(job_relocation_rates.to_frame(), rate_column='job_relocation_probability')
    movers = rm.find_movers(jobs.to_frame(jobs.local_columns)[jobs.building_id > 0])  # relocate only placed jobs
    jobs.update_col_from_series("building_id",
                                pd.Series(-1, index=movers), cast=True)
    logger.info("{} jobs selected to move. {} jobs are unplaced in total.".format((jobs.local["building_id"] <= 0).sum(), movers.size))


@orca.step('update_persons_jobs')
def update_persons_jobs(jobs, persons):

    # Persons whose jobs have relocated no longer have those jobs
    persons.update_col_from_series("job_id",
                                   pd.Series(np.where(persons.job_id.isin
                                             (jobs.building_id[jobs.building_id == -1].
                                              index), -1, persons.job_id),
                                             index=persons.index),
                                             cast=True)

    # Persons whose job no longer exists should have their job_id set to  -1
    persons.update_col_from_series("job_id",
                                   pd.Series(np.where(persons.job_id.isin
                                             (jobs.index), persons.job_id, -1),
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


@orca.step('households_transition')
def households_transition(households, household_controls,
                          year, settings, persons):
    return run_households_transition(households, household_controls,
                              year, settings, persons)    

def run_households_transition(households, household_controls,
                          year, settings, persons, is_allocation = False):
    orig_size_hh = households.local.shape[0]
    orig_size_pers = persons.local.shape[0]
    orig_pers_index = persons.index
    orig_hh_index = households.index
    orig_hh_local_columns = households.local_columns
    
    config = settings['households_transition']
    if len(config.get('remove_columns', [])) > 0:
        for column in [config.get('remove_columns', [])]:
            if column in household_controls.local.columns:
                household_controls.local.drop(config.get('remove_columns', []), axis = 1, inplace = True)
    res = utils.full_transition(households, household_controls, year, config, "building_id",
                                linked_tables={"persons":
                                               (persons.local,
                                                'household_id')})
    # the transition model removes index name, so put it back
    orca.get_table("persons").index.name = persons.index.name
    orca.get_table("households").index.name = households.index.name
        
    if not is_allocation:
        # Need to resave the table in orca because computed columns became local columns.
        # Not needed in allocation mode, since subregional geo id should be visible to other models.
        resave_table_in_orca(orca.get_table("households"), orig_hh_local_columns)
            
    # changes to households/persons table are not reflected in local scope
    # need to reset vars to get changes.
    households = orca.get_table('households')
    persons = orca.get_table("persons") 
    
    logger.info("Net change: {} households (before: {}, after: {})".format(households.local.shape[0] - orig_size_hh, orig_size_hh, households.local.shape[0]))
    logger.info("Net change: {} persons (before: {}, after: {})".format(persons.local.shape[0] - orig_size_pers, orig_size_pers, persons.local.shape[0]))
    
    # need to make some updates to the persons & households table
    households = update_local_scope(households, "is_inmigrant", 
                                    np.where(~households.index.isin (orig_hh_index), 1, 0))
    households = update_local_scope(households, "previous_building_id", 
                                    np.where(~households.index.isin (orig_hh_index), -1, households.previous_building_id))
    
    # new workers dont have jobs yet, set job_id to -1
    persons = update_local_scope(persons, "job_id", 
                                    np.where(~persons.index.isin(orig_pers_index), -1, persons.job_id))
    # dont know their work at home status yet, set to 0:
    persons = update_local_scope(persons, "work_at_home", 
                                    np.where(~persons.index.isin(orig_pers_index), 0, persons.work_at_home))    
    # set non-worker job_id to -2
    persons = update_local_scope(persons, "job_id", 
                                    np.where(persons.employment_status > 0, persons.job_id, -2))

    return res

def update_local_scope(table, column, values):
    table.update_col_from_series(column, pd.Series(values, index=table.index), cast=True)
    return table

@orca.step('jobs_transition')
def jobs_transition(jobs, employment_controls, year, settings):
    return run_jobs_transition(jobs, employment_controls, year, settings)
    
def run_jobs_transition(jobs, employment_controls, year, settings, is_allocation = False):
    orig_size = jobs.local.shape[0]
    orig_job_local_columns = jobs.local_columns
    config = settings['jobs_transition']
    if len(config.get('remove_columns', [])) > 0:
            for column in [config.get('remove_columns', [])]:
                if column in employment_controls.local.columns:
                    employment_controls.local.drop(config.get('remove_columns', []), axis = 1, inplace = True)
 
    res = utils.full_transition(jobs, employment_controls, year, config, "building_id")
    
    # the transition model removes index name, so put it back
    orca.get_table("jobs").index.name = jobs.index.name
    logger.info("Net change: {} jobs (before: {}, after: {})".format(orca.get_table("jobs").local.shape[0]-orig_size, orig_size, orca.get_table("jobs").local.shape[0]))
    if not is_allocation:
        # Need to resave the table in orca because computed columns became local columns.
        # Not needed in allocation mode, since subregional geo id should be visible to other models.
        resave_table_in_orca(orca.get_table("jobs"), orig_job_local_columns)
   
    return res


@orca.step('governmental_jobs_scaling')
def governmental_jobs_scaling(jobs, buildings, year, settings):
    jobs_to_place_bool = np.logical_and(np.in1d(jobs.sector_id,
                                              [12, 13]), jobs.building_id < 0)
    logger.info("Locating {} governmental jobs".format(sum(jobs_to_place_bool)))
    loc_ids = run_scaling('number_of_governmental_jobs', jobs, jobs_to_place_bool, buildings, year, settings)
    logger.info("Number of unplaced governmental jobs: {}".format(np.logical_or(np.isnan(loc_ids), loc_ids < 0).sum()))
    
    
def run_scaling(number_of_agents_column, agents, agents_to_place_bool, buildings, year, settings, is_allocation = False):
    orca.add_column('buildings', 'existing', np.zeros(len(buildings),
                    dtype="int32"))
    alloc = AgentAllocationModel('existing', number_of_agents_column,
                                 as_delta=False)
    if is_allocation:
        subreg_geo_id = settings.get("control_geography_id", "city_id")
        subregs = np.unique(agents[subreg_geo_id][agents_to_place_bool])
        loc_ids = pd.Series(np.array([]))
        bldgs = orca.get_table("buildings").to_frame(buildings.local_columns +
                                                 [number_of_agents_column, subreg_geo_id,
                                                  'existing'])
        for subreg in subregs:
            place = np.logical_and(agents_to_place_bool, agents[subreg_geo_id] == subreg)
            if place.sum() == 0:
                continue
            this_loc_ids, loc_allo = alloc.locate_agents(bldgs.loc[bldgs[subreg_geo_id] == subreg], 
                                                         agents.local[place], year=year)
            loc_ids = pd.concat((loc_ids, this_loc_ids))
    else:
        agents_to_place =  agents.local[agents_to_place_bool]
        loc_ids, loc_allo = alloc.locate_agents(orca.get_table
                                            ("buildings").to_frame
                                            (buildings.local_columns +
                                             [number_of_agents_column,
                                              'existing']), agents_to_place,
                                            year=year)
    agents.local.loc[loc_ids.index, buildings.index.name] = loc_ids
    orca.add_table(agents.name, agents.local)
    return loc_ids

@orca.step('process_mpds')
def process_mpds(mpds_for_year, buildings):
    psrcdev.do_process_mpds(mpds_for_year, buildings)
                         
@orca.step('create_proforma_config')
def create_proforma_config(proforma_settings):
    yaml_file = misc.config("proforma_user.yaml")
    user_cfg = yamlio.yaml_to_dict(str_or_buffer=yaml_file)
    config = psrcdev.update_sqftproforma(user_cfg, proforma_settings)
    yamlio.convert_to_yaml(config, "proforma.yaml")
    

@orca.step('proforma_feasibility')
def proforma_feasibility(parcels, uses_and_forms, parcel_price_placeholder, parcel_sales_price_func, 
                         parcel_is_allowed_func, set_ave_unit_size_func, settings):

    return run_proforma_feasibility_model(parcels, uses_and_forms, parcel_price_placeholder, parcel_sales_price_func, 
                             parcel_is_allowed_func, set_ave_unit_size_func, settings.get("feasibility_model", {}))

def run_proforma_feasibility_model(parcels, uses_and_forms, parcel_price_placeholder, parcel_sales_price_func, 
                         parcel_is_allowed_func, set_ave_unit_size_func, model_settings):

    development_filter = model_settings.get("development_filter", "capacity_opportunity_non_gov") # variable should include empty parcels
    #development_filter = "developable"
    
    pcl = parcels.to_frame(parcels.local_columns + ['max_far', 'max_dua', 'max_height', 'max_coverage', 
                                                    'ave_unit_size_sf', 'ave_unit_size_mf', 'ave_unit_size_condo',
                                                    'parcel_size', 'land_cost'])

    # reduce parcel dataset to those that can be developed
    if development_filter is not None:
        pcl = pcl.loc[parcels[development_filter] == True]
        #pcl = pcl.iloc[1:1000,:]
    df = orca.DataFrameWrapper("parcels", pcl, copy_col=False)
    # create a feasibility dataset
    sqftproforma.run_feasibility(df, parcel_sales_price_func,
                                 parcel_is_allowed_func, cfg = model_settings.get("config_file", "proforma.yaml"),
                                proforma_uses = uses_and_forms,
                                lookup_modify_callback = set_ave_unit_size_func)
    #projects = orca.get_table("feasibility")
    #p = projects.local.stack(level=0)
    #pp = p.reset_index()
    #pp.rename(columns = {'level_1':'form'}, inplace=True)
    #pp.to_csv("proforma_projects.csv")
    return
    
@orca.step('developer_picker')
def developer_picker(feasibility, buildings, parcels, year, target_vacancy, proposal_selection_probabilities, proposal_selection, building_sqft_per_job):
    target_units = psrcdev.compute_target_units(target_vacancy, unlimited = False)
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
                        #profit_to_prob_func = proposal_selection_probabilities,
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

@orca.step('update_misc_building_columns')
def update_misc_building_columns(buildings):
    orca.add_column('buildings', 'existing', np.zeros(len(buildings),
                    dtype="int32"))

@orca.step('update_household_previous_building_id')
def update_household_previous_building_id(households):
    households.update_col_from_series("previous_building_id",
                                      households.building_id, cast=True)


@orca.step('update_buildings_lag1')
def update_buildings_lag1(buildings):
    df = buildings.to_frame(buildings.local_columns)
    orca.add_table('buildings_lag1', df, cache=True)


#@orca.step('clear_cache')
##def clear_cache():
#    orca.clear_cache()

##############################
### Models for ALLOCATION mode
##############################
@orca.step('boost_residential_density')
def boost_residential_density(buildings, year, settings):
    conf = settings.get("boost_residential_density", {})
    boost_density(buildings, "residential_units", year, settings['base_year'], conf)

@orca.step('boost_nonresidential_density')
def boost_nonresidential_density(buildings, year, settings):
    conf = settings.get("boost_nonresidential_density", {})
    attr = "non_residential_sqft"
    if year <= settings['base_year']:
        attr = "job_capacity"
    boost_density(buildings, attr, year, settings['base_year'], conf)


def boost_density(buildings, attribute, year, base_year, conf):
    filter = conf.get("filter", None)
    if filter:
        is_in = buildings[filter].astype('bool8')
    else:
        is_in = np.ones(len(buildings), dtype = "bool8")
    # if not first year boost new development only
    if year > base_year:
        is_in = np.logical_and(is_in, buildings.year_built == year)
    boost = conf.get("boost_factor", 1)
    buildings.update_col_from_series(attribute, (boost * buildings[attribute][is_in]).round(0), cast = True)  

@orca.step('proforma_feasibility_alloc')
def proforma_feasibility_alloc(isCY, parcels, uses_and_forms, parcel_price_placeholder, parcel_sales_price_func, 
                         parcel_is_allowed_func, parcel_is_allowed_func_with_cap, set_ave_unit_size_func, settings):
    if isCY:
        logger.info("Running proforma_feasibility for control year")
        return proforma_feasibility_CY(parcels, uses_and_forms, parcel_price_placeholder, parcel_sales_price_func, 
                         parcel_is_allowed_func, set_ave_unit_size_func, settings)
    logger.info("Running proforma_feasibility for non-control year")
    return proforma_feasibility(parcels, uses_and_forms, parcel_price_placeholder, parcel_sales_price_func, 
                         parcel_is_allowed_func_with_cap, set_ave_unit_size_func, settings)

@orca.step('proforma_feasibility_CY') # for running in control years, should have relaxed redevelopment filter
def proforma_feasibility_CY(parcels, uses_and_forms, parcel_price_placeholder, parcel_sales_price_func, 
                         parcel_is_allowed_func, set_ave_unit_size_func, settings):

    return run_proforma_feasibility_model(parcels, uses_and_forms, parcel_price_placeholder, parcel_sales_price_func, 
                             parcel_is_allowed_func, set_ave_unit_size_func, settings.get("feasibility_model_CY", {}))

@orca.step('developer_picker_alloc') 
def developer_picker_alloc(isCY, feasibility, buildings, parcels, year, target_vacancy, proposal_selection_probabilities, 
                        proposal_selection, building_sqft_per_job, settings):
    if isCY:
        logger.info("Running developer_picker for control year")
        return developer_picker_CY(feasibility, buildings, parcels, year, proposal_selection_probabilities, 
                        proposal_selection, building_sqft_per_job, settings)
    logger.info("Running developer_picker for non-control year")
    return developer_picker(feasibility, buildings, parcels, year, target_vacancy, proposal_selection_probabilities, 
                        proposal_selection, building_sqft_per_job)
    
@orca.step('developer_picker_CY') # for running in control years, runs for each subreg separately
def developer_picker_CY(feasibility, buildings, parcels, year, proposal_selection_probabilities, 
                        proposal_selection, building_sqft_per_job, settings):
    subreg_geo_id = settings.get("control_geography_id", "city_id")
    new_buildings = psrcdev.run_developer_CY(
        subreg_geo_id = subreg_geo_id,
        forms = [],
                        agents = None,
                        buildings = buildings,
                        supply_fname = ["residential_units", "job_spaces"],
                        feasibility = feasibility,
                        parcel_size = parcels.parcel_size,
                        ave_unit_size = {"single_family_residential": parcels.ave_unit_size_sf, 
                                         "multi_family_residential": parcels.ave_unit_size_mf,
                                         "condo_residential": parcels.ave_unit_size_condo},
                        cfg = 'developer_CY.yaml',
                        year = year,
                        add_more_columns_callback = add_extra_columns,
                        #profit_to_prob_func = proposal_selection_probabilities,
                        custom_selection_func = proposal_selection,
                        building_sqft_per_job = building_sqft_per_job
                        )

@orca.step('households_transition_alloc')
def households_transition_alloc(isCY, households, household_controls, year, settings, persons):
    run_households_transition(households, household_controls, year, settings, persons, is_allocation = isCY)
    pers = orca.get_table("persons")
    hh = orca.get_table("households")
    if (~pers.household_id.isin(hh.index)).any(): 
        # persons exist that do not have HHs 
        # (because those HHs were unplaced and thus, excluded from the Transition)       
        pers = pers.local.loc[pers["household_id"].isin(hh.index)]
        orca.add_table("persons", pers)
        logger.info("Total persons after cleaning: {}".format(len(pers)))

@orca.step('jobs_transition_alloc')
def jobs_transition_alloc(isCY, jobs, employment_controls, year, settings):
    return run_jobs_transition(jobs, employment_controls, year, settings, is_allocation = isCY)

@orca.step('households_relocation_alloc')
def households_relocation_alloc(isCY, households, household_relocation_rates):
    if isCY:
        logger.info("No households relocation in control year. {} households are unplaced in total.".format((households.local["building_id"] <= 0).sum()))
    else:
        households_relocation(households, household_relocation_rates)

@orca.step('jobs_relocation_alloc')
def jobs_relocation_alloc(isCY, jobs, job_relocation_rates):
    if isCY:
        logger.info("No jobs relocation in control year. {} jobs are unplaced in total.".format((jobs.local["building_id"] <= 0).sum()))
    else:
        jobs_relocation(jobs, job_relocation_rates)

@orca.step('hlcm_simulate_alloc')
def hlcm_simulate_alloc(isCY, households, buildings, persons, settings):
    if isCY:
        subreg_geo_id = settings.get("control_geography_id", "city_id")
        psrcutils.lcm_simulate_CY(subreg_geo_id, settings.get("hlcmcoefCY_file", "hlcmcoef.yaml"), households, buildings, 
                                  None, "building_id", "residential_units",
                             "vacant_residential_units", 
                             min_overfull_buildings=settings.get('min_overfull_buildings', 0), 
                             settings = settings.get("household_location_choice_model_CY", {}),
                             cast=True)
    else:
        hlcm_simulate_sample(households, buildings, persons, settings)


@orca.step('elcm_simulate_alloc')
def elcm_simulate_alloc(isCY, jobs, buildings, parcels, zones, gridcells, settings):
    if isCY:
        subreg_geo_id = settings.get("control_geography_id", "city_id")
        psrcutils.lcm_simulate_CY(subreg_geo_id, "elcmcoef.yaml", jobs, buildings, [parcels, zones, gridcells], 
                                  "building_id", "job_spaces", "vacant_job_spaces", 
                                  settings = settings.get("employment_location_choice_model_CY", {}),
                                  cast=True)
    else:
        elcm_simulate(jobs, buildings, parcels, zones, gridcells)

@orca.step('governmental_jobs_scaling_alloc')
def governmental_jobs_scaling_alloc(isCY, jobs, buildings, year, settings):
    if isCY:
        jobs_to_place_bool = np.logical_and(np.in1d(jobs.sector_id,
                                                  [12, 13]), jobs.building_id < 0)
        logger.info("Locating {} governmental jobs by subregion".format(sum(jobs_to_place_bool)))
        loc_ids = run_scaling('number_of_governmental_jobs', jobs, jobs_to_place_bool, buildings, year, settings, is_allocation = True)
        logger.info("Number of unplaced governmental jobs: {}".format(np.logical_or(np.isnan(loc_ids), loc_ids < 0).sum()))
    else:
        governmental_jobs_scaling(jobs, buildings, year, settings)

@orca.step('scaling_unplaced_jobs')
def scaling_unplaced_jobs(isCY, jobs, buildings, year, settings):
    if isCY:
        jobs_to_place_bool = jobs.building_id < 0
        logger.info("Locating {} unplaced jobs by subregion".format(sum(jobs_to_place_bool)))
        loc_ids = run_scaling('number_of_non_home_based_jobs', jobs, jobs_to_place_bool,
                              buildings, year, settings, is_allocation = True)
        logger.info("Number of unplaced jobs: {}".format(np.logical_or(np.isnan(loc_ids), loc_ids < 0).sum()))

@orca.step('scaling_unplaced_households')
def scaling_unplaced_households(isCY, households, buildings, year, settings):
    if isCY:
        hhs_to_place_bool = households.building_id < 0
        logger.info("Locating {} unplaced households by subregion".format(sum(hhs_to_place_bool)))
        loc_ids = run_scaling('number_of_households', households, hhs_to_place_bool, buildings, year, settings, is_allocation = True)
        logger.info("Number of unplaced households: {}".format(np.logical_or(np.isnan(loc_ids), loc_ids < 0).sum()))

@orca.step('delete_subreg_geo_from_households')
def delete_subreg_geo_from_households(households, settings):
    subreg_geo_id = settings.get("control_geography_id", "city_id")
    if subreg_geo_id not in households.local_columns:
        return
    cols = [col for col in households.local_columns if col != subreg_geo_id]
    resave_table_in_orca(households, cols)

@orca.step('delete_subreg_geo_from_jobs')
def delete_subreg_geo_from_jobs(jobs, settings):
    subreg_geo_id = settings.get("control_geography_id", "city_id")
    if subreg_geo_id not in jobs.local_columns:
        return
    cols = [col for col in jobs.local_columns if col != subreg_geo_id]
    resave_table_in_orca(jobs, cols)
    
    
@orca.step('cap_residential_development')
def cap_residential_development(isCY, parcels, household_controls, year, control_years, settings):
    if not isCY:
        subreg_geo_id = settings.get("control_geography_id", "city_id")
        return cap_development(parcels, household_controls, year, subreg_geo_id, control_years, 
                               "total_number_of_households", "residential_units", "cap_residential_development")
    
@orca.step('cap_nonresidential_development')
def cap_nonresidential_development(isCY, parcels, employment_controls, year, control_years, settings):
    if not isCY:
        subreg_geo_id = settings.get("control_geography_id", "city_id")
        return cap_development(parcels, employment_controls, year, subreg_geo_id, control_years, 
                               "total_number_of_jobs", "total_job_spaces", "cap_nonresidential_development")

def cap_development(parcels, control_totals, year, geo_id, control_years, 
                    ct_attribute, units_attribute, out_attribute):
    # find the closest control year and its targets
    cys = np.sort(np.array(control_years))
    cy = cys[cys > year].min()
    ct = control_totals.local.loc[cy][ct_attribute].groupby(control_totals.local.loc[cy][geo_id]).sum()
    # find what is already built
    units = parcels[units_attribute]
    units_geo = units.groupby(parcels[geo_id]).sum()
    # merge and find parcels where no more development desired
    mct = pd.concat((units_geo, ct), axis = 1)
    mct.columns = [units_attribute, ct_attribute]
    cap_geos = mct[units_attribute].fillna(0) > mct[ct_attribute]*1.05
    # save as a new attribute of the parcels table
    cap_pcl = parcels[geo_id].isin(cap_geos[cap_geos == True].index.values)
    if(out_attribute in parcels.columns):
        parcels.update_col_from_series(out_attribute, cap_pcl)
    else:
        parcels.update_col(out_attribute, cap_pcl)

@orca.step('households_events_model')
def households_events_model(households, households_events, year, buildings, settings):
    run_agent_events_model(households, households_events, year, settings, geo_id = "parcel_id", 
                           location_characteristics = ['subreg_id', 'building_type_id'],
                           disaggregate_to = buildings, disaggregation_weight_column = "residential_units")

@orca.step('households_zone_events_model')
def households_zone_events_model(households, households_zone_events, year, buildings, settings):
    run_agent_events_model(households, households_zone_events, year, settings, geo_id = "zone_id", 
                           location_characteristics = ['subreg_id'], disaggregate_to = buildings, 
                           disaggregation_weight_column = "residential_units")


def run_agent_events_model(agents, events, year, settings, geo_id, location_characteristics = [], 
                           disaggregate_to = None, disaggregation_weight_column = None):
    year_events = events.to_frame()[events.scheduled_year == year]
    if len(year_events) == 0:
        return
    agents_df = agents.to_frame(list(set(agents.local_columns + [geo_id] + location_characteristics))) # the list(set()) construct ensures unique values in the list
    other_characteristics = set(year_events.columns).difference(set(["change_type", "is_percentage", "scheduled_year", "total_number", geo_id]))
    nagents_orig = len(agents)
    max_agent_id = agents_df.index.max()
    all_added_agents = {}
    
    year_events = year_events.sort_values("change_type", ascending=False) # sort so that "D"s are first
    unplaced_agents = pd.Series([], dtype = agents.index.dtype)
    
    # iterate over rows in the events table
    for irow in range(len(year_events)):
        # affected location id
        location_id = year_events.iloc[irow][geo_id]
        
        # find agents corresponding to the location
        agents_to_consider = agents_df[geo_id] == location_id
        
        # filter by the remaining characteristics in the table
        for characteristics in other_characteristics:
            if year_events.iloc[irow][characteristics] != -1:
                agents_to_consider = np.logical_and(agents_to_consider, agents_df[characteristics] == year_events.iloc[irow][characteristics])
        agents_to_consider = agents_df[agents_to_consider]
        
        # what is the count to add/remove?
        count = year_events.iloc[irow].total_number
        if year_events.iloc[irow].is_percentage > 0:
            count = np.round(count * len(agents_to_consider)/100)
        elif year_events.iloc[irow].change_type == "D":
            count = min(count, len(agents_to_consider))
            
        if count == 0: # if nothing to add or remove skip to the next event
            continue
        
        if year_events.iloc[irow].change_type == "D":  # delete agents
            # randomly sample the desired count 
            agents_to_unplace = np.random.choice(agents_to_consider.index, size = count, replace = False)
            # unplace selected agents
            updated_locations = np.where(agents.index.isin(agents_to_unplace), -1, agents[geo_id])
            if geo_id in agents.local_columns:
                update_local_scope(agents, geo_id, updated_locations)
            elif disaggregate_to is None:
                agents.add_column(geo_id, updated_locations)
            if not disaggregate_to is None and geo_id != disaggregate_to.index.name:
                update_local_scope(agents, disaggregate_to.index.name, 
                                np.where(agents.index.isin(agents_to_unplace), -1, agents[disaggregate_to.index.name]))
            unplaced_agents = pd.concat([unplaced_agents, pd.Series(agents_to_unplace)])
            logger.info("{} agents deleted from location {}".format(len(agents_to_unplace), location_id))
            
        elif year_events.iloc[irow].change_type == "A":  # add agents

            # find locations to assign
            if disaggregate_to is not None: # disaggregate from a higher geography into lower level
                buildings_in_loc = disaggregate_to[geo_id][disaggregate_to[geo_id] == location_id].index
                if len(buildings_in_loc) == 0:
                    logger.warning('No %s locations found for %s=%s. %s agents not created.' % (
                            disaggregate_to.name, geo_id, location_id, count))
                    continue                
                # sample disaggregated locations
                weights = None
                if not disaggregation_weight_column is None:
                    weights = disaggregate_to[disaggregation_weight_column].loc[buildings_in_loc]
                    if weights.sum() == 0:
                        weights = None
                    else:
                        weights = weights/weights.sum()
                new_locations = np.random.choice(buildings_in_loc, size = count, replace = True, p = weights)
                location_name = disaggregate_to.index.name
            else:
                new_locations = np.array([location_id] * count)
                location_name = geo_id
                
            
            # determine agents with the desire characteristics to impute missing characteristics to the new agents
            loc_indicator = pd.Series(True, index = agents_df.index)
            loc_indicator_pool = pd.Series(True, index = unplaced_agents)
            for locchar in location_characteristics:
                if year_events.iloc[irow][locchar] == -1:
                    continue
                this_loc_indicator = np.logical_and(loc_indicator, agents[locchar] == year_events.iloc[irow][locchar])
                loc_indicator_pool = np.logical_and(loc_indicator_pool, agents[locchar].loc[unplaced_agents] == year_events.iloc[irow][locchar])
                if this_loc_indicator.sum() > 0 and count/this_loc_indicator.sum() < 0.33 : # accept only if enough agents remain to sample from
                    loc_indicator = this_loc_indicator
                    
            new_count = count
            if loc_indicator_pool.sum() > 0: # first use the pool of unplaced agents to be placed in the new locations
                eligible_agents_from_pool = loc_indicator_pool[(loc_indicator_pool == True)].index
                if loc_indicator_pool.sum() > count: # sample
                    sampled_agents = np.random.choice(eligible_agents_from_pool, size = count, replace = False)
                else:
                    sampled_agents = eligible_agents_from_pool
                
                agents_df.loc[sampled_agents, location_name] = new_locations[0:len(sampled_agents)]
                update_local_scope(agents, location_name, agents_df[location_name])
                unplaced_agents = unplaced_agents[~unplaced_agents.isin(sampled_agents)]      
                new_count = count - sampled_agents.size
                if new_count > 0:
                    new_locations = new_locations[len(sampled_agents):]
                
            logger.info('%s agents added to location %s by using previously unplaced agents' % (count - new_count, location_id))
            if new_count <= 0: # if the pool had enough agents, skip to the next event
                continue
            
            # if we need more agents to add, create new ones
            # initiate new agents
            data = {agents_df.index.name: np.arange(1, new_count + 1, 1) + max_agent_id,
                    location_name: new_locations}            
            sampled_agents = np.random.choice(loc_indicator[(loc_indicator == True)].index, size = new_count, replace = False)
                
            for characteristics in other_characteristics:
                if year_events.iloc[irow][characteristics] != -1 and not characteristics in location_characteristics:
                    data[characteristics] = np.array([year_events.iloc[irow][characteristics]] * new_count)            
            # impute remaining attributes
            for attr in agents.local_columns:
                if attr not in data.keys():
                    data[attr] = agents[attr].loc[sampled_agents].values                
            all_added_agents[location_id] = pd.DataFrame(data).set_index(agents_df.index.name)
            max_agent_id = all_added_agents[location_id].index.max()
            logger.info('%s agents added to location %s (%s unplaced and %s new agents)' % (count, location_id, count - new_count, new_count))
    
    if len(all_added_agents) > 0:
        new_agents = pd.concat(all_added_agents.values())
        agents_final = pd.concat([agents.to_frame(agents.local_columns), new_agents[agents.local_columns]])
        orca.add_table(agents.name, agents_final)
        logger.info('%s agents added in total' % (len(agents_final)-nagents_orig))



def resave_table_in_orca(table, cols):
    tbl = table.to_frame(cols)
    orca.add_table(table.name, tbl)