import os 
import sys
import orca
import logging
import urbansim_defaults.utils as utils
import psrc_urbansim.utils as psrcutils
from . import datasources
from . import variables
import numpy as np
import pandas as pd
from psrc_urbansim.mod.allocation import AgentAllocationModel
import urbansim.developer as dev
from . import developer_models as psrcdev
import os 
from urbansim.utils import misc
import statsmodels.api as sm
from statsmodels.formula.api import logit, probit, poisson, ols
import random
from urbansim.utils import misc
from psrc_urbansim.binary_discrete_choice import BinaryDiscreteChoiceModel
from . import dcm_weighted_sampling as psrc_dcm

logger = logging.getLogger(__name__)

def update_local_scope(table, column, values):
    table.update_col_from_series(column, pd.Series(values, index=table.index), cast=True)
    return table

def to_frame(tbl, join_tbls, cfg, additional_columns=[]):
    """
    Leverage all the built in functionality of the sim framework to join to
    the specified tables, only accessing the columns used in cfg (the model
    yaml configuration file), an any additionally passed columns (the sim
    framework is smart enough to figure out which table to grab the column
    off of)

    Parameters
    ----------
    tbl : DataFrameWrapper
        The table to join other tables to
    join_tbls : list of DataFrameWrappers or strs
        A list of tables to join to "tbl"
    cfg : str
        The filename of a yaml configuration file from which to parse the
        strings which are actually used by the model
    additional_columns : list of strs
        A list of additional columns to include

    Returns
    -------
    A single DataFrame with the index from tbl and the columns used by cfg
    and any additional columns specified
    """
    join_tbls = join_tbls if isinstance(join_tbls, list) else [join_tbls]
    tables = [tbl] + join_tbls
    cfg = BinaryDiscreteChoiceModel.from_yaml(str_or_buffer=cfg)
    tables = [t for t in tables if t is not None]
    columns = misc.column_list(tables, cfg.columns_used()) + additional_columns
    if len(tables) > 1:
        df = orca.merge_tables(target=tables[0].name,
                               tables=tables, columns=columns)
    else:
        df = tables[0].to_frame(columns)
    utils.check_nas(df)
    return df

# WAHCM
def work_at_home_estimate(cfg, choosers, chosen_fname, join_tbls, out_cfg=None):
    cfg = misc.config(cfg) 
    choosers = to_frame(choosers, join_tbls, cfg, additional_columns=['work_at_home'])
    if out_cfg is not None:
        out_cfg = misc.config(out_cfg)
    return BinaryDiscreteChoiceModel.fit_from_cfg(choosers, chosen_fname,
                                           cfg,
                                           outcfgname=out_cfg)

def work_at_home_simulate(cfg, choosers, join_tbls):
    cfg = misc.config(cfg) 
    choosers = to_frame(choosers, join_tbls, cfg)
    return BinaryDiscreteChoiceModel.predict_from_cfg(choosers, cfg)


@orca.step('wahcm_estimate')
def wahcm_estimate(persons_for_estimation, households_for_estimation, zones):
    return work_at_home_estimate("wahcm.yaml", persons_for_estimation, 'work_at_home', 
                                 [households_for_estimation, zones], 'wahcmcoeff.yaml')    
        
@orca.step('wahcm_simulate')
def wahcm_simulate(persons, jobs, households, zones):
    do_wahcm_simulate(persons, jobs, households, zones)
    
@orca.step('wahcm_simulate_alloc')
def wahcm_simulate_alloc(isCY, persons, jobs, households, zones, settings):
    if isCY:
        do_wahcm_simulate(persons, jobs, households, zones, subreg_geo_id = settings.get("control_geography_id", "city_id"))
    else:
        wahcm_simulate(persons, jobs, households, zones)

def do_wahcm_simulate(persons, jobs, households, zones, subreg_geo_id = None):
    work_at_home_prob = work_at_home_simulate("wahcmcoeff.yaml", persons, 
                                 [households, zones])[1]
    job_cols = jobs.local_columns
    if subreg_geo_id is not None:
        job_cols = job_cols + [subreg_geo_id]
    jobs_df = jobs.to_frame(job_cols)
    
    home_based_jobs = jobs_df[(jobs_df.home_based_status == 1) & (jobs_df.vacant_jobs>0)]

    if subreg_geo_id is None:
        # If there are more vacant home based jobs than home workers, sample home workers using the exact number of vacant home based jobs, weighted by the probablities from the wachm:
        if len(home_based_jobs) > len(work_at_home_prob):
            selected_jobs = home_based_jobs.sample(len(work_at_home_prob)).index.to_series()
            home_workers = work_at_home_prob
            # Otherwise 
        else:
            selected_jobs = home_based_jobs.index.to_series()
            home_workers = work_at_home_prob.sample(len(home_based_jobs), weights = work_at_home_prob.values)
    else: # assign jobs located within the home-residence of workers
        home_workers = pd.Series([], dtype = work_at_home_prob.dtype)
        selected_jobs = pd.Series([], dtype = home_based_jobs.index.dtype)
        subregs = np.unique(persons[subreg_geo_id][work_at_home_prob.index])
        for subreg in subregs:
            this_hb_jobs = home_based_jobs[home_based_jobs[subreg_geo_id] == subreg]
            if this_hb_jobs.size == 0:
                next
            this_home_workers = work_at_home_prob[persons.index[np.logical_and(persons[subreg_geo_id] == subreg, persons.index.isin(work_at_home_prob.index))]]
            if len(this_hb_jobs) > len(this_home_workers):
                this_hb_jobs = this_hb_jobs.sample(len(this_home_workers))
            else:
                this_home_workers = this_home_workers.sample(len(this_hb_jobs), weights = this_home_workers.values)
            selected_jobs = pd.concat((selected_jobs, this_hb_jobs.index.to_series()))
            home_workers = pd.concat((home_workers, this_home_workers))
            
    # update job_id on the persons table
    # should not matter which person gets which home-based job; the arrays should have the same length
    combine_indexes = pd.DataFrame([home_workers.index, selected_jobs]).transpose()
    combine_indexes.columns = ['person_id', 'job_id']
    combine_indexes.set_index('person_id', inplace=True)
    combine_indexes['work_at_home'] = 1
    
    # updates job_id, work_at_home on the persons table where index (person_id) matches in combine_indexes
    persons.update_col_from_series("job_id", combine_indexes.job_id, cast = True)
    persons.update_col_from_series('work_at_home', combine_indexes.work_at_home, cast = True)
    logger.info("%s additional people assigned to work at home." % len(combine_indexes))
                           
    # building_id on jobs table for home based workers should be the household building_id of the person assigned the job
    # get building_id:
    combine_indexes['building_id'] = 0
    combine_indexes.building_id.update(persons.household_building_id)
    
    #update building_id & vacant_jobs on jobs table:
    combine_indexes.reset_index(level = None, inplace = True)
    combine_indexes.set_index('job_id', inplace=True)
    combine_indexes['vacant_jobs'] = 0
    
    # update jobs table- building_id of at home workers and 0 for vacant_jobs
    jobs.update_col_from_series('building_id', combine_indexes.building_id, cast = True)
    jobs.update_col_from_series('vacant_jobs', combine_indexes.vacant_jobs, cast = True)
    logger.info("Number of unplaced home-based jobs: %s" % len(jobs.local[(jobs.local.home_based_status==1) 
                              & (jobs.local.vacant_jobs > 0) & (jobs.building_id > 0)]))

  
@orca.step('wplcm_simulate')
def wplcm_simulate(persons, households, jobs):
    #jobs.index.name = 'job_id'
    res = psrc_dcm.lcm_simulate("wplcmcoef.yaml", persons, jobs,
                             0, None, "job_id", "number_of_jobs",
                             "vacant_jobs", cast=True)

