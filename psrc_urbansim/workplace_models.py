import os 
import sys
import orca
import random
import urbansim_defaults.utils as utils
import psrc_urbansim.utils as psrcutils
import datasources
import variables
import numpy as np
import pandas as pd
from psrc_urbansim.mod.allocation import AgentAllocationModel
import urbansim.developer as dev
import developer_models as psrcdev
import os 
from urbansim.utils import misc
from choicemodels import choicemodels
import statsmodels.api as sm
from statsmodels.formula.api import logit, probit, poisson, ols
import random
from urbansim.utils import misc
from psrc_urbansim.binary_discrete_choice import BinaryDiscreteChoiceModel


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
    # persons/households have been modified, need to update all columns
    persons.clear_cached()
    households.clear_cached()
    work_at_home_prob = work_at_home_simulate("wahcmcoeff.yaml", persons, 
                                 [households, zones])[1]
    jobs_df = jobs.to_frame()
    persons_df =  persons.to_frame()
    home_based_jobs = jobs_df[(jobs_df.home_based_status == 1) & (jobs_df.vacant_jobs>0)]

    # sample home workers using the exact number of vacant home based jobs, weighted by the probablities from the wachm:
    home_workers = work_at_home_prob.sample(len(home_based_jobs), weights = work_at_home_prob.values)

    # update job_id on the persons table
    # should not matter which person gets which home-based job
    combine_indexes = pd.DataFrame([home_workers.index, home_based_jobs.index]).transpose()
    combine_indexes.columns = ['person_id', 'job_id']
    combine_indexes.set_index('person_id', inplace=True)
    combine_indexes['work_at_home'] = 1
    
    # updates job_id, work_at_home on the persons table where index (person_id) matches in combine_indexes
    persons_df.update(combine_indexes, overwrite=True)
    
    # building_id on jobs table for home based workers should be the household building_id of the person assigned the job
    # get building_id:
    combine_indexes['household_building_id'] = 0
    combine_indexes.update(persons_df, overwrite=True)
    
    #update building_id on jobs table:
    combine_indexes.reset_index(level = None, inplace = True)
    combine_indexes.set_index('job_id', inplace=True)
    combine_indexes['building_id'] = combine_indexes.household_building_id
    # drop all columns except building_id
    combine_indexes = combine_indexes[['building_id']]
    # update job capacity to reflect these jobs are taken
    combine_indexes['vacant_jobs'] = 0
    # update jobs table- building_id of at home workers and 0 for vacant_jobs
    jobs_df.update(combine_indexes, overwrite=True)

    orca.add_table('jobs', jobs_df, cache = True)
    orca.add_table('persons', persons_df, cache = True)
  
@orca.step('wplcm_simulate')
def wplcm_simulate(persons, households, jobs):
    # can only send in jobs that have a valid building_id, so remove unlocated jobs for now
    jobs_df = jobs.to_frame()
    located_jobs_df = jobs_df[jobs_df.building_id>0]
    orca.add_table('jobs', located_jobs_df)
    jobs = orca.get_table('jobs')
    res = utils.lcm_simulate("wplcmcoef.yaml", persons, jobs, None,
                              "job_id", "number_of_jobs", "vacant_jobs", cast=True)
    # add all jobs back to the jobs table
    orca.add_table('jobs', jobs_df, cache = True)
