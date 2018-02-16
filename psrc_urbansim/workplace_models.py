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


@orca.step('wahcm_estimate')
def wahcm_estimate(persons_for_estimation, households_for_estimation, zones):
    return work_at_home_estimate("wahcm.yaml", persons_for_estimation, 'work_at_home', 
                                 [households_for_estimation, zones], 'wahcmcoeff.yaml')

    #cfg = misc.config(cfg)
    #coefficients = np.array([-0.000650279, -0.0195561, -0.235065, -0.0811701, -0.896509, -4.18865])
    #df = persons_for_estimation.to_frame()
    #df = df[df.is_worker==1]
    #df.age13 = np.where(df.age13>0,1,0)
    #df['parttime'] = np.where(df.employment_status == 2, 1, 0)
    #df['constant'] = 1
    #train_cols = ['kemp30m', 'age', 'age13', 'edu', 'parttime', 'constant']
    #logit = sm.Logit(df['work_at_home'], df[train_cols])
    #result = logit.fit()
    #result_df = pd.DataFrame(result.predict(), columns=['probability'])
    #result_df['random'] = np.random.random(len(result_df))

## WAHCM
#@orca.step('wahcm_estimate')
#def wahcm_simulate(persons_for_estimation, jobs, parcels, zones):
#    coefficients = np.array([-0.000650279, -0.0195561, -0.235065, -0.0811701, -0.896509, -4.18865])
#    df = persons_for_estimation.to_frame()
#    df = df[df.is_worker==1]
#    df.age13 = np.where(df.age13>0,1,0)
#    df['parttime'] = np.where(df.employment_status == 2, 1, 0)
#    df['constant'] = 1
#    train_cols = ['kemp30m', 'age', 'age13', 'edu', 'parttime', 'constant']
#    logit = sm.Logit(df['work_at_home'], df[train_cols])
#    result = logit.fit()
#    result_df = pd.DataFrame(result.predict(), columns=['probability'])
#    result_df['random'] = np.random.random(len(result_df))
    

@orca.step('wahcm_simulate')
def wahcm_simulate(persons, jobs, parcels, zones):
    coefficients = np.array([-0.000650279, -0.0195561, -0.235065, -0.0811701, -0.896509, -4.18865])
    df = persons.to_frame()
    df.age13 = np.where(df.age13>0,1,0)
    df['parttime'] = np.where(df.employment_status == 2, 1, 0)
    df['constant'] = 1
    cols = ['kemp30m', 'age', 'age13', 'edu', 'parttime', 'constant']
    df = df[cols]
    binary_choice = choicemodels.Logit(df)
    binary_choice.predict(coefficients)
    return utils.lcm_simulate("wahcmcoef.yaml", persons, "job_id",
                              jobs, None, out_cfg="wplcmcoef.yaml")

@orca.step('wplcm_simulate')
def wplcm_simulate(persons, jobs):
    return utils.lcm_simulate("wplcmcoef.yaml", persons, jobs, None,
                              "job_id", "number_of_jobs", "vacant_jobs", cast=True)