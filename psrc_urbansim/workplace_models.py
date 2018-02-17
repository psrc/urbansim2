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

    

@orca.step('wahcm_simulate')
def wahcm_simulate(persons, households, parcels, zones):
    
    #this part is handeld by the to_frame function above:
    columns = misc.column_list([persons, households, zones], 
                               ['age', 'edu', 'employment_status', 'persons_under_13', 
                                'jobs_within_30_min_tt_hbw_am_drive_alone'])
    df = orca.merge_tables(persons,tables=[persons,households,zones], columns=columns)
    #handled by the prediction filter in the yaml:
    df = df[df.employment_status>0]

    #boolean variables should be specified in .yaml and handeled by patsy, so below will not be needed
    # example: I(1*(employment_status ==2))  
    df['parttime'] = np.where(df.employment_status == 2, 1, 0)
    df['persons_under_13'] = np.where(df.persons_under_13 > 0, 1, 0)
    
    # constant is handled by adding +1 to end of model expression in yaml, so below will not be needed
    df['constant'] = 1
    
    # This is in the .yaml file, handled by the BinaryDiscreteChoiceModel code
    coefficients = np.array([0.3742693230223763, 0.2143111891006935, 0.03620835983681646, -4.06024070020186, 0.023217851964759246, 3.662406622009484e-07])
    cols = ['parttime', 'persons_under_13', 'age', 'constant', 'edu', 'jobs_within_30_min_tt_hbw_am_drive_alone']
    
    # Constructor requires and observation column, but since we are not estimating any will do so using constant. 
    logit = choicemodels.Logit(df['constant'], df[cols])

    # Get the prediction probabilities for each worker
    result_df = pd.DataFrame(logit.predict(coefficients), columns=['probability'])

    # Monte carlo:
    result_df['mc'] = np.random.random(len(result_df))
  
    # Person works at home if probibility > random number. 
    result_df['work_at_home'] = np.where(result_df.probability > result_df.mc, 1, 0)
    
    # Reindex to persons table
    result_df = result_df.reindex(df.index)
    work_at_home_col = result_df.work_at_home.reindex(persons.index).fillna(0)

    #Add to persons table
    orca.add_column(persons, 'work_at_home', work_at_home_col) 
 
  
@orca.step('wplcm_simulate')
def wplcm_simulate(persons, jobs):
    return utils.lcm_simulate("wplcmcoef.yaml", persons, jobs, None,
                              "job_id", "number_of_jobs", "vacant_jobs", cast=True)