import pandas as pd
import orca
import numpy as np
from urbansim.utils import misc, yamlio
from urbansim_defaults.utils import to_frame, yaml_to_class
from urbansim.models.regression import YTRANSFORM_MAPPING
import os

def change_store(store_name):
    orca.add_injectable("store",
                       pd.HDFStore(os.path.join(misc.data_dir(),
                                                store_name), mode="r"))

def reduce_df_size(df):
    df_float = df.select_dtypes(include=['float'])
    for col in df_float.columns:
        df[col] = pd.to_numeric(df_float[col], downcast='float')
    return df

def hedonic_simulate(cfg, tbl, join_tbls, out_fname, cast=False, 
                     compute_residuals = False, residual_name = None, add_residuals = False,
                     settings = {}):
    """
    Simulate the hedonic model for the specified table

    Parameters
    ----------
    cfg : string
        The name of the yaml config file from which to read the hedonic model
    tbl : DataFrameWrapper
        A dataframe for which to estimate the hedonic
    join_tbls : list of strings
        A list of land use dataframes to give neighborhood info around the
        buildings - will be joined to the buildings using existing broadcasts
    out_fname : string
        The output field name (should be present in tbl) to which to write
        the resulting column to
    cast : boolean
        Should the output be cast to match the existing column.    
    """
    cfg = misc.config(cfg)
    df = to_frame(tbl, join_tbls, cfg)
    df = reduce_df_size(df)
    price_or_rent, _ = yaml_to_class(cfg).predict_from_cfg(df, cfg)

    #cfgdict = yamlio.yaml_to_dict(str_or_buffer=cfg)
    ytransform_back = YTRANSFORM_MAPPING[settings.get("ytransform_back", None)]
    
    if compute_residuals or add_residuals:
        if residual_name is None:
            residual_name = "_%s_residuals_" % out_fname    
        if compute_residuals:
            print "Computing residuals"
            orig_values = df[out_fname]
            ytransform_out = YTRANSFORM_MAPPING[settings.get("ytransform_out", None)]
            if ytransform_out is not None:
                orig_values = ytransform_out(orig_values)            
            residuals = orig_values - price_or_rent
            if(residual_name in tbl.columns):
                residuals = residuals[~residuals.isna()]
                tbl.update_col_from_series(residual_name, residuals)
            else:
                residuals[residuals.isna()] = 0
                tbl.update_col(residual_name, residuals)
        if add_residuals:
            if not residual_name in tbl.columns:
                print "WARNING: Residual column not available."
            else:
                price_or_rent = price_or_rent + tbl[residual_name].ix[price_or_rent.index]
    if ytransform_back is not None:
        price_or_rent = ytransform_back(price_or_rent)
    tbl.update_col_from_series(out_fname, price_or_rent, cast=cast)
