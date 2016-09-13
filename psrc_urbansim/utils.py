import pandas as pd
import orca
from urbansim.utils import misc
from urbansim_defaults.utils import yaml_to_class, to_frame
import os

def change_store(store_name):
    orca.add_injectable("store",
                       pd.HDFStore(os.path.join(misc.data_dir(),
                                                store_name), mode="r"))

def hedonic_simulate(cfg, tbl, join_tbls, out_fname, cast=True):
    """
    This differs from its original version in urbansim_defaults.utils in default casting 
    of the resulting attribute.
    
    Simulate the hedonic model for the specified table.

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
    """
    cfg = misc.config(cfg)
    df = to_frame(tbl, join_tbls, cfg)
    price_or_rent, _ = yaml_to_class(cfg).predict_from_cfg(df, cfg)
    tbl.update_col_from_series(out_fname, price_or_rent, cast=cast)