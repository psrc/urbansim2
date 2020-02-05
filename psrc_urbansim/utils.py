import pandas as pd
import orca
import numpy as np
from urbansim.utils import misc, yamlio
from urbansim_defaults.utils import to_frame, yaml_to_class, check_nas, _print_number_unplaced
from urbansim.models.regression import YTRANSFORM_MAPPING
from urbansim.models import util
import os
from dcm_weighted_sampling import PSRC_SegmentedMNLDiscreteChoiceModel, MNLDiscreteChoiceModelWeightedSamples, resim_overfull_buildings

def change_store(store_name):
    orca.add_injectable("store",
                       pd.HDFStore(os.path.join(misc.data_dir(),
                                                store_name), mode="r"))

def reduce_df_size(df):
    df_float = df.select_dtypes(include=['float'])
    for col in df_float.columns:
        df[col] = pd.to_numeric(df_float[col], downcast='float')
    return df

def deep_merge(source, destination):
    """
    Recursive merge of dictionaries.
    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(b, a) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            destination[key] = value
    return

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

def _update_prediction_sample_size(cls, sample_size):
    cls.prediction_sample_size = sample_size
    for _, m in cls._group.models.items():
        m.prediction_sample_size = sample_size
    
def lcm_simulate_CY(subreg_geo_id, cfg, choosers, buildings, join_tbls, out_fname,
                 supply_fname, vacant_fname, min_overfull_buildings=0,
                 settings = {}, cast=False,
                 alternative_ratio=2.0):
    """
    Simulate the location choices for the specified choosers for each subregion separately

    Parameters
    ----------
    subreg_geo_id: string
        Name of the subregion's identifier.
    cfg : string
        The name of the yaml config file from which to read the location
        choice model
    choosers : DataFrameWrapper
        A dataframe of agents doing the choosing
    buildings : DataFrameWrapper
        A dataframe of buildings which the choosers are locating in and which
        have a supply
    join_tbls : list of strings
        A list of land use dataframes to give neighborhood info around the
        buildings - will be joined to the buildings using existing broadcasts.
    out_fname : string
        The column name to write the simulated location to
    supply_fname : string
        The string in the buildings table that indicates the amount of
        available units there are for choosers, vacant or not
    vacant_fname : string
        The string in the buildings table that indicates the amount of vacant
        units there will be for choosers
    enable_supply_correction : Python dict
        Should contain keys "price_col" and "submarket_col" which are set to
        the column names in buildings which contain the column for prices and
        an identifier which segments buildings into submarkets
    cast : boolean
        Should the output be cast to match the existing column.
    alternative_ratio : float, optional
        Value to override the setting in urbansim.models.dcm.predict_from_cfg.
        Above this ratio of alternatives to choosers (default of 2.0), the
        alternatives will be sampled to improve computational performance
    """
    import dcm_weighted_sampling as dcmsampl
    
    cfg = misc.config(cfg)

    choosers_df = to_frame(choosers, [], cfg, additional_columns=[out_fname, subreg_geo_id])

    additional_columns = [supply_fname, vacant_fname, subreg_geo_id]
    locations_df = to_frame(buildings, join_tbls, cfg,
                            additional_columns=additional_columns)

    available_units = buildings[supply_fname]
    vacant_units = buildings[vacant_fname]

    all_movers = choosers_df[(choosers_df[out_fname] == -1)]
    print "There are %d total available units" % available_units.sum()
    print "    and %d total choosers from which %d are movers" % (len(choosers), len(all_movers))
    print "    but there are %d overfull buildings" % \
          len(vacant_units[vacant_units < 0])

    vacant_units = vacant_units[vacant_units > 0]

    # sometimes there are vacant units for buildings that are not in the
    # locations_df, which happens for reasons explained in the warning below
    indexes = np.repeat(vacant_units.index.values,
                        vacant_units.values.astype('int'))
    isin = pd.Series(indexes).isin(locations_df.index)
    missing = len(isin[isin == False])
    indexes = indexes[isin.values]
    units = locations_df.loc[indexes].reset_index()
    check_nas(units)

    print "    for a total of %d temporarily empty units" % vacant_units.sum()
    print "    in %d buildings total in the region" % len(vacant_units)

    if missing > 0:
        print "WARNING: %d indexes aren't found in the locations df -" % \
            missing
        print "    this is usually because of a few records that don't join "
        print "    correctly between the locations df and the aggregations tables"

    subregs = np.unique(all_movers[subreg_geo_id])
    lcm = dcmsampl.yaml_to_class(cfg).from_yaml(str_or_buffer=cfg)
    
    # modify sample size if needed
    lcm.prediction_sample_size = settings.get("prediction_sample_size", lcm.prediction_sample_size)
    orig_sample_size = lcm.prediction_sample_size
    
    dcm_weighted = MNLDiscreteChoiceModelWeightedSamples(None, lcm, None)
    
    # run LCM for each subregion
    for subreg in subregs:
        this_filter = np.logical_and(choosers_df[out_fname] == -1, choosers_df[subreg_geo_id] == subreg)
        movers = choosers_df[this_filter]
        this_sreg_units = units[units[subreg_geo_id] == subreg]
        # need to filter alternatives now in order to modify the sample size if needed
        this_sreg_units = util.apply_filter_query(this_sreg_units, lcm.alts_predict_filters)  
        print("\nSubregion {}".format(subreg))
        print("-------------")
        print "There are %d total movers and %d alternatives for this subregion" % (len(movers), len(this_sreg_units))        
        
        if len(movers) == 0 or len(this_sreg_units) == 0:
            print "Skipping LCM"
            continue

        # adjust sampling size if too few alternatives
        if len(this_sreg_units) < orig_sample_size:
            _update_prediction_sample_size(dcm_weighted.model, len(this_sreg_units))
        else:
            _update_prediction_sample_size(dcm_weighted.model, orig_sample_size)
            
        
        print "Sampling", dcm_weighted.model.prediction_sample_size, "alternatives"
        # predict
        new_units, probabilities = dcm_weighted.predict_with_resim(movers, this_sreg_units)
        print("Assigned %d choosers to new units" % len(new_units.dropna()))        

        # new_units returns nans when there aren't enough units,
        # get rid of them and they'll stay as -1s
        #new_units = new_units.dropna()

        # go from units back to buildings
        new_buildings = pd.Series(this_sreg_units.loc[new_units.values][out_fname].values,
                              index=new_units.index)
 
        choosers.update_col_from_series(out_fname, new_buildings, cast=cast)
        _print_number_unplaced(choosers, out_fname)
        
        resim_overfull_buildings(buildings, vacant_fname, choosers, out_fname, min_overfull_buildings, new_buildings, probabilities, 
                                 new_units, this_sreg_units, 
                                 choosers_filter = this_filter, location_filter = buildings.index.isin(this_sreg_units[out_fname]),
                                 niterations = 10, cast = cast)

    vacant_units = buildings[vacant_fname]
    print "    and there are now %d empty units" % vacant_units.sum()
    print "    and %d overfull buildings" % len(vacant_units[vacant_units < 0])


def psrc_to_frame(tbl, join_tbls, cfg, additional_columns=[], check_na = True):
    """
    Like the original code in urbansim_defaults.utils, but checks for nas only if required.
    
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
    cfg = yaml_to_class(cfg).from_yaml(str_or_buffer=cfg)
    tables = [t for t in tables if t is not None]
    columns = misc.column_list(tables, cfg.columns_used()) + additional_columns
    if len(tables) > 1:
        df = orca.merge_tables(target=tables[0].name,
                               tables=tables, columns=columns)
    else:
        df = tables[0].to_frame(columns)
    if check_na:
        check_nas(df)
    return df

