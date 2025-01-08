import pandas as pd
import orca
import numpy as np
import logging
from urbansim.utils import misc, yamlio
from urbansim_defaults.utils import to_frame, yaml_to_class, check_nas
from urbansim.models.regression import YTRANSFORM_MAPPING
from urbansim.models import util, transition
import os
from psrc_urbansim.dcm_weighted_sampling import PSRC_SegmentedMNLDiscreteChoiceModel, MNLDiscreteChoiceModelWeightedSamples, resim_overfull_buildings

logger = logging.getLogger(__name__)

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
            logger.info("Computing residuals")
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
                logger.warning("Residual column not available.")
            else:
                price_or_rent = price_or_rent + tbl[residual_name].loc[price_or_rent.index]
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
    import psrc_urbansim.dcm_weighted_sampling as dcmsampl
    
    cfg = misc.config(cfg)

    choosers_df = to_frame(choosers, [], cfg, additional_columns=[out_fname, subreg_geo_id])

    additional_columns = [supply_fname, vacant_fname, subreg_geo_id]
    locations_df = to_frame(buildings, join_tbls, cfg,
                            additional_columns=additional_columns)

    available_units = buildings[supply_fname]
    vacant_units = buildings[vacant_fname]

    all_movers = choosers_df[(choosers_df[out_fname] == -1)]
    logger.info("There are {} total available units".format(available_units.sum()))
    logger.info("    and {} total choosers from which {} are movers".format(len(choosers), len(all_movers)))
    logger.info("    but there are {} overfull buildings".format(len(vacant_units[vacant_units < 0])))

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

    logger.info("    for a total of {} temporarily empty units".format(vacant_units.sum()))
    logger.info("    in {} buildings total in the region".format(len(vacant_units)))

    if missing > 0:
        logger.warning("{} indexes aren't found in the locations df -".format(missing))
        logger.info("    this is usually because of a few records that don't join ")
        logger.info("    correctly between the locations df and the aggregations tables")

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
        logger.info("\n---- Subregion {} ----".format(subreg))
        logger.info("There are {} total movers and {} alternatives for this subregion".format(len(movers), len(this_sreg_units)))  
        
        if len(movers) == 0 or len(this_sreg_units) == 0:
            logger.info("Skipping LCM")
            continue

        # adjust sampling size if too few alternatives
        if len(this_sreg_units) < orig_sample_size:
            _update_prediction_sample_size(dcm_weighted.model, len(this_sreg_units))
        else:
            _update_prediction_sample_size(dcm_weighted.model, orig_sample_size)
            
        
        logger.info("Sampling {} alternatives".format(dcm_weighted.model.prediction_sample_size))
        # predict
        new_units, probabilities = dcm_weighted.predict_with_resim(movers, this_sreg_units)
        logger.info("Assigned {} choosers to new units".format(len(new_units.dropna())))       

        # new_units returns nans when there aren't enough units,
        # get rid of them and they'll stay as -1s
        #new_units = new_units.dropna()

        # go from units back to buildings
        new_buildings = pd.Series(this_sreg_units.loc[new_units.values][out_fname].values,
                              index=new_units.index)
 
        choosers.update_col_from_series(out_fname, new_buildings, cast=cast)
        _log_number_unplaced(choosers, out_fname)
        
        resim_overfull_buildings(buildings, vacant_fname, choosers, out_fname, min_overfull_buildings, new_buildings, probabilities, 
                                 new_units, this_sreg_units, 
                                 choosers_filter = this_filter, location_filter = buildings.index.isin(this_sreg_units[out_fname]),
                                 niterations = 10, cast = cast)

    vacant_units = buildings[vacant_fname]
    logger.info("    and there are now {} empty units".format(vacant_units.sum()))
    logger.info("    and {} overfull buildings".format(len(vacant_units[vacant_units < 0])))


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

def _remove_developed_buildings(old_buildings, new_buildings, unplace_agents):
    # this is a copy from urbansim_defaults.utils, with prints changed to logger.info
    redev_buildings = old_buildings.parcel_id.isin(new_buildings.parcel_id)
    if "not_demolish" in old_buildings.columns:
        redev_buildings = redev_buildings & (old_buildings.not_demolish == False)
    l = len(old_buildings)
    drop_buildings = old_buildings[redev_buildings]

    if "dropped_buildings" in orca.orca._TABLES:
        prev_drops = orca.get_table("dropped_buildings").to_frame()
        orca.add_table("dropped_buildings", pd.concat([drop_buildings, prev_drops]))
    else:
        orca.add_table("dropped_buildings", drop_buildings)

    old_buildings = old_buildings[np.logical_not(redev_buildings)]
    l2 = len(old_buildings)
    if l-l2 > 0:
        logger.info("Dropped {} buildings because they were redeveloped".format(l-l2))

    for tbl in unplace_agents:
        agents = orca.get_table(tbl).local
        displaced_agents = agents.building_id.isin(drop_buildings.index)
        logger.info("Unplaced {} before: {}".format(tbl, len(agents.query(
                                              "building_id == -1"))))
        agents.building_id[displaced_agents] = -1
        logger.info("Unplaced {} after: {}".format(tbl, len(agents.query(
                                             "building_id == -1"))))

    return old_buildings

def _log_number_unplaced(df, fieldname):
    logger.info("Total currently unplaced: %d" % df[fieldname].value_counts().get(-1, 0))
    
    
    
def full_transition(agents, agent_controls, year, settings, location_fname, linked_tables=None):
    """
    Run a transition model based on control totals specified in the usual
    UrbanSim way
    Passes sampling_threshold and sampling_hierarchy to the model version from Scott Bridwell

    Parameters
    ----------
    agents : DataFrameWrapper
        Table to be transitioned
    agent_controls : DataFrameWrapper
        Table of control totals
    year : int
        The year, which will index into the controls
    settings : dict
        Contains the configuration for the transition model - is specified
        down to the yaml level with a "total_column" which specifies the
        control total and an "add_columns" param which specified which
        columns to add when calling to_frame (should be a list of the columns
        needed to do the transition
    location_fname : str
        The field name in the resulting dataframe to set to -1 (to unplace
        new agents)
    linked_tables : dict of tuple, optional
        Dictionary of table_name: (table, 'column name') pairs. The column name
        should match the index of `agents`. Indexes in `agents` that
        are copied or removed will also be copied and removed in
        linked tables.

    Returns
    -------
    Nothing
    """
    
    sampling_threshold = settings.get("sampling_threshold", None)
    sampling_hierarchy = settings.get("sampling_hierarchy", [])
    if sampling_threshold is not None: # need to compute the various variables for the control dataset
        for col in sampling_hierarchy:
            agent_controls[col]  # to trigger computation
    ct = agent_controls.to_frame()
    hh = agents.to_frame(agents.local_columns +
                         settings.get('add_columns', []))
    print("Total agents before transition: {}".format(len(hh)))
    linked_tables = linked_tables or {}
    for table_name, (table, col) in linked_tables.items():
        print("Total %s before transition: %s" % (table_name, len(table)))
  
    tran = transition.TabularTotalsTransition(ct, settings['total_column'],
                                              sampling_threshold = sampling_threshold,
                                              sampling_hierarchy = sampling_hierarchy)
    model = transition.TransitionModel(tran)
    new, added_hh_idx, new_linked = model.transition(hh, year, linked_tables=linked_tables)
    new.loc[added_hh_idx, location_fname] = -1
    print("Total agents after transition: {}".format(len(new)))
    orca.add_table(agents.name, new)
    for table_name, table in new_linked.items():
        print("Total %s after transition: %s" % (table_name, len(table)))
        orca.add_table(table_name, table)
