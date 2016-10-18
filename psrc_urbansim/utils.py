import pandas as pd
import orca
import numpy as np
from urbansim.utils import misc
from urbansim_defaults.utils import yaml_to_class, to_frame, check_nas, _print_number_unplaced
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
    
    
def lcm_simulate(cfg, choosers, buildings, join_tbls, out_fname,
                 supply_fname, vacant_fname,
                 enable_supply_correction=None):
    """
    This differs from its original version in urbansim_defaults.utils in default casting 
    of the resulting attribute, in line:
    choosers.update_col_from_series(out_fname, new_buildings, cast=TRUE)   
    
    Simulate the location choices for the specified choosers

    Parameters
    ----------
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
    """
    cfg = misc.config(cfg)

    choosers_df = to_frame(choosers, [], cfg, additional_columns=[out_fname])

    additional_columns = [supply_fname, vacant_fname]
    if enable_supply_correction is not None and \
       "submarket_col" in enable_supply_correction:
        additional_columns += [enable_supply_correction["submarket_col"]]
    if enable_supply_correction is not None and \
       "price_col" in enable_supply_correction:
        additional_columns += [enable_supply_correction["price_col"]]
    locations_df = to_frame(buildings, join_tbls, cfg,
                            additional_columns=additional_columns)

    available_units = buildings[supply_fname]
    vacant_units = buildings[vacant_fname]

    print "There are %d total available units" % available_units.sum()
    print "    and %d total choosers" % len(choosers)
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

    movers = choosers_df[choosers_df[out_fname] == -1]
    print "There are %d total movers for this LCM" % len(movers)

    if enable_supply_correction is not None:
        assert isinstance(enable_supply_correction, dict)
        assert "price_col" in enable_supply_correction
        price_col = enable_supply_correction["price_col"]
        assert "submarket_col" in enable_supply_correction
        submarket_col = enable_supply_correction["submarket_col"]

        lcm = yaml_to_class(cfg).from_yaml(str_or_buffer=cfg)

        if enable_supply_correction.get("warm_start", False) is True:
            raise NotImplementedError()

        multiplier_func = enable_supply_correction.get("multiplier_func", None)
        if multiplier_func is not None:
            multiplier_func = orca.get_injectable(multiplier_func)

        kwargs = enable_supply_correction.get('kwargs', {})
        new_prices, submarkets_ratios = supply_and_demand(
            lcm,
            movers,
            units,
            submarket_col,
            price_col,
            base_multiplier=None,
            multiplier_func=multiplier_func,
            **kwargs)

        # we will only get back new prices for those alternatives
        # that pass the filter - might need to specify the table in
        # order to get the complete index of possible submarkets
        submarket_table = enable_supply_correction.get("submarket_table", None)
        if submarket_table is not None:
            submarkets_ratios = submarkets_ratios.reindex(
                orca.get_table(submarket_table).index).fillna(1)
            # write final shifters to the submarket_table for use in debugging
            orca.get_table(submarket_table)["price_shifters"] = submarkets_ratios

        print "Running supply and demand"
        print "Simulated Prices"
        print buildings[price_col].describe()
        print "Submarket Price Shifters"
        print submarkets_ratios.describe()
        # we want new prices on the buildings, not on the units, so apply
        # shifters directly to buildings and ignore unit prices
        orca.add_column(buildings.name,
                        price_col+"_hedonic", buildings[price_col])
        new_prices = buildings[price_col] * \
            submarkets_ratios.loc[buildings[submarket_col]].values
        buildings.update_col_from_series(price_col, new_prices)
        print "Adjusted Prices"
        print buildings[price_col].describe()

    if len(movers) > vacant_units.sum():
        print "WARNING: Not enough locations for movers"
        print "    reducing locations to size of movers for performance gain"
        movers = movers.head(int(vacant_units.sum()))

    new_units, _ = yaml_to_class(cfg).predict_from_cfg(movers, units, cfg)

    # new_units returns nans when there aren't enough units,
    # get rid of them and they'll stay as -1s
    new_units = new_units.dropna()

    # go from units back to buildings
    new_buildings = pd.Series(units.loc[new_units.values][out_fname].values,
                              index=new_units.index)

    choosers.update_col_from_series(out_fname, new_buildings, cast=True)
    _print_number_unplaced(choosers, out_fname)

    if enable_supply_correction is not None:
        new_prices = buildings[price_col]
        if "clip_final_price_low" in enable_supply_correction:
            new_prices = new_prices.clip(lower=enable_supply_correction[
                "clip_final_price_low"])
        if "clip_final_price_high" in enable_supply_correction:
            new_prices = new_prices.clip(upper=enable_supply_correction[
                "clip_final_price_high"])
        buildings.update_col_from_series(price_col, new_prices)

    vacant_units = buildings[vacant_fname]
    print "    and there are now %d empty units" % vacant_units.sum()
    print "    and %d overfull buildings" % len(vacant_units[vacant_units < 0])
                                                             
                                                             
def simple_relocation(choosers, relocation_rate, fieldname, cast=True):
    """
    This differs from its original version in urbansim_defaults.utils in default casting 
    of the resulting attribute (added argument)
    
    Run a simple rate based relocation model

    Parameters
    ----------
    tbl : DataFrameWrapper or DataFrame
        Table of agents that might relocate
    rate : float
        Rate of relocation
    location_fname : str
        The field name in the resulting dataframe to set to -1 (to unplace
        new agents)

    Returns
    -------
    Nothing
    """
    print "Total agents: %d" % len(choosers)
    _print_number_unplaced(choosers, fieldname)

    print "Assigning for relocation..."
    chooser_ids = np.random.choice(choosers.index, size=int(relocation_rate *
                                   len(choosers)), replace=False)
    choosers.update_col_from_series(fieldname,
                                    pd.Series(-1, index=chooser_ids), cast=cast)

    _print_number_unplaced(choosers, fieldname)