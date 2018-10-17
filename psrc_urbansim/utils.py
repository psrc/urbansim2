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


def run_developer(forms, agents, buildings, supply_fname, feasibility,
                  parcel_size, ave_unit_size, current_units, cfg, year=None,
                  target_vacancy=0.1, form_to_btype_callback=None,
                  add_more_columns_callback=None,
                  remove_developed_buildings=True,
                  unplace_agents=['households', 'jobs'],
                  num_units_to_build=None, profit_to_prob_func=None,
                  custom_selection_func=None, pipeline=False):
    """
    Run the developer model to pick and build buildings

    Parameters
    ----------
    forms : string or list of strings
        Passed directly dev.pick
    agents : DataFrame Wrapper
        Used to compute the current demand for units/floorspace in the area
    buildings : DataFrame Wrapper
        Used to compute the current supply of units/floorspace in the area
    supply_fname : string
        Identifies the column in buildings which indicates the supply of
        units/floorspace
    feasibility : DataFrame Wrapper
        The output from feasibility above (the table called 'feasibility')
    parcel_size : series
        The size of the parcels.  This was passed to feasibility as well,
        but should be passed here as well.  Index should be parcel_ids.
    ave_unit_size : series
        The average residential unit size around each parcel - this is
        indexed by parcel, but is usually a disaggregated version of a
        zonal or accessibility aggregation.
    current_units : series
        The current number of units on the parcel.  Is used to compute the
        net number of units produced by the developer model.  Many times
        the developer model is redeveloping units (demolishing them) and
        is trying to meet a total number of net units produced.
    cfg : str
        The name of the yaml file to read pro forma configurations from
    year : int
        The year of the simulation - will be assigned to 'year_built' on the
        new buildings
    target_vacancy : float
        The target vacancy rate - used to determine how much to build
    form_to_btype_callback : function
        Will be used to convert the 'forms' in the pro forma to
        'building_type_id' in the larger model
    add_more_columns_callback : function
        Takes a dataframe and returns a dataframe - is used to make custom
        modifications to the new buildings that get added
    remove_developed_buildings : optional, boolean (default True)
        Remove all buildings on the parcels which are being developed on
    unplace_agents : optional, list of strings (default ['households', 'jobs'])
        For all tables in the list, will look for field building_id and set
        it to -1 for buildings which are removed - only executed if
        remove_developed_buildings is true
    num_units_to_build : optional, int
        If num_units_to_build is passed, build this many units rather than
        computing it internally by using the length of agents adn the sum of
        the relevant supply columin - this trusts the caller to know how to
        compute this.
    profit_to_prob_func : func
        Passed directly to dev.pick
    custom_selection_func : func
        User passed function that decides how to select buildings for
        development after probabilities are calculated. Must have
        parameters (self, df, p) and return a numpy array of buildings to
        build (i.e. df.index.values)
    pipeline : bool
        Passed to add_buildings


    Returns
    -------
    Writes the result back to the buildings table and returns the new
    buildings with available debugging information on each new building
    """
    cfg = misc.config(cfg)

    target_units = (num_units_to_build
                    or compute_units_to_build(len(agents),
                                              buildings[supply_fname].sum(),
                                              target_vacancy))

    dev = develop.Developer.from_yaml(feasibility.to_frame(), forms,
                                      target_units, parcel_size,
                                      ave_unit_size, current_units,
                                      year, str_or_buffer=cfg)

    print("{:,} feasible buildings before running developer".format(
        len(dev.feasibility)))

    new_buildings = dev.pick(profit_to_prob_func, custom_selection_func)
    orca.add_table("feasibility", dev.feasibility)

    if new_buildings is None or len(new_buildings) == 0:
        return

    new_buildings = add_buildings(dev.feasibility, buildings, new_buildings,
                                  form_to_btype_callback,
                                  add_more_columns_callback,
                                  supply_fname, remove_developed_buildings,
                                  unplace_agents, pipeline)

    # This is a change from previous behavior, which return the newly merged
    # "all buildings" table
    return new_buildings
