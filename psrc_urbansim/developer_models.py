import os
import numpy as np
import pandas as pd
import orca
from developer import develop
import developer.utils as devutils
from urbansim.utils import misc
from urbansim_defaults.utils import yaml_to_class, to_frame, check_nas, _print_number_unplaced, _remove_developed_buildings

#from urbansim_defaults.utils import apply_parcel_callbacks, lookup_by_form


@orca.injectable('proposal_selection', autocall=False)
def proposal_selection(self, df, p, targets):
    """
    Passed to custom_selection_func in Developer.pick().
    """
    chunksize = self.config.get("chunk_size", 100)
    pf = orca.get_injectable("pf_config")
    uses = pf.uses[(pf.residential_uses == self.residential).values]
    all_choice_idx = pd.Series([], dtype = "int32")
    orig_df = df.copy()
    # remove proposals for which target vacancy is already met
    proposals_to_remove = filter_by_vacancy(df, uses, targets)
    if proposals_to_remove.size > 0:
        df = df.drop(proposals_to_remove)
        p = p.reindex(df.index)
        p = p / p.sum()
        
    while len(df) > 0:        
        # sample by chunks
        choice_idx = pd.Series(weighted_random_choice_multiparcel_by_count(df, p, count = chunksize))
                    #targets['single_family_residential'] + targets['multi_family_residential'] + targets['condo_residential'])
        all_choice_idx = pd.concat([all_choice_idx, choice_idx])
        proposals_to_remove = pd.concat([pd.Series(filter_by_vacancy(orig_df, uses, targets, all_choice_idx)), choice_idx])
        proposals_to_remove = proposals_to_remove[proposals_to_remove.isin(df.index)]
        if proposals_to_remove.size > 0:
            df = df.drop(proposals_to_remove)
            p = p.reindex(df.index)
            p = p / p.sum()
            
    return all_choice_idx


def filter_by_vacancy(df, uses, targets, choice_idx = None):
    fdf = orca.get_injectable("pf_config").forms_df
    vacancy_met = pd.Series([], dtype = "int32")
    for use in uses:
        btdistr = fdf[use][df.form]
        if btdistr.sum() == 0:
            continue
        btdistr.index = df.index 
        if (choice_idx is None and targets[use] == 0) or (choice_idx is not None and target_vacancy_met(df.loc[choice_idx], targets[use], fdf[use])):
            vacancy_met = pd.concat([vacancy_met, btdistr.index[btdistr > 0].to_series()])
    return vacancy_met.unique()
        
def target_vacancy_met(choices, target, forms_df):    
    btdistr = forms_df[choices.form]
    btdistr.index = choices.index
    units = btdistr*choices.net_units
    return units.sum() >= target   

    
def compute_target_units(vacancy_rate):
    pf = orca.get_injectable("pf_config")
    pfbt = pd.DataFrame({"use": pf.uses}, index=pf.residential_uses.index)
    vac = pd.concat((pfbt, vacancy_rate.local, pf.residential_uses), axis=1)
    bld = orca.get_table("buildings")
    agents_attr = {0: "number_of_jobs", 1: "number_of_households"}
    units_attr = {0: "job_spaces", 1: "residential_units"}
    target_units = {}
    for bt in vac.index:
        agentattr = agents_attr[vac.loc[bt].is_residential]
        unitattr = units_attr[vac.loc[bt].is_residential]
        is_builting_type = bld["building_type_id"] == bt
        number_of_agents = (bld[agentattr] * is_builting_type).sum()
        existing_units =  (bld[unitattr] * is_builting_type).sum()
        target_units[vac.loc[bt].use] = int(max(
            (number_of_agents / (1 - vac.loc[bt].target_vacancy_rate) - existing_units), 0))
    return target_units
    
def run_developer(forms, agents, buildings, supply_fname, feasibility,
                  parcel_size, ave_unit_size, current_units, cfg, year=None,
                  target_vacancy=0.1, form_to_btype_callback=None,
                  add_more_columns_callback=None,
                  remove_developed_buildings=True,
                  unplace_agents=['households', 'jobs'],
                  num_units_to_build=None, profit_to_prob_func=None,
                  custom_selection_func=None, pipeline=False, keep_suboptimal=True):
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
    keep_suboptimal: optional, int
        Whether or not to retain all proposals in the feasibility table
        instead of dropping sub-optimal forms and proposals.


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
    dev = develop.Developer.from_yaml(
    #dev = PSRCDeveloper.from_yaml(
                                      feasibility.to_frame(), forms,
                                      target_units, parcel_size,
                                      ave_unit_size, current_units,
                                      year, str_or_buffer=cfg, 
                                      keep_suboptimal=keep_suboptimal)
    # keep developer config
    dev.config = devutils.yaml_to_dict(yaml_str = None, str_or_buffer=cfg)
    
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

def add_buildings(feasibility, buildings, new_buildings,
                  form_to_btype_callback, add_more_columns_callback,
                  supply_fname, remove_developed_buildings, unplace_agents,
                  pipeline=False):
    """
    Parameters
    ----------
    feasibility : DataFrame
        Results from SqFtProForma lookup() method
    buildings : DataFrameWrapper
        Wrapper for current buildings table
    new_buildings : DataFrame
        DataFrame of selected buildings to build or add to pipeline
    form_to_btype_callback : func
        Callback function to assign forms to building types
    add_more_columns_callback : func
        Callback function to add columns to new_buildings table; this is
        useful for making sure new_buildings table has all required columns
        from the buildings table
    supply_fname : str
        Name of supply column for this type (e.g. units or job spaces)
    remove_developed_buildings : bool
        Remove all buildings on the parcels which are being developed on
    unplace_agents : list of strings
        For all tables in the list, will look for field building_id and set
        it to -1 for buildings which are removed - only executed if
        remove_developed_buildings is true
    pipeline : bool
        If True, will add new buildings to dev_sites table and pipeline rather
        than directly to buildings table

    Returns
    -------
    new_buildings : DataFrame
    """

    if form_to_btype_callback is not None:
        new_buildings["building_type_id"] = new_buildings.apply(
            form_to_btype_callback, axis=1)

    # This is where year_built gets assigned
    if add_more_columns_callback is not None:
        new_buildings = add_more_columns_callback(new_buildings)

    print("Adding {:,} buildings with {:,} {}".format(
        len(new_buildings),
        int(new_buildings[supply_fname].sum()),
        supply_fname))

    print("{:,} feasible buildings after running developer".format(
        len(feasibility)))

    building_columns = buildings.local_columns + ['construction_time']
    old_buildings = buildings.to_frame(building_columns)
    new_buildings = new_buildings[building_columns]

    if remove_developed_buildings:
        old_buildings = _remove_developed_buildings(
            old_buildings, new_buildings, unplace_agents)

    if pipeline:
        # Overwrite year_built
        current_year = orca.get_injectable('year')
        new_buildings['year_built'] = ((new_buildings.construction_time // 12)
                                       + current_year)
        new_buildings.drop('construction_time', axis=1, inplace=True)
        pl.add_sites_orca('pipeline', 'dev_sites', new_buildings, 'parcel_id')
    else:
        new_buildings.drop('construction_time', axis=1, inplace=True)
        all_buildings = merge_buildings(old_buildings, new_buildings)
        orca.add_table("buildings", all_buildings)

    return new_buildings

def merge_buildings(old_df, new_df, return_index=False):
    """
    Merge two dataframes of buildings.  The old dataframe is
    usually the buildings dataset and the new dataframe is a modified
    (by the user) version of what is returned by the pick method.

    Parameters
    ----------
    old_df : DataFrame
        Current set of buildings
    new_df : DataFrame
        New buildings to add, usually comes from this module
    return_index : bool
        If return_index is true, this method will return the new
        index of new_df (which changes in order to create a unique
        index after the merge)

    Returns
    -------
    df : DataFrame
        Combined DataFrame of buildings, makes sure indexes don't overlap
    index : pd.Index
        If and only if return_index is True, return the new index for the
        new_df DataFrame (which changes in order to create a unique index
        after the merge)
    """
    maxind = np.max(old_df.index.values)
    new_df = new_df.reset_index(drop=True)
    new_df.index = new_df.index + maxind + 1
    concat_df = pd.concat([old_df, new_df], verify_integrity=True)
    concat_df.index.name = 'building_id'

    if return_index:
        return concat_df, new_df.index

    return concat_df


def weighted_random_choice_by_count(df, p, target_units = None, count = None):
    """
    Proposal selection using weighted random choice. 
    It can be restricted by number of proposals to select.

    Parameters
    ----------
    df : DataFrame
        Proposals to select from
    p : Series
        Weights for each proposal
    target_units: int (optional)
        Number of units to build
    count: int (optional)
        Number of proposals to select. Only used if target_units is None.

    Returns
    -------
    build_idx : ndarray
        Index of buildings selected for development

    """
    # We don't know how many developments we will need, as they
    # differ in net_units. If all developments have net_units of 1
    # than we need target_units of them. So we choose the smaller
    # of available developments and target_units.
    if target_units is not None:
        num_to_sample = int(min(len(df.index), target_units))
    else:
        num_to_sample = count
    choices = np.random.choice(df.index.values,
                               size=num_to_sample,
                               replace=False, p=p)
    if target_units is None:
        return choices
    tot_units = df.net_units.loc[choices].values.cumsum()
    ind = int(np.searchsorted(tot_units, target_units,
                              side="left")) + 1
    return choices[:ind]


def weighted_random_choice_multiparcel_by_count(df, p, target_units = None, count = None):
    """
    Proposal selection using weighted random choice in the context of multiple
    proposals per parcel.

    Parameters
    ----------
    df : DataFrame
        Proposals to select from
    p : Series
        Weights for each proposal
    target_units: int
        Number of units to build

    Returns
    -------
    build_idx : ndarray
        Index of buildings selected for development

    """
    choice_idx = weighted_random_choice_by_count(df, p, target_units, count)
    choices = df.loc[choice_idx]
    new_target = None
    new_count = None
    while True:
        # If multiple proposals sampled for a given parcel, keep only one
        choice_counts = choices.parcel_id.value_counts()
        chosen_multiple = choice_counts[choice_counts > 1].index.values
        single_choices = choices[~choices.parcel_id.isin(chosen_multiple)]
        duplicate_choices = choices[choices.parcel_id.isin(chosen_multiple)]
        keep_choice = duplicate_choices.parcel_id.drop_duplicates(keep='first')
        dup_choices_to_keep = duplicate_choices.loc[keep_choice.index]
        choices = pd.concat([single_choices, dup_choices_to_keep])

        if choices.net_units.sum() >= target_units:
            break

        df = df[~df.parcel_id.isin(choices.parcel_id)]
        if len(df) == 0:
            break

        p = p.reindex(df.index)
        p = p / p.sum()
        if target_units is not None:
            new_target = target_units - choices.net_units.sum()
        else:
            new_count = count - choices.shape[0]
        next_choice_idx = weighted_random_choice_by_count(df, p, new_target, new_count)
        next_choices = df.loc[next_choice_idx]
        choices = pd.concat([choices, next_choices])
    return choices.index.values


class PSRCDeveloper(develop.Developer):
    """
    Child of the UDST developer class. 
    The purpose is mostly to write our own methods in order to overwrite some default behavior.
    """
    
    def __init__(self, feasibility, forms, target_units, parcel_size,
                 ave_unit_size, current_units, *args, **kwargs):
        develop.Developer.__init__(self, feasibility, forms, target_units, parcel_size, 
                         ave_unit_size, current_units,  *args, **kwargs)
        pcl = orca.get_table("parcels")
        self.current_units_res = pcl[current_units[0]]
        self.current_units_nonres = pcl[current_units[1]]
        
    def _calculate_net_units(self, df):
        """
        Helper method to pick(). Calculates the net_units column,
        and removes buildings that have net_units of 0 or less.

        Parameters
        ----------
        df : DataFrame
            DataFrame of buildings from _remove_infeasible_buildings()

        Returns
        -------
        df : DataFrame
        """
        if len(df) == 0 or df.empty:
            return df
        #TODO: for each proposal compute residential units (from self.ave_unit_size) and 
        #      job_spaces using feasibility_bt & convert to a common currency
        #if self.residential:
        #df['net_units_res'] = df.residential_units - df.current_units_res
        #else:
        #df['net_units_nonres'] = df.job_spaces - df.current_units_nonres
        #TODO: convert to a common currency
        #df['net_units'] = 
        return df[df.net_units > 0] 
    
    def _remove_infeasible_buildings(self, df):
        """
        Helper method to pick(). Removes buildings from the DataFrame if:
            - max_profit_far is 0 or less
            - parcel_size is larger than max_parcel_size

        Also calculates useful DataFrame columns from object attributes
        for later calculations.

        Parameters
        ----------
        df : DataFrame
            DataFrame of buildings from _get_dataframe_of_buildings()

        Returns
        -------
        df : DataFrame
        """
        if len(df) == 0 or df.empty:
            return df

        df = df[df.max_profit_far > 0]
        #self.ave_unit_size[
        #    self.ave_unit_size < self.min_unit_size
        #] = self.min_unit_size
        #df.loc[:, 'ave_unit_size'] = self.ave_unit_size
        df.loc[:, 'parcel_size'] = self.parcel_size
        df.loc[:, 'current_units_res'] = self.current_units_res
        df.loc[:, 'current_units_nonres'] = self.current_units_nonres
        df = df[df.parcel_size < self.max_parcel_size]

        #df['residential_units'] = (df.residential_sqft /
        #                           df.ave_unit_size).round()
        #df['job_spaces'] = (df.non_residential_sqft /
        #                    self.bldg_sqft_per_job).round()

        return df    