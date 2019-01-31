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
def proposal_selection(self, proposals, p, targets):
    """
    Passed to custom_selection_func in Developer.pick().
    """
    chunksize = self.config.get("chunk_size", 100)
    pf = self.pf_config
    all_choices = pd.Series([], dtype = "int32")
    
    # working proposal dataset - to be reduced in each iteration
    df = proposals.copy()
    
    # Set index of proposals and p to feasibility_id 
    # to match self.net_units
    df.set_index("feasibility_id", inplace = True, drop = False)
    p.index = df.index
    
    # Keep a full proposal dataset for checking the vacancy 
    # in each iteration step.
    orig_df = df.copy()
    
    # remove proposals for which target vacancy is already met
    proposals_to_remove = filter_by_vacancy(orig_df, pf.uses, targets, net_units = self.net_units)
    if proposals_to_remove.size > 0:
        df = df.drop(proposals_to_remove)
        p = p.drop(proposals_to_remove)
        p = p / p.sum()
        
    while len(df) > 0:        
        # Sample by chunks and check vacancy after each step
        choice_idx = pd.Series(weighted_random_choice_multiparcel_by_count(df, p, count = min(chunksize, df.shape[0])))
        all_choices = pd.concat([all_choices, choice_idx])
        # remove proposals that:
        proposals_to_remove = pd.concat([pd.Series( # 1) satisfy target vacancy 
                                            filter_by_vacancy(orig_df, pf.uses, targets, 
                                                              net_units = self.net_units, choices = all_choices)), 
                                         choice_idx, # 2) were sampled
                                         df.index[df.parcel_id.isin(df.parcel_id[choice_idx])].to_series() # 3) are on parcels for which proposals were sampled
                                         ])
        proposals_to_remove = proposals_to_remove[proposals_to_remove.isin(df.index)]
        if proposals_to_remove.size > 0:
            df = df.drop(proposals_to_remove)
            p = p.drop(proposals_to_remove)
            p = p / p.sum()
            
    return proposals.index[proposals.feasibility_id.isin(all_choices)]


def filter_by_vacancy(df, uses, targets, net_units, choices = None):
    # Iterate over building types and check net_units vs. targets. 
    # Return feasibility_id of proposals that should be switched off.
    vacancy_met = pd.Series([], dtype = "int32")
    for use in uses:
        if targets[use] is None:
            continue
        units = net_units[use].reindex(df.index)
        if units.sum() == 0:
            continue
        if (choices is None and targets[use] == 0) or (choices is not None and units.loc[choices].sum() >= targets[use]):
            vacancy_met = pd.concat([vacancy_met, units.index[units > 0].to_series()])
    return vacancy_met.unique()

    
def compute_target_units(vacancy_rate, unlimited = False):
    pf = orca.get_injectable("pf_config")
    if unlimited:
        return dict.fromkeys(pf.uses)
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
                  parcel_size, ave_unit_size, cfg, current_units = ["units", "job_spaces"], year=None,
                  target_vacancy=0.1, form_to_btype_callback=None,
                  add_more_columns_callback=None,
                  remove_developed_buildings=True,
                  unplace_agents=['households', 'jobs'],
                  num_units_to_build=None, profit_to_prob_func=None,
                  custom_selection_func=None, pipeline=False, keep_suboptimal=True,
                  building_sqft_per_job = None):
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
    #dev = develop.Developer.from_yaml(
    dev = PSRCDeveloper.from_yaml(
                                      feasibility.to_frame(), forms,
                                      target_units, parcel_size,
                                      ave_unit_size, current_units,
                                      year, str_or_buffer=cfg, 
                                      keep_suboptimal=keep_suboptimal)
    # keep developer config
    dev.config = devutils.yaml_to_dict(yaml_str = None, str_or_buffer=cfg)
    
    print("{:,} feasible buildings before running developer".format(
        len(dev.feasibility)))

    dev.feasibility_bt = orca.get_table("feasibility_bt").local
    dev._calculate_units_from_sqft(building_sqft_per_job)
    dev._calculate_current_units(current_units)
    
    new_buildings = dev.pick(profit_to_prob_func, custom_selection_func)
    orca.add_table("feasibility", dev.feasibility)

    if new_buildings is None or len(new_buildings) == 0:
        return
    
    # Disaggragate buildings into one building per building type
    new_buildings.set_index("feasibility_id", drop = False, inplace = True)
    dev.feasibility_bt_units = dev.feasibility_bt_units.reindex(new_buildings.index)
    disaggr_buildings = disaggregate_buildings(new_buildings, dev.feasibility_bt_units[dev.pf_config.uses], 
                                               dev.building_types, dev.pf_config.forms_df)
    
    # Join old and new buildings
    new_buildings = add_buildings(buildings, disaggr_buildings,
                                  form_to_btype_callback,
                                  add_more_columns_callback,
                                  supply_fname, remove_developed_buildings,
                                  unplace_agents, pipeline)

    return new_buildings

def disaggregate_buildings(buildings, bt_units, building_types, forms):
    # Takes a dataset of selected proposals (buildings) which can have multiple building types in one row,
    # and disaggregates it into one record per building type.
    # The bt_units dataset contains proposed units by building type for each feasibility_id.
    
    # convert proposed units to long format
    longdf = bt_units.reset_index().melt(id_vars = ["feasibility_id"], var_name = "building_type_name", 
                                 value_name = "units").set_index(["feasibility_id", "building_type_name"])
    longdf = longdf[longdf.units > 0]
    
    # Now we have one record per building type. Merge the other proposal columns into it.
    dbuildings = longdf.merge(buildings, how = "left", left_index = True, right_index = True)
    
    # Some attributes need to be split. It depend if building is residential or not.
    # Compute ratios of total residential and total nonresidential units.
    frms = forms.copy()
    res_types = building_types.name[building_types.is_residential == 1]
    nonres_types = building_types.name[building_types.is_residential == 0]
    frms.loc[:, "res_ratio"] = forms[res_types].sum(axis = 1)
    frms.loc[:, "nonres_ratio"] = 1 - frms.res_ratio
    ratio = pd.DataFrame({"ratio": np.zeros(len(dbuildings), dtype = "float32"),
                          "total_ratio" : np.zeros(len(dbuildings), dtype = "float32"),
                          "feasibility_id": dbuildings.index.get_level_values("feasibility_id"),
                          "building_type_name": dbuildings.index.get_level_values("building_type_name"),
                          "form": dbuildings.form})
    ratio.set_index(["feasibility_id", "building_type_name", "form"], inplace = True)
    idx_obj = pd.IndexSlice
    # Iterate over building types and fill in the ratios
    for use in building_types.name:
        if use in res_types.values:
            ratio_name = "res_ratio"
        else:
            ratio_name = "nonres_ratio"
        subblds = dbuildings.xs(use, level = "building_type_name")
        if len(subblds) == 0: 
            continue
        btdistr = frms.loc[subblds.form, [use, ratio_name]]
        btdistr.index = subblds.index
        idx = idx_obj[btdistr.index.values, use, subblds.form]
        ratio.loc[idx, "ratio"] = btdistr[use].values
        ratio.loc[idx, "total_ratio"] = btdistr[ratio_name].values
        
    ratio.index = ratio.index.droplevel("form")
    
    # split attributes that are specifically either res or non-res
    for attr in ["residential_sqft", "non_residential_sqft"]:
        dbuildings[attr]  = dbuildings[attr] * (ratio.ratio / ratio.total_ratio).fillna(0)
            
    # split attributes that are common for res and non-res
    for attr in ["building_sqft", "building_cost", "building_revenue"]:
        dbuildings[attr]  = dbuildings[attr] * ratio.ratio

    # assign building_type_id
    bts = building_types.reset_index().set_index("name").rename_axis("building_type_name")
    dbuildings["building_type_id"] = bts.loc[dbuildings.index.get_level_values("building_type_name")].building_type_id.values
    dbuildings["is_residential"] = bts.loc[dbuildings.index.get_level_values("building_type_name")].is_residential.values
    
    # assign residential units and job_spaces
    dbuildings.loc[dbuildings.is_residential == 1, "residential_units"] = np.maximum(dbuildings.loc[dbuildings.is_residential == 1, "units"].round(), 1)
    dbuildings.loc[dbuildings.is_residential == 0, "job_spaces"] = np.maximum(dbuildings.loc[dbuildings.is_residential == 0, "units"].round(), 1)
    
    # set res attributes for non-res buildings to 0 and vice versa
    for attr in ["residential_sqft", "residential_units"]:
        dbuildings.loc[dbuildings.is_residential == 0, attr] = 0
    for attr in ["non_residential_sqft", "job_spaces"]:
        dbuildings.loc[dbuildings.is_residential == 1, attr] = 0
    
    # assign sqft_per_unit
    dbuildings.loc[:, "sqft_per_unit"] = 1
    dbuildings.loc[dbuildings.is_residential == 1, "sqft_per_unit"] = dbuildings.building_sqft / dbuildings.residential_units
    
    # drop the building_type_name index
    dbuildings.index = dbuildings.index.droplevel("building_type_name")
    
    return dbuildings
    

def add_buildings(buildings, new_buildings,
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

    # Correspondence of some the building columns to the proforma attributes
    # They only need to be defined if the names differ.
    new_cols = {'job_capacity': new_buildings.job_spaces,
                'land_area': new_buildings.building_sqft / new_buildings.stories, 
                'improvement_value': new_buildings.building_revenue
                }
    if add_more_columns_callback is not None:
        new_buildings = add_more_columns_callback(new_buildings, new_cols)

    if supply_fname is not None:
        if not isinstance(supply_fname, list):
            supply_fname = [supply_fname]
        for fname in supply_fname:
            print("Adding {:,} buildings with {:,} {}".format(
                len(new_buildings[new_buildings[fname] > 0]),
                int(new_buildings[fname].sum()),
                fname))

#    print("{:,} feasible buildings after running developer".format(
#        len(feasibility)))

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
    choices = np.random.choice(df.index,
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
        Proposals to select from. It has an unique index which is not necessarily parcel_id
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

        if target_units is None or choices.net_units.sum() >= target_units:
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
        
        #self.current_units_res = pcl[current_units[0]]
        #self.current_units_nonres = pcl[current_units[1]]
        self.pf_config = orca.get_injectable("pf_config")
        self.building_types = pd.concat([pd.Series(self.pf_config.uses, index = self.pf_config.residential_uses.index).rename("name"), 
                                         self.pf_config.residential_uses], axis = 1)
        # compute current units by building type
        pcl = orca.get_table("parcels")
        feas_bt = pd.merge(feasibility.loc[:, ["form", "feasibility_id"]], self.pf_config.forms_df, left_on = "form", right_index = True)
        self.current_units = {}
        for bt in self.building_types.index:
            if self.building_types.is_residential.ix[bt] == 1:
                unitsname = current_units[0]
            else:
                unitsname = current_units[1]
            btname = self.building_types.name.ix[bt]
            self.current_units[btname] = pcl["%s_%s" % (btname, unitsname)]
        
    def _calculate_current_units(self, current_units_attribute):
        # compute current units by building type (only BT of the corresponding form are considered)
        pcl = orca.get_table("parcels")
        self.current_units = pd.merge(self.feasibility.loc[:, ["form", "feasibility_id"]], self.pf_config.forms_df > 0, left_on = "form", right_index = True)
        for bt in self.building_types.index:
            if self.building_types.is_residential.ix[bt] == 1:
                unitsname = current_units_attribute[0]
            else:
                unitsname = current_units_attribute[1]
            btname = self.building_types.name.ix[bt]
            self.current_units[btname] = self.current_units[btname].values * pcl["%s_%s" % (btname, unitsname)].ix[self.current_units[btname].index].values
        self.current_units.set_index("feasibility_id", inplace = True)
        
    def _calculate_units_from_sqft(self, bldg_sqft_per_job = None):
        # Convert sqft into residential units and job_spaces. 
        # It is done based on building types of the proposals
        # In case of jobs, it uses the building_sqft_per_job table.
        
        units = self.feasibility_bt.copy()
        
        # compute residential units by building type
        for bt in self.building_types.name[self.building_types.is_residential == 1]:
            units.loc[:, bt] = units.loc[:, bt]/self.ave_unit_size[bt]
        units.loc[:, "residential_units"] = units.loc[:, self.building_types.name[self.building_types.is_residential == 1].values.tolist()].sum(axis = 1)
        self.feasibility.loc[:, "residential_units"] = units.loc[:, "residential_units"].values
        
        # compute job_spaces by building type
        if bldg_sqft_per_job is not None:
            pcl = orca.get_table("parcels")
            series1 = bldg_sqft_per_job.building_sqft_per_job.to_frame()
            series2 = pd.merge(self.feasibility, pd.DataFrame(pcl.zone_id), left_index=True, right_index=True)
          
        for bt in self.building_types.name[self.building_types.is_residential == 0].values:
            if bldg_sqft_per_job is None:
                denom = self.bldg_sqft_per_job
            else:
                series2.loc[:, "building_type_id"] = self.building_types[self.building_types.name == bt].index
                denom = pd.merge(series2, series1, left_on=['zone_id', 'building_type_id'], right_index=True, how="left").building_sqft_per_job
            units.loc[:, bt] = units.loc[:, bt]/denom.values
        units.loc[:, "job_spaces"] = units.loc[:, self.building_types.name[self.building_types.is_residential == 0].values.tolist()].sum(axis = 1)
        self.feasibility.loc[:, "job_spaces"] = units.loc[:, "job_spaces"].values
        self.feasibility_bt_units = units.set_index("feasibility_id")
        
    def _calculate_net_units(self, df):
        """
        Helper method to pick(). Calculates the net_units by building type,
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
        
        self.net_units = self.feasibility_bt_units[self.building_types.name.values] - self.current_units[self.building_types.name.values]
        # need to index by feasibility_id
        df["parcel_id"] = df.index
        dfc = df.set_index("feasibility_id", drop = False)
        dfc['net_units_res'] = self.net_units[self.building_types.name[self.building_types.is_residential == 1]].sum(axis = 1)
        dfc['net_units_nonres'] = self.net_units[self.building_types.name[self.building_types.is_residential == 0]].sum(axis = 1)
        # This is needed for some outputs (but does not make sense as we're adding DUs and job spaces)
        dfc['net_units'] = self.net_units.sum(axis = 1)
        return dfc[(dfc.net_units_res > 0) | (dfc.net_units_nonres > 0)].set_index("parcel_id")

    
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
        #df.loc[:, 'ave_unit_size_sf'] = self.ave_unit_size["single_family_residential"]
        df.loc[:, 'parcel_size'] = self.parcel_size
        #df.loc[:, 'current_units_res'] = self.current_units_res
        #df.loc[:, 'current_units_nonres'] = self.current_units_nonres
        df = df[df.parcel_size < self.max_parcel_size]

        #df['residential_units'] = (df.residential_sqft /
        #                           df.ave_unit_size).round()
        #df['job_spaces'] = (df.non_residential_sqft /
        #                    self.bldg_sqft_per_job).round()

        return df    