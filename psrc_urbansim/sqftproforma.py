import os
import numpy as np
import pandas as pd
import orca
from developer import sqftproforma
from urbansim.utils import misc
from urbansim_defaults.utils import to_frame
from developer.utils import yaml_to_dict
from developer.utils import columnize

#from urbansim_defaults.utils import apply_parcel_callbacks, lookup_by_form

@orca.injectable("proforma_settings")
def proforma_settings(building_types, #land_use_types, development_templates, development_template_components, generic_land_use_types
                      ):
    formfile = os.path.join(misc.data_dir(), "forms.csv")
    forms = pd.read_csv(formfile)
    forms = forms.set_index("form_name")
    #uses =  pd.merge(development_template_components.local[["building_type_id", "template_id", "description", "percent_building_sqft"]],
    #                     development_templates.local[["land_use_type_id", "density_type"]], left_on="template_id", right_index=True, how="left")
    #uses.description.loc[np.core.defchararray.startswith(uses.description.values.astype("string"), "sfr")] = "sfr" # since there are 2 sfr uses (sfr_plat, sfr_parcel)
    # remove template_id in order to remove duplicates
    #blduses = uses.drop("template_id", 1).drop_duplicates()
    # add template_id back in order to group the components into forms
    #blduses[["template_id"]] = uses[["template_id"]] 
    # to make sure that all components of included templates are present (in case they were dropped)
    #blduses = uses[uses.template_id.isin(blduses.template_id.values)]
    blduses = building_types.local[["building_type_name", "is_residential"]]
    blduses = blduses.loc[blduses.building_type_name.isin(forms.columns), :]
    blduses["building_type_id"] = blduses.index
    #blduses = pd.merge(blduses, building_types.local[["building_type_name", "is_residential"]], left_on="building_type_id", right_index=True, how="left")
    #blduses = pd.merge(blduses, land_use_types.local[["land_use_name", "generic_land_use_type_id"]], left_on="land_use_type_id", right_index=True, how="left")
    #blduses = pd.merge(blduses, generic_land_use_types.local[["generic_land_use_type_name"]], how="left", on = "generic_land_use_type_id")
    # rename duplicated description
    #tmp = blduses[['template_id', 'description']].drop_duplicates()
    #is_dupl = tmp.duplicated('description')
    #if is_dupl.any():
        #dupltmp = tmp[is_dupl]
        #for desc in np.unique(dupltmp.description):
            #thisdescr = dupltmp[dupltmp.description == desc]
            #thisdescr.loc[:, "description"] = blduses['description'][thisdescr.index]+ np.arange(2,thisdescr.index.size+2).astype("str")
            #for templ in thisdescr.template_id.values:
                #blduses.loc[blduses.template_id == templ, 'description'] = thisdescr.loc[thisdescr.template_id == templ, "description"].values[0]
    return [blduses, forms]

# Empty function. Series indexed by parcel_id
@orca.injectable("parcel_price_placeholder", autocall=False)
def parcel_price_placeholder(use, **kwargs):
    return orca.get_table('parcels').land_value

# Return price per sqft for given use (building type). Series indexed by parcel_id
@orca.injectable("parcel_sales_price_func", autocall=False)
def parcel_sales_price_func(use, config):
    pcl = orca.get_table('parcels')
    # Temporarily use the expected sales price model coefficients
    isusecoef = config.price_coefs.building_type_name == use
    coef_const = config.price_coefs[np.logical_and(isusecoef, config.price_coefs.coefficient_name == "constant")].estimate
    coef_lvpsf = config.price_coefs[np.logical_and(isusecoef, config.price_coefs.coefficient_name == "lnclvalue_psf")].estimate
    values = coef_const.values + coef_lvpsf.values*np.log(pcl.land_value/pcl.parcel_sqft)
    if sum(isusecoef) > 2:
        coef_size = config.price_coefs[np.logical_and(isusecoef, config.price_coefs.coefficient_name == "lnpsqft")].estimate
        values = values + coef_size.values*np.log(pcl.parcel_sqft)
    #return np.exp(coef_const.values + coefvalue.values*np.log(pcl.land_value/pcl.parcel_sqft)).replace(np.inf, np.nan)
    #return np.exp(coef_const.values + coefvalue.values*np.log(pcl.land_value/pcl.parcel_sqft) + coefsize.values*np.log(pcl.parcel_sqft)).replace(np.inf, np.nan) 
    return np.exp(values).replace(np.inf, np.nan)
    #return values.replace(np.inf, np.nan).replace(-np.inf, np.nan)

@orca.injectable("parcel_is_allowed_func", autocall=False)
def parcel_is_allowed_func(form):
    config = orca.get_injectable("pf_config")
    glu = config.form_glut[form]
    zoning = orca.get_table('parcel_zoning').local
    res = zoning[[glu[0]]] > 0
    if len(glu) > 1:
        for gt in glu[1:]:
            res = np.logical_or(res, zoning[[gt]] > 0)
    return res

@orca.injectable("set_ave_unit_size_func", autocall=False)
def set_ave_unit_size_func(pf, form, df):
    attrs = []
    #if pf.forms_df.ix[form, "condo_residential"] > 0:
    #    attrs = attrs + ["ave_unit_size_condo"]
    if pf.forms_df.ix[form, "multi_family_residential"] > 0:
        attrs = attrs + ["ave_unit_size_mf"]
    if pf.forms_df.ix[form, "single_family_residential"] > 0:
        attrs = attrs + ["ave_unit_size_sf"]
    if len(attrs) == 0:
        return df
    df["ave_unit_size"] = df[attrs].mean(axis = 1)
    return df
    
def update_sqftproforma(default_settings, yaml_file, proforma_uses, **kwargs):    
    # extract uses
    blduses = proforma_uses[0]
    forms_df = proforma_uses[1]
    #blduses = proforma_uses[["building_type_id", "building_type_name", "is_residential"]].drop_duplicates()
    # put uses into the same order as the config file
    blduses = pd.merge(pd.DataFrame({"uses":default_settings.uses}), blduses, left_on="uses", right_on="building_type_name")
    forms_df_extr = forms_df[blduses.uses]
    # store in a dictionary
    local_settings = {}
    local_settings["uses"] = blduses.uses.values
    local_settings["residential_uses"] = blduses.is_residential
    local_settings["residential_uses"].index = blduses.building_type_id
    # get coefficient file for modeling price
    #coeffile = os.path.join(misc.data_dir(), "expected_sales_unit_price_component_model_coefficients.csv")
    coeffile = os.path.join(misc.data_dir(), "total_value_psf_coefficients.csv")
    coefs = pd.read_csv(coeffile)
    #coefs = pd.merge(coefs, proforma_uses[['building_type_name', "building_type_id"]].drop_duplicates(), right_on="building_type_id", left_on="sub_model_id", how="left")
    coefs = pd.merge(coefs, blduses[['building_type_name', "building_type_id"]].drop_duplicates(), right_on="building_type_id", left_on="sub_model_id", how="left")
    local_settings["price_coefs"] = coefs
    
    #bsqft_coef = coefs[np.logical_and(coefs.coefficient_name == "lnbsqft", ~coefs.building_type_name.isna())][["estimate", "building_type_name"]]
    #local_settings["bsqft_coefs"] = np.transpose(bsqft_coef.set_index("building_type_name"))
    
    # Assemble forms
    forms = {}
    form_glut = {}
    form_density_type = {}

    #for formid in np.unique(proforma_uses.template_id):
    for form_name in forms_df.index.values:
        #subuse = proforma_uses[proforma_uses.template_id==formid]
        #submerge = pd.merge(blduses, subuse, on='building_type_name', how="left")
        #form_name = subuse.description.values[0]
        #forms[form_name] = submerge.percent_building_sqft.fillna(0).values/100.
        #form_glut[form_name] = subuse.generic_land_use_type_name.values[0]
        #form_density_type[form_name] = subuse.density_type.values[0]
        forms[form_name] = forms_df_extr.ix[form_name].values
        form_glut[form_name] = forms_df.glut.ix[form_name].split(":")
        form_density_type[form_name] = forms_df.density_type.ix[form_name]

    # Conversion similar to sqftproforma._convert_types()
    local_settings["res_ratios"] = {}
    for form in forms.keys():
        forms[form] /= forms[form].sum() # normalize
        local_settings["res_ratios"][form] = pd.Series(forms[form][np.where(local_settings["residential_uses"])]).sum()
            
    all_default_settings = yaml_to_dict(None, yaml_file)
    local_settings["forms"] = forms
    #local_settings["forms_df"] = pd.DataFrame(forms, index = local_settings["uses"]).transpose()
    local_settings["forms_df"] = forms_df_extr
    local_settings["form_glut"] = form_glut
    local_settings["form_density_type"] = form_density_type
    local_settings["form_groups"] = forms_df[["group"]]
    local_settings["forms_to_test"] = None
    local_settings['minimum_floor_space'] = all_default_settings.get('minimum_floor_space', 150)
    local_settings['percent_of_max_profit'] = all_default_settings.get('percent_of_max_profit', 0) # Default is no restriction
    local_settings['percent_of_max_profit_per_group'] = all_default_settings.get('percent_of_max_profit_per_group', False)
    local_settings['proposals_to_keep_per_parcel'] = all_default_settings.get('proposals_to_keep_per_parcel', None)
    local_settings['proposals_to_keep_per_form'] = all_default_settings.get('proposals_to_keep_per_form', None)
    local_settings['proposals_to_keep_per_group'] = all_default_settings.get('proposals_to_keep_per_group', None)
    pf = default_settings
    for attr in local_settings.keys():
        setattr(pf, attr, local_settings[attr])
    pf.reference_dict = sqftproforma.SqFtProFormaReference(**pf.__dict__).reference_dict

    pf = update_sqftproforma_reference(pf)    
    return pf

def update_sqftproforma_reference(pf):
    for name, config in pf.reference_dict.keys():
        if name in ['tcu', 'warehouse']:
            pf.reference_dict[(name, config)]['ave_cost_sqft'][pf.reference_dict[(name, config)].far > pf.max_industrial_height] = np.nan
    return pf      
    

def run_feasibility(parcels, parcel_price_callback,
                    parcel_use_allowed_callback, lookup_modify_callback=None,
                    pipeline=False,
                    cfg=None, **kwargs):
    """
    Execute development feasibility on all development sites

    Parameters
    ----------
    parcels : DataFrame Wrapper
        The data frame wrapper for the parcel data
    parcel_price_callback : function
        A callback which takes each use of the pro forma and returns a series
        with index as parcel_id and value as yearly_rent
    parcel_use_allowed_callback : function
        A callback which takes each form of the pro forma and returns a series
        with index as parcel_id and value and boolean whether the form
        is allowed on the parcel
    pipeline : bool, optional
        If True, removes parcels from consideration if already in dev_sites
        table
    cfg : str, optional
        The name of the yaml file to read pro forma configurations from
    """

    cfg = misc.config(cfg)
    
    # Create default SqFtProForma
    pf = (PSRCSqFtProForma.from_yaml(str_or_buffer=cfg)
          if cfg else PSRCSqFtProForma.from_defaults())
    # Update default values using templates and store
    pf = update_sqftproforma(pf, cfg, **kwargs)
    orca.add_injectable("pf_config", pf)
    
    sites = (pl.remove_pipelined_sites(parcels) if pipeline
             else parcels.to_frame(parcels.local_columns))
    #df = apply_parcel_callbacks(sites, parcel_price_callback,
    #                            pf, **kwargs)
    
    # apply parcel filter
    df = sites.query(pf.parcel_filter)
    
    # compute price for each use
    for use in pf.uses:        
        df[use] = parcel_price_callback(use, pf)
            
    #feasibility = lookup_by_form(df, parcel_use_allowed_callback, pf, **kwargs)
    
    print "Describe of the yearly rent by use"
    print df[pf.uses].describe()

    # Computing actual feasibility
    d = {}
    forms = pf.forms_to_test or pf.forms
    for form in forms:
        print "Computing feasibility for form %s" % form
        #if parcel_id_col is not None:
        #    parcels = df[parcel_id_col].unique()
        #    allowed = (parcel_use_allowed_callback(form).loc[parcels])
        #    newdf = df.loc[misc.reindex(allowed, df.parcel_id)]
        #else:
        allowed = parcel_use_allowed_callback(form).loc[df.index]
        newdf = df[allowed.values]
        
        # Core function - computes profitability
        d[form] = pf.lookup(form, newdf, only_built = pf.only_built,
                            pass_through = pf.pass_through, modify_df = lookup_modify_callback)
        # apply parking ratio to res and non-res sqft
        if d[form].size > 0:
            d[form].residential_sqft = d[form].residential_sqft * (1 - d[form].parking_ratio)
            d[form].non_residential_sqft = d[form].non_residential_sqft * (1 - d[form].parking_ratio)

    # Collect results     
    # put feasibility into long format
    form_feas = []
    for form_name in d.keys():
        df_feas_form = d[form_name]
        df_feas_form['form'] = form_name
        form_feas.append(df_feas_form)
    
    feasibility = pd.concat(form_feas, sort=False)
    feasibility['parcel_id'] = feasibility.index.values
    
    # remove proposals below the minimum floor space
    feasibility = feasibility[feasibility.building_sqft > pf.minimum_floor_space]
    
    # add form group
    feasibility = pd.merge(feasibility, pf.form_groups, left_on = "form", right_index = True)
    
    # select only proposals with largest profit per parking and form
    feassort = feasibility.sort_values('max_profit', ascending=False)
    feasibility = feassort.groupby([feassort.index, 'form', 'max_profit_far']).head(1)
    
    # keep specified number of proposals per form
    if pf.proposals_to_keep_per_form is not None:
        feassort = feasibility.sort_values('max_profit', ascending=False)
        feasibility = feassort.groupby([feassort.index, 'form']).head(pf.proposals_to_keep_per_form) 
        
    # keep specified number of proposals per group
    if pf.proposals_to_keep_per_group is not None:
        feassort = feasibility.sort_values('max_profit', ascending=False)
        feasibility = feassort.groupby([feassort.index, 'group']).head(pf.proposals_to_keep_per_group) 
        
    # keep specified number of proposals per parcel
    if pf.proposals_to_keep_per_parcel is not None:
        feassort = feasibility.sort_values('max_profit', ascending=False)
        feasibility = feassort.groupby(feassort.index).head(pf.proposals_to_keep_per_parcel)     
    
    # adjust profit so that all parcels get developed, i.e. shift by the maximum negative profit per sqft
    feasibility['parcel_size'] = df.parcel_size
    #feasibility['max_profit_psf'] = feasibility.max_profit / feasibility.building_sqft
    feasibility['max_profit_psf'] = feasibility.max_profit / feasibility.parcel_size
    feasibility['max_profit_psf_parcel'] = feasibility.groupby(feasibility.index)['max_profit_psf'].transform(max)
    feasibility['max_profit_orig'] = feasibility['max_profit']
    if (feasibility.max_profit_psf_parcel < 0).any():
        feasibility.loc[feasibility.max_profit_psf < feasibility.max_profit_psf_parcel, 'max_profit_psf'] = np.nan
        feasibility.loc[feasibility.max_profit_psf_parcel < -100, 'max_profit_psf_parcel'] = -100        
        max_neg_profit_psf = feasibility.loc[feasibility.max_profit_psf_parcel < 0].max_profit_psf_parcel.min() - 0.001
        #feasibility['max_profit'] = feasibility['max_profit_orig'] - max_neg_profit_psf*feasibility.building_sqft
        feasibility['max_profit'] = feasibility['max_profit_orig'] - max_neg_profit_psf*feasibility.parcel_size
        # adjust max_profit of proposals where all proposals per parcels would be eliminated (all < -100). 
        # Set the profit so that the profit per sqft is 0.001 (thus will have a low weight)
        iadj = np.logical_and(~np.isnan(feasibility.max_profit_psf), feasibility.max_profit < 0)
        #feasibility.loc[iadj, 'max_profit'] = 0.001 * feasibility.loc[iadj, 'building_sqft']
        feasibility.loc[iadj, 'max_profit'] = 0.001 * feasibility.loc[iadj, 'parcel_size']
        
    feasibility.drop(['max_profit_psf_parcel', 'max_profit_psf'], axis=1, inplace = True)

    # remove proposals with negative adjusted profit
    feasibility = feasibility[feasibility.max_profit > 0]
    
    # keep proposals with profit within given percentage of max profit (per group or per parcel)
    if pf.percent_of_max_profit > 0:
        if pf.percent_of_max_profit_per_group:
            feasibility['max_profit_parcel'] = feasibility.groupby([feasibility.index, 'group'])['max_profit'].transform(max)
        else:
            feasibility['max_profit_parcel'] = feasibility.groupby(feasibility.index)['max_profit'].transform(max)
        feasibility['ratio'] = feasibility.max_profit/feasibility.max_profit_parcel
        feasibility = feasibility[feasibility.ratio >= pf.percent_of_max_profit / 100.]
        feasibility.drop(['max_profit_parcel', 'ratio'], axis=1, inplace = True)
        
    feasibility.index.name = 'parcel_id'
    
    # normalize weights within groups (divide by the number of proposals within each group)
    feasibility['max_profit'] = feasibility.max_profit / feasibility.groupby([feasibility.index, 'group'])["max_profit"].transform("count")
    
    # add attribute that enumerates proposals (can be used as a unique index)
    feasibility["feasibility_id"] = np.arange(1, len(feasibility)+1, dtype = "int32")
    # create a dataset with disaggregated sqft by building type
    feas_bt = pd.merge(feasibility.loc[:, ["form", "feasibility_id", "residential_sqft", "non_residential_sqft"]], pf.forms_df, left_on = "form", right_index = True)
    feas_bt.set_index(['form'], append = True, inplace = True)
    feas_bt[pf.uses[pf.residential_uses.values == 1]] = feas_bt[pf.uses[pf.residential_uses.values == 1]].multiply(feas_bt.residential_sqft, axis = "index")
    feas_bt[pf.uses[pf.residential_uses.values == 0]] = feas_bt[pf.uses[pf.residential_uses.values == 0]].multiply(feas_bt.non_residential_sqft, axis = "index")
    orca.add_table('feasibility_bt', feas_bt)     
           
    orca.add_table('feasibility', feasibility)
    return feasibility

class PSRCSqFtProForma(sqftproforma.SqFtProForma):
    
    def _min_max_fars(self, df, resratio):
        """
        This updates the parent method - we do not want to minimize between far 
        from dua and far.
        Parameters
        ----------
        df : DataFrame
            DataFrame of developable sites/parcels passed to lookup() method
        resratio : numeric
            Residential ratio for this form

        Returns
        -------
        Series
        """    
        
        df['max_far_from_heights_times_coverage'] = df.max_far_from_heights * df.max_coverage
        if 'max_dua' in df.columns and resratio > 0:
            # if max_dua is in the data frame, ave_unit_size must also be there
            assert 'ave_unit_size' in df.columns

            df['max_far_from_dua'] = (
                # this is the max_dua times the parcel size in acres, which
                # gives the number of units that are allowable on the parcel
                df.max_dua * (df.parcel_size / 43560) *

                # times by the average unit size which gives the square footage
                # of those units
                df.ave_unit_size /

                # divided by the building efficiency which is a
                # factor that indicates that the actual units are not the whole
                # FAR of the building
                self.building_efficiency /

                # divided by the resratio which is a  factor that indicates
                # that the actual units are not the only use of the building
                resratio /
                
                # divided by the parcel size again in order to get FAR.
                # I recognize that parcel_size actually
                # cancels here as it should, but the calc was hard to get right
                # and it's just so much more transparent to have it in there
                # twice
                
                df.parcel_size)
            if resratio > 0.9999:
                df['max_far_total'] = df.max_far_from_dua
            else:
                # if it is a real mix of res and non-res, sum max_far and max_far_from_dua 
                df['max_far_total'] = np.where(np.isnan(df.max_far), df.max_far_from_dua, df.max_far + df.max_far_from_dua)            
            #return df[['max_far', 'max_far_from_dua', 'max_far_from_heights']].min(axis=1)
        else:
            # if max_far is given than take that otherwise max_far_from_heights
            df['max_far_total'] = np.where(np.isnan(df.max_far), df.max_far_from_heights_times_coverage, df.max_far)
        # cap at max_far_from_heights
        return df[['max_far_total', 'max_far_from_heights_times_coverage']].min(axis=1)
        
    def check_is_reasonable(self):
        fars = pd.Series(self.fars)
        #assert len(fars[fars > 20]) == 0
        assert len(fars[fars <= 0]) == 0
        for k, v in self.forms.items():
            assert isinstance(v, dict)
            for k2, v2 in self.forms[k].items():
                assert isinstance(k2, str)
                assert isinstance(v2, float)
            for k2, v2 in self.forms[k].items():
                assert isinstance(k2, str)
                assert isinstance(v2, float)
        for k, v in self.parking_rates.items():
            assert isinstance(k, str)
            assert k in self.uses
            assert 0 <= v < 5
        for k, v in self.parking_sqft_d.items():
            assert isinstance(k, str)
            assert k in self.parking_configs
            assert 50 <= v <= 1000
        for k, v in self.parking_sqft_d.items():
            assert isinstance(k, str)
            assert k in self.parking_cost_d
            assert 10 <= v <= 300
        for v in self.heights_for_costs:
            assert isinstance(v, int) or isinstance(v, float)
            if np.isinf(v):
                continue
            assert 0 <= v <= 1000
        for k, v in self.costs.items():
            assert isinstance(k, str)
            assert k in self.uses
            for i in v:
                assert 10 < i < 1000
                
                
    def _lookup_parking_cfg(self, form, parking_config, df,
                            modify_df, modify_revenues, modify_costs,
                            modify_profits):
        """
        This is the core square foot pro forma calculation. For each form and
        parking configuration, generate DataFrame with profitability
        information

        Parameters
        ----------
        form : str
            Name of form
        parking_config : str
            Name of parking configuration
        df : DataFrame
            DataFrame of developable sites/parcels passed to lookup() method
        modify_df : func
            Function to modify lookup DataFrame before profit calculations.
            Must have (self, form, df) as parameters.
        modify_revenues : func
            Function to modify revenue ndarray during profit calculations.
            Must have (self, form, df, revenues) as parameters.
        modify_costs : func
            Function to modify cost ndarray during profit calculations.
            Must have (self, form, df, costs) as parameters.
        modify_profits : func
            Function to modify profit ndarray during profit calculations.
            Must have (self, form, df, profits) as parameters.

        Returns
        -------
        outdf : DataFrame
        """
        # don't really mean to edit the df that's passed in
        df = df.copy()

        # Reference table for this form and parking configuration
        dev_info = self.reference_dict[(form, parking_config)]

        # Helper values
        cost_sqft_col = columnize(dev_info.ave_cost_sqft.values)
        cost_sqft_index_col = columnize(dev_info.index.values)
        parking_sqft_ratio = columnize(dev_info.parking_sqft_ratio.values)
        heights = columnize(dev_info.height.values)
        months = columnize(dev_info.construction_months.values)
        resratio = self.res_ratios[form]
        nonresratio = 1.0 - resratio
        

        # Allow for user modification of DataFrame here
        df = modify_df(self, form, df) if modify_df else df

        # ZONING FILTERS
        # Minimize between max_fars and max_heights
        df['max_far_from_heights'] = (df.max_height
                                      / self.height_per_story
                                      * self.parcel_coverage)

        df['min_max_fars'] = self._min_max_fars(df, resratio)

        if self.only_built:
            df = df.query('min_max_fars > 0 and parcel_size > 0')

        # turn fars and heights into nans which are not allowed by zoning
        # (so we can fillna with one of the other zoning constraints)
        fars = np.repeat(cost_sqft_index_col, len(df.index), axis=1)
        mask = ~np.isnan(fars)  # mask out existing nans for safer comparison
        if form == "multi_family_residential":
            min_far_ratio = 0.6
        else:
            min_far_ratio = 0.4
        mask *= np.logical_or(np.nan_to_num(fars) > df.min_max_fars.values + .01, np.nan_to_num(fars) < df.min_max_fars.values*min_far_ratio)
        fars[mask] = np.nan

        heights = np.repeat(heights, len(df.index), axis=1)
        mask = ~np.isnan(heights)
        mask *= np.nan_to_num(heights) > df.max_height.values + .01
        fars[mask] = np.nan

        # PROFIT CALCULATION
        # parcel sizes * possible fars
        building_bulks = fars * df.parcel_size.values

        # cost to build the new building
        building_costs = building_bulks * cost_sqft_col

        # add cost to buy the current building
        total_construction_costs = building_costs + df.land_cost.values
        #total_construction_costs = building_costs
        
        # Financing costs
        loan_amount = total_construction_costs * self.loan_to_cost_ratio
        months = np.repeat(months, len(df.index), axis=1)
        interest = (loan_amount
                    * self.drawdown_factor
                    * (self.interest_rate / 12 * months))
        points = loan_amount * self.loan_fees
        total_financing_costs = interest + points
        total_development_costs = (total_construction_costs
                                   + total_financing_costs)
        #total_development_costs = total_construction_costs

        # rent to make for the new building

        #rent = 0 * building_bulks
        #for use in self.uses[self.forms[form] > 0]:
            #bblk = building_bulks * self.forms[use]
        #    rent = rent +  self.forms_df.ix[form][use] * np.exp(df[use][np.newaxis, :] + np.log(building_bulks) * self.bsqft_coefs[use].values)
        #rent = rent / building_bulks
        df['weighted_rent'] = np.dot(df[self.uses], self.forms[form])
        #df['weighted_rent'] = np.dot(rent, self.forms[form])
        building_revenue = (building_bulks
                            * (1 - parking_sqft_ratio)
                            * self.building_efficiency
                            * df.weighted_rent.values
                            #* rent
                            / self.cap_rate)

        # profit for each form, including user modification of
        # revenues, costs, and/or profits

        building_revenue = (modify_revenues(self, form, df, building_revenue)
                            if modify_revenues else building_revenue)

        total_development_costs = (
            modify_costs(self, form, df, total_development_costs)
            if modify_costs else total_development_costs)

        profit = building_revenue - total_development_costs
        
        #profit = building_revenue

        profit = (modify_profits(self, form, df, profit)
                  if modify_profits else profit)

        profit = profit.astype('float')
        profit[np.isnan(profit)] = -np.inf

        if self.proposals_to_keep > 1:
            maxprofit_sorted_indexes = np.argsort(-profit, axis=0)
            maxprofitind = maxprofit_sorted_indexes[:self.proposals_to_keep]
        else:
            maxprofitind = np.argmax(profit, axis=0)

        def twod_get(indexes, arr):
            if indexes.ndim == 1:
                return arr[indexes, np.arange(indexes.size)].astype('float')
            elif indexes.ndim == 2:
                arr = arr[indexes, np.arange(indexes.shape[1])]
                return arr.astype('float').flatten()

        if self.proposals_to_keep == 1:
            outdf_index = df.index
        else:
            outdf_index = np.tile(df.index, self.proposals_to_keep)

        outdf = pd.DataFrame({
            'building_sqft': twod_get(maxprofitind, building_bulks),
            'building_cost': twod_get(maxprofitind, building_costs),
            'parking_ratio': parking_sqft_ratio[maxprofitind].flatten(),
            'stories': twod_get(maxprofitind,
                                heights) / self.height_per_story,
            'total_cost': twod_get(maxprofitind, total_development_costs),
            'building_revenue': twod_get(maxprofitind, building_revenue),
            'max_profit_far': twod_get(maxprofitind, fars),
            'max_profit': twod_get(maxprofitind, profit),
            'parking_config': parking_config,
            'construction_time': twod_get(maxprofitind, months),
            'financing_cost': twod_get(maxprofitind, total_financing_costs)
        }, index=outdf_index)

        if self.pass_through:
            outdf[self.pass_through] = df[self.pass_through]

        outdf["residential_sqft"] = (outdf.building_sqft *
                                     self.building_efficiency *
                                     resratio)
        outdf["non_residential_sqft"] = (outdf.building_sqft *
                                         self.building_efficiency *
                                         nonresratio)

        if self.only_built:
            outdf = outdf.query('max_profit > 0').copy()
        else:
            outdf = outdf.loc[outdf.max_profit != -np.inf].copy()

        return outdf