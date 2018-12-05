import os
import numpy as np
import pandas as pd
import orca
from developer import sqftproforma
from urbansim.utils import misc
from urbansim_defaults.utils import yaml_to_class, to_frame

#from urbansim_defaults.utils import apply_parcel_callbacks, lookup_by_form

@orca.injectable("proforma_settings")
def proforma_settings(land_use_types, building_types, development_templates, development_template_components):
    uses =  pd.merge(development_template_components.local[["building_type_id", "template_id", "description", "percent_building_sqft"]],
                         development_templates.local[["land_use_type_id"]], left_on="template_id", right_index=True, how="left")
    uses.description.iloc[np.core.defchararray.startswith(uses.description.values.astype("string"), "sfr")] = "sfr" # since there are 2 sfr uses (sfr_plat, sfr_parcel)
    # remove template_id in order to remove duplicates
    blduses = uses.drop("template_id", 1).drop_duplicates()
    # add template_id back in order to group the components into forms
    blduses[["template_id"]] = uses[["template_id"]] 
    # to make sure that all components of included templates are present (in case they were dropped)
    blduses = uses[uses.template_id.isin(blduses.template_id.values)]
    blduses = pd.merge(blduses, building_types.local[["building_type_name", "is_residential"]], left_on="building_type_id", right_index=True, how="left")
    blduses = pd.merge(blduses, land_use_types.local[["land_use_name", "generic_land_use_type_id"]], left_on="land_use_type_id", right_index=True, how="left")
    # rename duplicated description
    tmp = blduses[['template_id', 'description']].drop_duplicates()
    is_dupl = tmp.duplicated('description')
    if is_dupl.any():
        dupltmp = tmp[is_dupl]
        for desc in np.unique(dupltmp.description):
            thisdescr = dupltmp[dupltmp.description == desc]
            blduses['description'][thisdescr.index] = blduses['description'][thisdescr.index]+ np.arange(2,thisdescr.index.size+2).astype("str")
    return blduses

# Empty function. Series indexed by parcel_id
@orca.injectable("parcel_price_placeholder", autocall=False)
def parcel_price_placeholder(use, **kwargs):
    return orca.get_table('parcels').land_value

# Return price per sqft for given use (building type). Series indexed by parcel_id
@orca.injectable("parcel_sales_price_sqft_func", autocall=False)
def parcel_sales_price_sqft_func(use, config):
    pcl = orca.get_table('parcels')
    # Temporarily use the expected sales price model coefficients
    coef_const = config.price_coefs[np.logical_and(config.price_coefs.building_type_name == use, config.price_coefs.coefficient_name == "constant")].estimate
    coef = config.price_coefs[np.logical_and(config.price_coefs.building_type_name == use, config.price_coefs.coefficient_name == "lnclvalue_psf")].estimate
    return np.exp(coef_const.values + coef.values*np.log(pcl.land_value/pcl.parcel_sqft)).replace(np.inf, np.nan)

@orca.injectable("parcel_is_allowed_func", autocall=False)
def parcel_is_allowed_func(form):
    config = orca.get_injectable("pf_config")
    bt_distr = config.forms[form]
    glu = config.form_glut[form]
    zoning = orca.get_table('parcel_zoning')
    btused = config.residential_uses.index[bt_distr > 0]
    is_res_bt = config.residential_uses[btused]
    units = ["far", "units_per_acre"]
    parcels = orca.get_table('parcels')
    result = pd.Series(0, index=parcels.index)
    for typ in is_res_bt.index:
        unit = units[is_res_bt[typ]]
        this_zoning = zoning.local.loc[np.logical_and(zoning.index.get_level_values("constraint_type") == unit, 
                                                      zoning.index.get_level_values("generic_land_use_type_id") == glu)]
        pcls = this_zoning.index.get_level_values("parcel_id")
        result[pcls] = result[pcls] + 1
    return (result == is_res_bt.index.size)

def update_sqftproforma(default_settings, proforma_uses, **kwargs):
    local_settings = {}
    blduses = proforma_uses[["building_type_id", "building_type_name", "is_residential"]].drop_duplicates()
    blduses = pd.merge(pd.DataFrame({"uses":default_settings.uses}), blduses, left_on="uses", right_on="building_type_name")
    local_settings["uses"] = blduses.uses.values
    local_settings["residential_uses"] = blduses.is_residential
    local_settings["residential_uses"].index = blduses.building_type_id
    coeffile = os.path.join(misc.data_dir(), "expected_sales_unit_price_component_model_coefficients.csv")
    coefs = pd.read_csv(coeffile)
    coefs = pd.merge(coefs, proforma_uses[['building_type_name', "building_type_id"]].drop_duplicates(), right_on="building_type_id", left_on="sub_model_id", how="left")
    local_settings["price_coefs"] = coefs    
    forms = {}
    form_glut = {}
    for formid in np.unique(proforma_uses.template_id):
        subuse = proforma_uses[proforma_uses.template_id==formid]
        submerge = pd.merge(blduses, subuse, on='building_type_name', how="left")
        forms[subuse.description.values[0]] = submerge.percent_building_sqft.fillna(0).values/100.
        form_glut[subuse.description.values[0]] = subuse.generic_land_use_type_id.values[0]

    # Convertion similar to sqftproforma._convert_types()
    local_settings["res_ratios"] = {}
    new_btype_id = {}
    for form in forms.keys():
        forms[form] /= forms[form].sum() # normalize
        local_settings["res_ratios"][form] = pd.Series(forms[form][np.where(local_settings["residential_uses"])]).sum()
        # find future building type
        bts = local_settings["uses"][forms[form] > 0]
        if bts.size == 1: # no mixed use
            new_btype_id[form] = blduses.building_type_id.values[blduses.building_type_name.values == bts[0]][0]
        else: # mixed use
            new_btype_id[form] = 10 # TODO: refine mixed use building types
            
    local_settings["forms"] = forms
    local_settings["forms_df"] = pd.DataFrame(forms, index = local_settings["uses"]).transpose()
    local_settings["form_glut"] = form_glut
    local_settings["new_btype_id"] = new_btype_id
    local_settings["forms_to_test"] = None
    local_settings['percent_of_max_profit'] = orca.settings['percent_of_max_profit']
    pf = default_settings
    for attr in local_settings.keys():
        setattr(pf, attr, local_settings[attr])
    pf.reference_dict = sqftproforma.SqFtProFormaReference(**pf.__dict__).reference_dict

    pf = update_sqftproforma_reference(pf)    
    return pf

def update_sqftproforma_reference(pf):
    for name, config in pf.reference_dict.keys():
        if name in ['tcu', 'warehouse']:
            pf.reference_dict[(name, config)]['ave_cost_sqft'][pf.fars > pf.max_industrial_height] = np.nan
    return pf      
    


def run_feasibility(parcels, parcel_price_callback,
                    parcel_use_allowed_callback, pipeline=False,
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
    pf = (sqftproforma.SqFtProForma.from_yaml(str_or_buffer=cfg)
          if cfg else sqftproforma.SqFtProForma.from_defaults())
    # Update default values using templates and store
    pf = update_sqftproforma(pf, **kwargs)
    orca.add_injectable("pf_config", pf)
    
    sites = (pl.remove_pipelined_sites(parcels) if pipeline
             else parcels.to_frame(parcels.local_columns))
    #df = apply_parcel_callbacks(sites, parcel_price_callback,
    #                            pf, **kwargs)

    # compute price for each use
    df = sites
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
        newdf = df[allowed]
        
        # Core function - computes profitability
        d[form] = pf.lookup(form, newdf, only_built = pf.only_built,
                            pass_through = pf.pass_through)

    # Collect results     
    if pf.proposals_to_keep > 1:
        # feasibility is in long format
        form_feas = []
        for form_name in d.keys():
            df_feas_form = d[form_name]
            df_feas_form['form'] = form_name
            form_feas.append(df_feas_form)
        
        feasibility = pd.concat(form_feas, sort=False)
        if pf.percent_of_max_profit > 0:
            feasibility['max_profit_parcel'] = feasibility.groupby([feasibility.index, 'form'])['max_profit'].transform(max)
            feasibility['ratio'] = feasibility.max_profit/feasibility.max_profit_parcel
            feasibility = feasibility[feasibility.ratio >= pf.percent_of_max_profit]
            feasibility.drop(['max_profit_parcel', 'ratio'], axis=1, inplace = True)
        feasibility.index.name = 'parcel_id'
        # add attribute that enumerates proposals (can be used as a unique index)
        feasibility["feasibility_id"] = np.arange(1, len(feasibility)+1, dtype = "int32")
        # create a dataset with disaggregated sqft by building type
        feas_bt = pd.merge(feasibility.loc[:, ["form", "feasibility_id", "residential_sqft", "non_residential_sqft"]], pf.forms_df, left_on = "form", right_index = True)
        feas_bt.set_index(['form'], append = True, inplace = True)
        feas_bt[pf.uses[pf.residential_uses.values == 1]] = feas_bt[pf.uses[pf.residential_uses.values == 1]].multiply(feas_bt.residential_sqft, axis = "index")
        feas_bt[pf.uses[pf.residential_uses.values == 0]] = feas_bt[pf.uses[pf.residential_uses.values == 0]].multiply(feas_bt.non_residential_sqft, axis = "index")
        orca.add_table('feasibility_bt', feas_bt)
    else:
        # feasibility is in wide format
        feasibility = pd.concat(d.values(), keys = d.keys(), axis=1)        
           
    orca.add_table('feasibility', feasibility)
    return feasibility
