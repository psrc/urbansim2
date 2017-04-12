import os
import numpy as np
import pandas as pd
import orca
from urbansim.developer import sqftproforma, developer
from urbansim.utils import misc

@orca.injectable("proforma_settings")
def proforma_settings(land_use_types, building_types, development_templates, development_template_components):
    #templ_lu = np.unique(development_templates.land_use_type_id)
    #uses = land_use_types.local.loc[templ_lu]
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
    blduses = pd.merge(blduses, land_use_types.local[["land_use_name"]], left_on="land_use_type_id", right_index=True, how="left")
    # rename duplicated description
    tmp = blduses[['template_id', 'description']].drop_duplicates()
    is_dupl = tmp.duplicated('description')
    if is_dupl.any():
        dupltmp = tmp[is_dupl]
        for desc in np.unique(dupltmp.description):
            thisdescr = dupltmp[dupltmp.description == desc]
            blduses['description'][thisdescr.index] = blduses['description'][thisdescr.index]+ np.arange(2,thisdescr.index.size+2).astype("str")
    return blduses

# Return price per sqft for given use (building type). Series indexed by parcel_id
@orca.injectable("price_per_sqft_func", autocall=False)
def price_per_sqft_func(use, config):
    # Temporarily use the expected sales price model coefficients
    coef_const = config.price_coefs[np.logical_and(config.price_coefs.building_type_name == use, config.price_coefs.coefficient_name == "constant")].estimate
    coef = config.price_coefs[np.logical_and(config.price_coefs.building_type_name == use, config.price_coefs.coefficient_name == "lnclvalue_psf")].estimate
    pcl = orca.get_table('parcels')
    return np.exp(coef_const.values + coef.values*np.log(pcl.land_value/pcl.parcel_sqft)).replace(np.inf, np.nan)


def update_sqftproforma(default_settings, proforma_uses):
    local_settings = default_settings
    blduses = proforma_uses[["building_type_name", "is_residential"]].drop_duplicates()
    local_settings.uses = blduses.building_type_name.values
    local_settings.residential_uses = blduses.is_residential
    coeffile = os.path.join(misc.data_dir(), "expected_sales_unit_price_component_model_coefficients.csv")
    coefs = pd.read_csv(coeffile)
    coefs = pd.merge(coefs, proforma_uses[['building_type_name', "building_type_id"]].drop_duplicates(), right_on="building_type_id", left_on="sub_model_id", how="left")
    local_settings.price_coefs = coefs    
    forms = {}
    for formid in np.unique(proforma_uses.template_id):
        subuse = proforma_uses[proforma_uses.template_id==formid]
        submerge = pd.merge(blduses, subuse, on='building_type_name', how="left")
        forms[subuse.description.values[0]] = submerge.percent_building_sqft.fillna(0).values/100.
    local_settings.forms = forms
    return local_settings
    

@orca.step('proforma_feasibility')
def proforma_feasibility(parcels, proforma_settings, price_per_sqft_func
#                    parcel_use_allowed_callback, residential_to_yearly=True
):
    """
    Execute development feasibility on all parcels

    Parameters
    ----------
    parcels : DataFrame Wrapper
        The data frame wrapper for the parcel data
    price_per_sqft_func : function
        A callback which takes each use of the pro forma and returns a series
        with index as parcel_id and value as yearly_rent
    parcel_use_allowed_callback : function
        A callback which takes each form of the pro forma and returns a series
        with index as parcel_id and value and boolean whether the form
        is allowed on the parcel
    residential_to_yearly : boolean (default true)
        Whether to use the cap rate to convert the residential price from total
        sales price per sqft to rent per sqft

    Returns
    -------
    Adds a table called feasibility to the sim object (returns nothing)
    """
    pf = sqftproforma.SqFtProForma()
    
    pf.config = update_sqftproforma(pf.config, proforma_settings)
    
    #df = parcels.to_frame()
    df = parcels.local

    # add prices for each use
    for use in pf.config.uses:
        df[use] = price_per_sqft_func(use=use, config=pf.config)

    # convert from cost to yearly rent
#    if residential_to_yearly:
#        df["residential"] *= pf.config.cap_rate

    print "Describe of the yearly rent by use"
    print df[pf.config.uses].describe()

    d = {}
    for form in pf.config.forms:
        print "Computing feasibility for form %s" % form
        d[form] = pf.lookup(form, df[parcel_use_allowed_callback(form)])

    far_predictions = pd.concat(d.values(), keys=d.keys(), axis=1)

    orca.add_table("feasibility", far_predictions)
