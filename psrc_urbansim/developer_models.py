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

# Return price per sqft for given use (building type). Series indexed by parcel_id
@orca.injectable("price_per_sqft_func", autocall=False)
def price_per_sqft_func(use, config):
    # Temporarily use the expected sales price model coefficients
    coef_const = config.price_coefs[np.logical_and(config.price_coefs.building_type_name == use, config.price_coefs.coefficient_name == "constant")].estimate
    coef = config.price_coefs[np.logical_and(config.price_coefs.building_type_name == use, config.price_coefs.coefficient_name == "lnclvalue_psf")].estimate
    pcl = orca.get_table('parcels')
    return np.exp(coef_const.values + coef.values*np.log(pcl.land_value/pcl.parcel_sqft)).replace(np.inf, np.nan)

@orca.injectable("parcel_is_allowed_func", autocall=False)
def parcel_is_allowed_func(form, bt_distr, glu, config):
    form_to_btype = orca.get_injectable("form_to_btype")
    zoning = orca.get_table('parcel_zoning')
    btused = config.residential_uses.index[bt_distr > 0]
    is_res_bt = config.residential_uses[btused]
    units = ["far", "units_per_acre"]
    result = pd.Series(0, index=orca.get_table('parcels').index)
    for typ in is_res_bt.index:
        unit = units[is_res_bt[typ]]
        this_zoning = zoning.local.loc[np.logical_and(zoning.index.get_level_values("constraint_type") == unit, 
                                                      zoning.index.get_level_values("generic_land_use_type_id") == glu)]
        pcls = this_zoning.index.get_level_values("parcel_id")
        result[pcls] = result[pcls] + 1
    allowed = result == is_res_bt.index.size
    return allowed
    # we have zoning by building type but want
    # to know if specific forms are allowed
    #allowed = [zoning
    #           ['type%d' % typ] == 't' for typ in form_to_btype[form]]
    #return pd.concat(allowed, axis=1).max(axis=1).\
    #    reindex(orca.get_table('parcels').index).fillna(False)

def update_sqftproforma(default_settings, proforma_uses):
    local_settings = default_settings
    blduses = proforma_uses[["building_type_id", "building_type_name", "is_residential"]].drop_duplicates()
    local_settings.uses = blduses.building_type_name.values
    local_settings.residential_uses = blduses.is_residential
    local_settings.residential_uses.index = blduses.building_type_id
    coeffile = os.path.join(misc.data_dir(), "expected_sales_unit_price_component_model_coefficients.csv")
    coefs = pd.read_csv(coeffile)
    coefs = pd.merge(coefs, proforma_uses[['building_type_name', "building_type_id"]].drop_duplicates(), right_on="building_type_id", left_on="sub_model_id", how="left")
    local_settings.price_coefs = coefs    
    forms = {}
    form_glut = {}
    for formid in np.unique(proforma_uses.template_id):
        subuse = proforma_uses[proforma_uses.template_id==formid]
        submerge = pd.merge(blduses, subuse, on='building_type_name', how="left")
        forms[subuse.description.values[0]] = submerge.percent_building_sqft.fillna(0).values/100.
        form_glut[subuse.description.values[0]] = subuse.generic_land_use_type_id.values[0]
    local_settings.parking_rates = np.array(local_settings.uses.size*[1.])
    # Convertion similar to sqftproforma._convert_types()
    local_settings.res_ratios = {}
    cost = {} # cost per buiilding type and height 
    for use in local_settings.uses:
        cost[use] = [160.0, 175.0, 200.0, 230.0] # cost of building per height (should be different per form)
    for form in forms.keys():
        forms[form] /= forms[form].sum() # normalize
        local_settings.res_ratios[form] = pd.Series(forms[form])[local_settings.residential_uses].sum()
    local_settings.costs = np.transpose(np.array([cost[use] for use in local_settings.uses]))
    local_settings.forms = forms
    local_settings.form_glut = form_glut
    return local_settings
    
# TODO: put the step into models.py and leave this as a function
@orca.step('proforma_feasibility')
def proforma_feasibility(parcels, proforma_settings, price_per_sqft_func,
                    parcel_is_allowed_func 
                    #residential_to_yearly=True
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
    
    #pf = sqftproforma.SqFtProForma(config=pf.config)
    # TODO: make sure all types are in the right format and eliminate _convert_types()
    #pf.config._convert_types()
    pf._generate_lookup()
    
    df = parcels.to_frame(parcels.local_columns + ['max_far', 'max_dua', 'max_height', 'ave_unit_size', 'parcel_size', 'land_cost'])
    #df = parcels.local

    # add prices for each use
    for use in pf.config.uses:
        df[use] = price_per_sqft_func(use=use, config=pf.config)

    # convert from cost to yearly rent
#    if residential_to_yearly:
#        df["residential"] *= pf.config.cap_rate

    print "Describe of the yearly rent by use"
    print df[pf.config.uses].describe()

    d = {}
    residential_forms = []
    non_residential_forms = []
    for form, btdistr in pf.config.forms.iteritems():
        print "Computing feasibility for form %s" % form
        d[form] = pf.lookup(form, df[parcel_is_allowed_func(form, btdistr, pf.config.form_glut[form], pf.config)])
        if (btdistr[pf.config.residential_uses.values>0] > 0).any():
            residential_forms.append(form)
        else:
            non_residential_forms.append(form)

    far_predictions = pd.concat(d.values(), keys=d.keys(), axis=1)
    far_predictions.residential_forms = residential_forms
    far_predictions.non_residential_forms = non_residential_forms
    orca.add_table("feasibility", far_predictions)

