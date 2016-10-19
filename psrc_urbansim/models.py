import orca
import random
import urbansim_defaults.utils as utils
import psrc_urbansim.utils as psrcutils
import dataset
import variables
import numpy as np
import pandas as pd

@orca.injectable()
def year(iter_var):
    return iter_var

# Residential REPM
@orca.step('repmres_estimate')
def repmres_estimate(parcels):
    return utils.hedonic_estimate("repmres.yaml", parcels, None)

@orca.step('repmres_simulate')
def repmres_simulate(parcels):
    return psrcutils.hedonic_simulate("repmres.yaml", parcels, None, "land_value")

# Non-Residential REPM
@orca.step('repmnr_estimate')
def repmnr_estimate(parcels):
    return utils.hedonic_estimate("repmnr.yaml", parcels, None)

@orca.step('repmnr_simulate')
def repmnr_simulate(parcels):
    return psrcutils.hedonic_simulate("repmnr.yaml", parcels, None, "land_value")


# HLCM
@orca.step('hlcm_estimate')
def hlcm_estimate(households, buildings, parcels, zones):
    return utils.lcm_estimate("hlcm.yaml", households, "building_id",
                              buildings, [parcels, zones])

@orca.step('hlcm_simulate')
def hlcm_simulate(households, buildings, parcels, zones):
    return psrcutils.lcm_simulate("hlcm.yaml", households, buildings, [parcels, zones],
                              "building_id", "residential_units", "vacant_residential_units")

# ELCM
@orca.step('elcm_estimate')
def elcm_estimate(jobs, buildings, parcels, zones):
    return utils.lcm_estimate("elcm.yaml", jobs, "building_id",
                              buildings, [parcels, zones])


@orca.step('elcm_simulate')
def elcm_simulate(jobs, buildings, parcels, zones):
    return utils.lcm_simulate("elcm.yaml", jobs, buildings, [parcels, zones],
                              "building_id", "job_spaces", "vacant_job_spaces")


@orca.step('households_relocation')
def households_relocation(households, household_relocation_rates):
    from urbansim.models import relocation as relo
    rm = relo.RelocationModel(household_relocation_rates.to_frame())
    movers = rm.find_movers(households.to_frame())
    print "%s households selected to move." % movers.size
    households.update_col_from_series("building_id",
                            pd.Series(-1, index=movers), cast=True)    

@orca.step('jobs_relocation')
def jobs_relocation(jobs, job_relocation_rates):
    from urbansim.models import relocation as relo
    rm = relo.RelocationModel(job_relocation_rates.to_frame(), rate_column='job_relocation_probability')
    movers = rm.find_movers(jobs.to_frame())
    print "%s jobs selected to move." % movers.size
    jobs.update_col_from_series("building_id",
                            pd.Series(-1, index=movers), cast=True) 


@orca.step('households_transition')
def households_transition(households):
    return utils.simple_transition(households, .05, "building_id")


@orca.step('jobs_transition')
def jobs_transition(jobs):
    return utils.simple_transition(jobs, .05, "building_id")


@orca.step('feasibility')
def feasibility(parcels):
    utils.run_feasibility(parcels,
                          variables.parcel_average_price,
                          variables.parcel_is_allowed,
                          residential_to_yearly=True)


def random_type(form):
    form_to_btype = sim.get_injectable("form_to_btype")
    return random.choice(form_to_btype[form])


def add_extra_columns(df):
    for col in ["residential_sales_price", "non_residential_rent"]:
        df[col] = 0
    return df


@orca.step('residential_developer')
def residential_developer(feasibility, households, buildings, parcels, year):
    utils.run_developer("residential",
                        households,
                        buildings,
                        "residential_units",
                        parcels.parcel_size,
                        parcels.ave_unit_size,
                        parcels.total_units,
                        feasibility,
                        year=year,
                        target_vacancy=.15,
                        form_to_btype_callback=random_type,
                        add_more_columns_callback=add_extra_columns,
                        bldg_sqft_per_job=400.0)


@orca.step('non_residential_developer')
def non_residential_developer(feasibility, jobs, buildings, parcels, year):
    utils.run_developer(["office", "retail", "industrial"],
                        jobs,
                        buildings,
                        "job_spaces",
                        parcels.parcel_size,
                        parcels.ave_unit_size,
                        parcels.total_job_spaces,
                        feasibility,
                        year=year,
                        target_vacancy=.15,
                        form_to_btype_callback=random_type,
                        add_more_columns_callback=add_extra_columns,
                        residential=False,
                        bldg_sqft_per_job=400.0)



