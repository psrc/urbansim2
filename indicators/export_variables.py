import os
import yaml
import orca
from urbansim.utils import misc
import psrc_urbansim.variables
import psrc_urbansim.datasources
import pandas as pd
import re

@orca.injectable('input_file')
def input_file():
    return "psrc_base_year_2023_alloc_py3.h5"


@orca.injectable('settings', cache=True)
def settings(): # needed to copy this from urbansim_defaults as the yaml.load call needs an additional argument 
    with open(os.path.join(misc.configs_dir(), "settings.yaml")) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        settings["store"] = input_file() # overwrite the input file
        # monkey patch on the settings object since it's pretty global
        # but will also be available as injectable
        orca.settings = settings
        return settings


@orca.step()
def export_variables(parcels):
    #attributes = ['building_sqft_pcl', 'residential_units', 'nonres_building_sqft', 'job_capacity', 'land_area', 'parcel_sqft',
    #              'number_of_households', 'number_of_jobs', 'land_cost', 'max_dua', 'max_far', 'land_use_type_id', 'number_of_buildings',
    #              'zone_id', 'faz_id', 'growth_center_id', 'city_id', 'subreg_id']
    attributes = ['parcel_sqft', 'growth_center_id', 'faz_id', 'city_id', 'subreg_id', 'target_id', 'control_id', 'control_hct_id',
                  'county_id', 'tod_id', 'zone_id', 'plan_type_id', 'census_tract_id']    
    data = {}
    for attr in attributes:
        data[re.sub('_pcl$', '', attr)] = parcels[attr]
    result_parcels = pd.DataFrame(data, index=parcels.index)
    #result_parcels.to_csv("parcels_for_viewer.csv")
    result_parcels.to_csv("csv_store/parcels_geos.csv")


# Export parcels attributes from the base year into a csv file
orca.run(['export_variables'], iter_vars=[2018])
