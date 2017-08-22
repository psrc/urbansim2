import orca
import psrc_urbansim.variables
import psrc_urbansim.datasources
import pandas as pd
import re

@orca.step()
def export_variables(parcels):
    attributes = ['building_sqft_pcl', 'residential_units', 'nonres_building_sqft', 'job_capacity', 'land_area', 'parcel_sqft',
                  'number_of_households', 'number_of_jobs', 'land_cost', 'max_dua', 'max_far', 'land_use_type_id', 'number_of_buildings',
                  'zone_id', 'faz_id', 'growth_center_id', 'city_id']
    data = {}
    for attr in attributes:
        data[re.sub('_pcl$', '', attr)] = parcels[attr]
    result_parcels = pd.DataFrame(data, index=parcels.index)
    result_parcels.to_csv("parcels_for_viewer.csv")


# Export parcels attributes from the base year into a csv file
orca.run(['export_variables'], iter_vars=[2014])
