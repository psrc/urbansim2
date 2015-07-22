run = "run_133ref_with_school_models"
run = "run_142.run_2015_07_15_13_39"

year = 2041
out_directory = "output%s%s" % (year, run)
output_file = "out%s%s.h5" % (year, run)
base_directory = os.path.join(os.getenv("OPUS_HOME", "e:/opus"), "data/psrc_parcel/runs", str(run))
base_directory = os.path.join("/Volumes/e$/opus/data/psrc_parcel/runs", str(run))
run_directory = os.path.join(base_directory, str(year))

tables = {"buildings": "building_id",
          "parcels": "parcel_id",
          "zones": "zone_id",
          "households": "household_id",
          "jobs": "job_id",
          "persons": "person_id",
          "fazes": "faz_id",
          "schools": "school_id"}

join_with_coordinates = True
dir_with_coordinates = "/Users/hana/workspace/data/psrc_parcel/datasets" 
parcels_with_coordinates = "parcels_for_google.csv"