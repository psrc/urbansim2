run = "run_113"
year = 2010
out_directory = "output"
output_file = "out%s%s.h5" % (year, run)

tables = {"buildings": "building_id",
          "parcels": "parcel_id",
          "zones": "zone_id",
          "households": "household_id",
          "jobs": "job_id",
          "persons": "person_id",
          "fazes": "faz_id"}