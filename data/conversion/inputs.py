run = "run_113"
year = 2010
out_directory = "output%s%s" % (year, run)
output_file = "out%s%s.h5" % (year, run)
run_directory = os.path.join(os.getenv("OPUS_HOME", "e:/opus"), "data/psrc_parcel/runs", str(run), str(year))

tables = {"buildings": "building_id",
          "parcels": "parcel_id",
          "zones": "zone_id",
          "households": "household_id",
          "jobs": "job_id",
          "persons": "person_id",
          "fazes": "faz_id"}