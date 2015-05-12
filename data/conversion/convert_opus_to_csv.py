import os

execfile('inputs.py')

run_directory = os.path.join(os.getenv("OPUS_HOME", "e:/opus"), "data/psrc_parcel/runs", str(run), str(year))
# convert tables into csv using opus
for table in tables.keys():
   cmd = "python -m opus_core.tools.convert_table flt csv -d %s -o %s -t %s --no-type-info" % (run_directory, out_directory, table)
   os.system(cmd)
   

    
