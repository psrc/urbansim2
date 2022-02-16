## Data differences between Opus and urbansim-2

A standard Opus cache needs a few changes before converting to hdf5 file. These are:

1. Add a table zoning_heights. This is a post-processed table from MS-SQL table heights, using the script [process_heights.R](https://github.com/psrc/urbansim2/tree/dev/data/zoning) and exported into Opus cache using 

```
python -m opus_core.tools.convert_table csv flt -d . -o  cache_dir_for_urbansim2/year -t zoning_heights
```

2. Make sure that in the tables of control totals there is no city_id attribute if not running in the allocation mode. 

3. In the table annual_household_relocation_rates, rename columns age_min and age_max to age_of_head_min and age_of_head_max.
