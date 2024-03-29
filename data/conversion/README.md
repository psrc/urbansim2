## Conversion scripts

Before converting an Opus cache into urbansim-2 format, make sure it has all the [necessary changes](READMEopusdifs.md).

To convert Opus cache into an urbansim-2 hdf5 file, run the script ``cache_to_hdf5.py`` with the first argument being the cache directory and the second argument 
being the name of the output file. For example, to convert the baseyear cache from modelsrv8, do:

```
python cache_to_hdf5.py //MODELSRV8/d$/opusgit/urbansim_data/data/psrc_parcel/base_year_2014_inputs/urbansim2_cache/2014 psrc_base_year_2014.h5
```
To convert Opus estimation cache to urbansim-2 estimation hdf5 file, run the script ''cache_to_hf5.py'' with the first argument being the estimation cache directory, 
the second argument being the name of the output file, and the final argument being a flag to create an estimation data set. 
For exmaple, to convert estimation cache on modelrv8, do:

```
python cache_to_hdf5.py //MODELSRV8/d$/opusgit/urbansim_data/data/psrc_parcel/base_year_2014_inputs/urbansim2_estimation_cache/2014 psrc_estimation_data.h5 --is-estimation
```

To check what was put into the hdf5 file, you can run 

```
python view_hdf5.py psrc_base_year_2014.h5
```

Note that these two scripts were taken from [here](https://github.com/apdjustino/urbansim/tree/master/scripts). The remaining scripts in this directory are not maintained anymore.

To use the other scripts, e.g. ``convert_opus_to_cvs.py``, do the following steps:

1. Set the inputs in inputs.py.
2. Make sure your PYHTONPATH points to the Opus packages.
3. Run ``convert_opus_to_csv.py``
4. Re-set PYTHONPATH to point to the udst packages, e.g. by running d:/udst/setpath.bat.

Results appear in the output directory.
