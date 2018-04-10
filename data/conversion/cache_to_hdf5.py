#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


def cache_to_df(dir_path):
    """
    Convert a directory of binary array data files to a Pandas DataFrame.
    Parameters
    ----------
    dir_path : str
    """
    table = {}
    for attrib in glob.glob(os.path.join(dir_path, '*')):
        attrib_name, attrib_ext = os.path.splitext(os.path.basename(attrib))
        if attrib_ext == '.lf8':
            attrib_data = np.fromfile(attrib, np.float64)
            table[attrib_name] = attrib_data

        elif attrib_ext == '.lf4':
            attrib_data = np.fromfile(attrib, np.float32)
            table[attrib_name] = attrib_data

        elif attrib_ext == '.li2':
            attrib_data = np.fromfile(attrib, np.int16)
            table[attrib_name] = attrib_data

        elif attrib_ext == '.li4':
            attrib_data = np.fromfile(attrib, np.int32)
            table[attrib_name] = attrib_data

        elif attrib_ext == '.li8':
            attrib_data = np.fromfile(attrib, np.int64)
            table[attrib_name] = attrib_data

        elif attrib_ext == '.ib1' or attrib_ext == '.ii1':
            attrib_data = np.fromfile(attrib, np.bool_)
            table[attrib_name] = attrib_data

        elif attrib_ext.startswith('.iS'):
            length_string = int(attrib_ext[3:])
            attrib_data = np.fromfile(attrib, ('a' + str(length_string)))
            table[attrib_name] = attrib_data

        else:
            print('Array {} is not a recognized data type'.format(attrib))

    df = pd.DataFrame(table)
    return df


DIRECTORIES = {
    'annual_employment_control_totals', 'annual_household_control_totals',
    'annual_household_relocation_rates', 'annual_job_relocation_rates',
    'buildings', 'building_sqft_per_job', 'building_types', 'counties',
    'development_constraints', 'development_event_history',
    'development_templates', 'development_template_components',
    'employment_adhoc_sector_group_definitions',
    'employment_adhoc_sector_groups', 'employment_sectors',
    'fazes', 'gridcells', 'land_use_types', 'jobs', 'households',
    'parcels', 'persons', 'schools', 'target_vacancies', 'travel_data',
    'zones', 'zoning_heights', 'jobs_for_estimation', 'persons_for_estimation'
}

NO_INDEX = ['annual_household_relocation_rates', 'annual_job_relocation_rates']


def convert_dirs(base_dir, hdf_name, is_estimation=False,
                 no_compress=False):
    """
    Convert nested set of directories to
    """
    print('Converting directories in {}'.format(base_dir))

    dirs = glob.glob(os.path.join(base_dir,  '*'))
    dirs = {d for d in dirs if os.path.basename(d) in DIRECTORIES}
    if not dirs:
        raise RuntimeError('No directories found matching known data.')

    # Only disable zlib-standard compression if user explicitly says so
    if no_compress:
        complib = None
    else:
        complib = 'zlib'

    store = pd.HDFStore(
        hdf_name, mode='w', complevel=1, complib=complib)

    if is_estimation:
        dirpath = os.path.join(base_dir, '/households_for_estimation')
        households_for_estimation_df = cache_to_df(dirpath)
        households_for_estimation_df['for_estimation'] = 1

    for dirpath in dirs:
        dirname = os.path.basename(dirpath)

        print(dirname)
        df = cache_to_df(dirpath)

        if dirname == 'households':
            # If estimation, append households_for_estimation records to households
            if is_estimation:
                for col in households_for_estimation_df.columns:
                    if col not in df.columns:
                        df[col] = 0
                df = pd.concat([df, households_for_estimation_df])
            keys = ['household_id']

        elif dirname == 'persons_for_estimation':
            keys = ['person_id']

        elif dirname == 'travel_data':
            keys = ['from_zone_id', 'to_zone_id']

        elif dirname == 'annual_employment_control_totals':
            if 'city_id' in df.columns:
                df = df.drop(columns=['city_id'])
            keys = ['year']

        elif dirname == 'annual_household_control_totals':
            df.income_max = np.where(df.income_max <> -1,
                                     df.income_max + 1, df.income_max)
            df.workers_max = np.where(df.workers_max > 0,
                                      df.workers_max + .5, df.workers_max)
            df.workers_max = np.where(df.workers_max == 0,
                                      .5, df.workers_max)
            df.persons_max = np.where(df.persons_max > 0,
                                      df.persons_max + .5, df.persons_max)
            #df = df.drop(columns=['city_id'])
            keys = ['year']

        elif dirname == 'annual_household_relocation_rates':
            df = df.rename(columns={'age_min': 'age_of_head_min',
                                    'age_max': 'age_of_head_max'})

        elif dirname == 'building_sqft_per_job':
            keys = ['zone_id', 'building_type_id']

        elif dirname == 'counties':
            keys = ['county_id']

        elif dirname == 'fazes':
            keys = ['faz_id']

        elif dirname == 'development_constraints':
            keys = ['constraint_id']

        elif dirname == 'development_event_history':
            keys = ['building_id']

        elif dirname == 'target_vacancies':
            keys = ['building_type_id', 'year']

        elif dirname == 'gridcells':
            keys = ['grid_id']

        elif dirname == 'employment_adhoc_sector_groups':
            keys = ['group_id']

        elif dirname == 'employment_sectors':
            keys = ['sector_id']

        elif dirname == 'employment_adhoc_sector_group_definitions':
            keys = ['sector_id', 'group_id']

        elif dirname == 'development_templates':
            keys = ['template_id']

        elif dirname == 'development_template_components':
            keys = ['component_id']

        elif dirname == 'zoning_heights':
            keys = ['plan_type_id']

        elif dirname == 'households_for_estimation':
                keys = ['household_id']
                households_df = df

        elif dirname == 'jobs_for_estimation':
                keys = ['job_id']

        elif dirname == 'buildings':
            keys = ['building_id']
            if is_estimation:
                df.ix[df['building_type_id'] == 0, 'building_type_id'] = 1

        else:
            keys = [dirname[:-1] + '_id']

        if dirname not in NO_INDEX:
            df = df.set_index(keys)

        for colname in df.columns:
            if df[colname].dtype == np.float64:
                df[colname] = df[colname].astype(np.float32)
            elif df[colname].dtype == np.int64:
                df[colname] = df[colname].astype(np.int32)
            else:
                df[colname] = df[colname]

        df.info()
        print(os.linesep)
        store.put(dirname, df)

    store.close()


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=(
            'Convert nested set of directories containing binary '
            'array data to an HDF5 file made from Pandas DataFrames.'))
    parser.add_argument('base_dir', help='Base path for conversion.')
    parser.add_argument('hdf_name', help='Name of output HDF5 file.')
    parser.add_argument('--is-estimation',
                        help=('Importing from estimation cache'),
                        default=False, dest='no_estimation',
                        action='store_true')
    parser.add_argument('--no-compress',
                        help=('Disable output file compression'),
                        default=False, dest='no_compress', action='store_true')
    return parser.parse_args(args)


def main(args=None):
    #base_dir = r'\\MODELSRV8\d$\opusgit\urbansim_data\data\psrc_parcel\base_year_2014_inputs\urbansim2_estimation_cache'
    #year = '2014'
    #lag_year = '2009'
    #is_estimation = True
    #hdf_name = r'D:\udst\psrc_urbansim\data\test.h5'
    #args = ["//modelsrv3/e$/opusgit/urbansim_data/data/psrc_parcel/SoundCast/Estimation/", "D:/udst/psrc_urbansim/data/test_.h5", ["2000", "2009", "2014"], "--is-estimation"]
    #args = ['//MODELSRV8/d$/opusgit/urbansim_data/data/psrc_parcel/base_year_2014_inputs/urbansim2_cache/2014', "D:/udst/psrc_urbansim/data/reg_cache_.h5"]
    args = [r'\\MODELSRV8\d$\opusgit\urbansim_data\data\psrc_parcel\base_year_2014_inputs\urbansim2_estimation_cache\2014', "D:/udst/psrc_urbansim/data/est_cache_.h5", "--is-estimation"]
    args = parse_args(args)
    convert_dirs(args.base_dir, args.hdf_name, args.no_estimation)


if __name__ == '__main__':
    sys.exit(main())