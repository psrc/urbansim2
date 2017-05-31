import pandas as pd
import orca
import numpy as np
from urbansim.utils import misc
from urbansim_parcels.utils import yaml_to_class, to_frame, check_nas, _print_number_unplaced
import os

def change_store(store_name):
    orca.add_injectable("store",
                       pd.HDFStore(os.path.join(misc.data_dir(),
                                                store_name), mode="r"))

