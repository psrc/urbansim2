import pandas as pd
import orca
import numpy as np
from urbansim.utils import misc
import os

def change_store(store_name):
    orca.add_injectable("store",
                       pd.HDFStore(os.path.join(misc.data_dir(),
                                                store_name), mode="r"))


