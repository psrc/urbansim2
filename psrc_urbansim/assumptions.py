import orca
import os
import pandas as pd
from urbansim.utils import misc


orca.add_injectable("store", pd.HDFStore(os.path.join(misc.data_dir(),
                                                     "psrc_base_year_2014.h5"),
                                        mode="r"))


