import urbansim.sim.simulation as sim
import os
import pandas as pd
from urbansim.utils import misc


sim.add_injectable("store", pd.HDFStore(os.path.join(misc.data_dir(),
                                                     "base_year.h5"),
                                        mode="r"))


