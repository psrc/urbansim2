import pandas as pd
import urbansim.sim.simulation as sim
from urbansim.utils import misc
import os

def change_store(store_name):
    sim.add_injectable("store",
                       pd.HDFStore(os.path.join(misc.data_dir(),
                                                store_name), mode="r"))