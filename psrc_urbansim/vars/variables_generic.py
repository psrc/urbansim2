import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc

def subtract_mean(values):
    return values - values.mean()


from urbansim.models import regression
regression.subtract_mean = subtract_mean
