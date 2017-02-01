import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc

def subtract_mean(values):
    return values - values.mean()

def equals(series1, series2):
    return (series1 == series2).astype('int16')

from urbansim.models import regression
from urbansim.models import dcm

regression.subtract_mean = subtract_mean
dcm.equals = equals