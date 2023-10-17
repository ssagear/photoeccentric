"""
This is the __init__.py file for photoeccentric
"""

__version__ = "0.2.1"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits
from tqdm import tqdm

from .stellardensity import *
from .eccentricity import *
from .photoeccentric import *
from .lcfitter import *
