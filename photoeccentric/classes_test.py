"""Ignore this for now
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.constants as c
from tqdm import tqdm
import batman

from .stellardensity import *
from .spectroscopy import *
from .photoeccentric import *


class CircTransitFit:

    def __init__(self, period=0, rprs=0, a_rs=0, inc=0, T14=0, T23=0):
        self.period = period
        self.rprs = rprs
        self.a_rs = a_rs
        self.inc = inc

        self.T14 = T14
        self.T23 = T23

    def get_data(self):
        print(f'{self.period}+{self.rprs}+{self.a_rs}+{self.inc}j')
