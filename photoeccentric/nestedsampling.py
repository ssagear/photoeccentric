import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.constants as c
from tqdm import tqdm
import batman

import emcee
import corner

import dynesty

from .stellardensity import *
from .spectroscopy import *
from .lcfitter import *


def tfit_loglike(theta):
    """
    Transit fit dynesty function

    model = ph.integratedlc_fitter()
    gerr = sigma of g distribution

    """

    per, rp, a, inc, t0 = theta

    model = ph.integratedlc_fitter(time, per, rp, a, inc, t0)
    sigma2 = flux_err ** 2

    return -0.5 * np.sum((flux - model) ** 2 / sigma2 + np.log(sigma2))


def tfit_prior_transform(utheta):
    """Transforms samples `u` drawn from the unit cube to samples to those
    from our uniform prior within [-10., 10.) for each variable."""

    uper, urp, ua, uinc, ut0 = utheta

    per = 3.*uper+3.
    rp = urp
    a = ua*15.+20.
    inc = uinc*3.+87.
    t0 = 2.*ut0-1.

    return per, rp, a, inc, t0


def tfit_Beta_prior_transform(utheta):
    """Transforms samples `u` drawn from the unit cube to samples to those
    from our uniform prior within [-10., 10.) for each variable."""

    uper, urp, ua, uinc, ut0 = utheta

    per = 3.*uper+3.
    rp = urp
    a = ua*15.+20.
    inc = uinc*3.+87.
    t0 = 2.*ut0-1.

    return per, rp, a, inc, t0
