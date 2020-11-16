import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
import glob


def asymmetric_gaussian(mean, sigma_minus, sigma_plus, nvals):
    """Generates an asymmetric Gaussian distribution based on a mean and 2 different sigmas (one (-) and one (+))
    Made by generating 2 symmetric Gaussians with different sigmas and sticking them together at the mean.
    The integral of the resulting Gaussian is 1.

    Parameters
    ----------
    mean: float
        Mean of distribution
    sigma_minus: float
        Negative sigma of distribtion
    sigma_plus: float
        Positive sigma of distribtion
    nvals: int
        Number of values

    Results
    -------
    dist: np.ndarray
        Asymmetric Gaussian distribution, length nvals
    """

    left = np.random.normal(mean, abs(sigma_minus), nvals+200)
    right = np.random.normal(mean, abs(sigma_plus), nvals+200)
    dist_left = left[left<mean]
    dist_right = right[right>=mean]
    dist = np.concatenate((dist_right, dist_left), axis=None)

    np.random.shuffle(dist)

    while len(dist) > nvals:
        dist = np.delete(dist, [np.random.randint(0, len(dist)-1)])
    else:
        return dist
