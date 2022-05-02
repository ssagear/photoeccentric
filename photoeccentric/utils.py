import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_lc_files(KIC, KICs, lcpath):
    """Gets a list of light curves from a directory."""

    import os

    lclist = []

    for i in range(len(KICs)):
        templst = []
        for subdir, dirs, files in os.walk(lcpath):
            for file in files:
                if str(KICs[i]) in file:
                    templst.append(os.path.join(subdir, file))
        lclist.append(templst)

    files = lclist[int(np.argwhere(KICs==KIC))]
    return files

def get_mid(time):
    """Returns approximately 1/2 of cadence time."""

    return (time[1]-time[0])/2.

def find_nearest(array, value):
    """Gets the nearest element of array to a value."""
    array = np.asarray(array)
    idx = np.nanargmin((np.abs(array - value)))
    return array[idx]


def find_nearest_index(array, value):
    """Gets the index of the nearest element of array to a value."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if len(np.where(array == array[idx])[0]) == 1:
        return int(np.where(array == array[idx])[0])
    else:
        return int(np.where(array == array[idx])[0][0])

def get_sigma_individual(SNR, N, Ntransits, tdepth):
    """Gets size of individual error bar for a certain light curve signal to noise ratio.

    Parameters
    ----------
    SNR: float
        Target light curve signal to noise ratio
    N: int
        Number of in-transit flux points for each transit
    Ntransits: int
        Number of transits in light light curve
    tdepth: float
        Transit depth (Rp/Rs)^2

    Returns
    -------
    sigma_individual: float
        Size of individual flux error bar
    """
    sigma_full = np.sqrt(Ntransits)*(tdepth/SNR)
    sigma_individual = sigma_full*np.sqrt(N)
    return sigma_individual


def get_N_intransit(tdur, cadence):
    """Estimates number of in-transit points for transits in a light curve.

    Parameters
    ----------
    tdur: float
        Full transit duration
    cadence: float
        Cadence/integration time for light curve

    Returns
    -------
    n_intransit: int
        Number of flux points in each transit
    """
    n_intransit = tdur//cadence
    return n_intransit

def mode(dist, window=5, polyorder=2, bin_type='int', bins=25):
    """Gets mode of a histogram.

    Parameters
    ----------
    dist: array
        Distribution

    Returns
    -------
    mode: float
        Mode
    """

    from scipy.signal import savgol_filter

    if bin_type == 'int':
        n, rbins = np.histogram(dist, bins=bins)
    elif bin_type == 'arr':
        n, rbins = np.histogram(dist, bins=bins)

    bin_centers = np.array([np.mean((rbins[i], rbins[i+1])) for i in range(len(rbins)-1)])
    smooth = savgol_filter(n, window, polyorder)
    mode = bin_centers[np.argmax(n)]
    return mode


def get_sigmas(dist):
    """Gets + and - sigmas from a distribution (gaussian or not). Ignores nan values.

    Parameters
    ----------
    dist: np.array
        Distribution from which sigmas are needed

    Returns
    -------
    sigma_minus: float
        - sigma
    sigma_plus: float
        + sigma
    """

    sigma_minus = np.nanpercentile(dist, 50)-np.nanpercentile(dist, 16)
    sigma_plus = np.nanpercentile(dist, 84)-np.nanpercentile(dist, 50)

    return sigma_minus, sigma_plus


def get_e_from_def(g, w):
    """Gets eccentricity from definition (eqn 4)

    Parameters
    ----------
    g: float
        g value
    w: float
        Omega (angle periapse/apoapse)

    Returns
    -------
    e: float
        Eccentricity calculated solely on g and w

    """
    num = np.sqrt(2)*np.sqrt(2*g**4-g**2*np.cos(2*w)-g**2-2*np.sin(w))
    den = 2*(g**2+np.sin(w)**2)
    e = num/den
    return e

def calc_a_from_rho(period, rho_star):
    """Calculate semimajor axis in stellar radii (a/Rs) from orbital period and average stellar density."""

    import scipy.constants as c
    a_rs = (((period*86400.0)**2)*((c.G*rho_star)/(3*c.pi)))**(1./3.)
    return a_rs


def get_cdf(dist, nbins=250):
    """Gets a CDF of a distribution."""

    counts, bin_edges = np.histogram(dist, bins=nbins, range=(np.min(dist), np.max(dist)))
    cdf = np.cumsum(counts)
    cdf = cdf/np.max(cdf)
    return bin_edges[1:], cdf


def get_cdf_val(cdfx, cdfy, val):
    cdfval = cdfy[find_nearest_index(cdfx, val)]
    return cdfval

def get_ppf_val(cdfx, cdfy, val):
    cdfval = cdfx[find_nearest_index(cdfy, val)]
    return cdfval


def calc_r(a_rs, e, w):
    """Calculate r (the planet-star distance) at any point during an eccentric orbit.
    Equation 20 in Murray & Correia Text

    Parameters
    ----------
    a_rs: float
        Semi-major axis (Stellar radius)
    e: float
        Eccentricity
    w: float
        Longitude of periastron (degrees)

    Returns
    -------
    r_rs: float
        Planet-star distance (Stellar radius)
    """

    wrad = w*(np.pi/180.)
    r_rs = (a_rs*(1-e**2))/(1+e*np.cos(wrad-(np.pi/2.)))
    return r_rs
