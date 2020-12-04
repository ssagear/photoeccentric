import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.constants as c
from tqdm import tqdm
import PyAstronomy.pyasl as pya

import emcee
import corner

from .stellardensity import *
from .spectroscopy import *


def get_T23(p, rp_earth, rs, T14, a, i):

    ing_eg = 2*pya.ingressDuration(a, rp_earth*11.2, rs, i, p)#rp needs to in jovian radii
    T23 = T14-ing_eg

    return T23#hours

def get_T23_errs(T23_dist):

    x, cdf = mt.get_cdf(T23_dist)
    T23_sigma_minus = mt.find_sigma(x, cdf, "-")
    T23_sigma_plus = mt.find_sigma(x, cdf, "+")

    return T23_sigma_minus, T23_sigma_plus


def get_planet_params(p, T14, T23):
    """Returns planet parameters in correct units.

    Parameters
    ----------
    p: float
        Planet orbital period (days)
    rp_earth: float
        Planet radius (earth radii)
    rs: float
        Stellar radius (solar radii)
    T14: float
        Total transit time - first to fourth contact (hours)
    a: float
        Planet semi-major axis (AU)
    i: float
        Orbital inclination (degrees)

    Returns
    -------
    p_seconds: float
        Orbital period (seconds)
    rprs: float
        Planet radius (stellar host radii)
    T14_seconds: float
        Total transit time - first to fourth contact (seconds)
    T23_seconds: float
        Full transit time - second to third contact (seconds)
    """

    p_seconds = p*86400
    T14_seconds = T14*3600
    T23_seconds = T23*3600

    return p_seconds, T14_seconds, T23_seconds


def get_rho_circ(rprs, T14, T23, p):
    """Returns stellar density, assuming a perfectly circular planetary orbit.

    Parameters
    ----------
    rprs: float
        Planet radius (stellar host radii)
    T14: float
        Total transit time - first to fourth contact (seconds)
    T23: float
        Full transit time - second to third contact (seconds)
    p: float
        Orbital period (seconds)

    Returns
    -------
    rho_circ: float
        Stellar density, assuming a circular orbit (kg/m^3)
    """

    delta = rprs**2
    num1 = 2*(delta**(0.25))
    den1 = np.sqrt((T14**2)-(T23**2))
    term1 = (num1/den1)**3

    num2 = 3*p
    den2 = c.G*(c.pi**2)
    term2 = num2/den2

    rho_circ = term1*term2

    return rho_circ


def get_g(rho_circ, rho_star):
    """Gets g

    Parameters
    ----------
    rho_circ: float
        Stellar density, assuming a circular orbit (kg/m^3)
    rho_star: float
        Stellar density, calculated from Kepler/Gaia/spectroscopy (kg/m^3)

    Returns
    -------
    g: float
        Cube root of ratio between rho_circ and rho_star
    """
    g = np.cbrt(rho_circ/rho_star)
    return g

def get_g_from_def(e, w):
    """Gets g from e and omega

    Parameters
    ----------
    e: float
        Eccentricity
    w: float
        Angle of periapse or something

    Returns
    -------
    g: float
        Cube root of ratio between rho_circ and rho_star
    """
    g = (1+e*np.sin(w))/np.sqrt(1-e**2)
    return g


def get_e(g, w):
    """Gets eccentricity (from photoeccentric effect)

    Parameters
    ----------
    g: float
        Cube root of ratio between rho_circ and rho_star
    w: float
        Angle of apoapse or periapse (?) (degrees, -90 < w < 90)

    Returns
    -------
    e: float
        Eccentricity of planet orbit
    """
    e = (np.sqrt(2)*(np.sqrt(2*g**4 - g**2*np.cos(2*w) - g**2 - 2*np.sin(w))))/(2*(g**2 + np.sin(w)**2))
    return e

def row_to_top(df, index):
    """Bring row to top

    Parameters
    ----------
    df: pandas.dataframe
        Dataframe to copy
    index: int
        Index of row to bring to top

    Returns
    -------
    df_cp: pandas.dataframe
        Copy of dataframe with specified row at top

    """
    df_cp = pd.concat([df.iloc[[index],:], df.drop(index, axis=0)], axis=0)
    return df_cp


def get_g_distribution(row, n_rhos):
    """Gets g distribution for a KOI.

    Parameters
    ----------
    row: int
        Row in pandas.dataframe of info from Exoplanet Archive. (change this to take KIC/KOI)
    n_rhos: int
        Number of values in distribution

    Returns
    -------
    gs: np.array
        g distribution for star/planet.
    """

    targ = spectplanets.iloc[row]
    print('KIC: ', targ.kepid)

    rhos = rho_lum[str(targ.kepid)].dropna()
    rhos = np.array(rhos)

    while len(rhos) > n_rhos:
        rhos = np.delete(rhos, [np.random.randint(0, len(rhos))])

    #ws = np.arange(-90., 300., 1.)

    gs = np.zeros((len(rhos)))
    #es = np.zeros(len(rhos))
    #es = np.zeros((len(ws), len(rhos)))

    rho_circ = np.zeros(len(rhos))
    rho_ratios = np.zeros(len(rhos))
    T23_dist = np.zeros((len(rhos)))

    per_dist = mt.asymmetric_gaussian(targ.koi_period, targ.koi_period_err1, targ.koi_period_err2, len(rhos))

    rs_dist = mt.asymmetric_gaussian(targ.koi_srad, targ.koi_srad_err1, targ.koi_srad_err2, len(rhos))
    rp_earth_dist = mt.asymmetric_gaussian(targ.koi_prad, targ.koi_prad_err1, targ.koi_prad_err2, len(rhos))
    rprs_dist = mt.asymmetric_gaussian(targ.koi_ror, targ.koi_ror_err1, targ.koi_ror_err2, len(rhos))

    T14_dist = mt.asymmetric_gaussian(targ.koi_duration, targ.koi_duration_err1, targ.koi_duration_err2, len(rhos))

    a = targ.koi_sma
    i = targ.koi_incl

    for j in tqdm(range(len(rhos))): #for element in histogram for star:
        T23_dist[j] = get_T23(per_dist[j], rp_earth_dist[j], rs_dist[j], T14_dist[j], a, i)

        p_seconds, T14_seconds, T23_seconds = get_planet_params(per_dist[j], T14_dist[j], T23_dist[j])
        rho_circ[j] = get_rho_circ(rprs_dist[j], T14_seconds, T23_seconds, p_seconds)

        rho_ratios[j] = rho_circ[j]/rhos[j]
        g = get_g(rho_circ[j], rhos[j])
        gs[j] = g

    return gs


def get_sigmas(dist):
    """Gets + and - sigmas from a distribution (gaussian or not) through a cdf

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
    x, cdf = get_cdf(dist)
    sigma_minus = find_sigma(x, cdf, "-")
    sigma_plus = ph.find_sigma(x, cdf, "+")

    return sigma_minus, sigma_plus

def get_e_from_def(g, w):
    """Gets eccentricity from definition (eqn 4, not really the definition tho)

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


def log_likelihood(theta, g, gerr):
    """Log of likelihood
    model = g(e,w)
    gerr = sigma of g distribution
    """
    w, e = theta
    model = (1+e*np.sin(w))/np.sqrt(1-e**2)
    sigma2 = gerr ** 2
    return -0.5 * np.sum((g - model) ** 2 / sigma2 + np.log(sigma2))


def log_prior(theta):
    """Log of prior
    e between 0 and 1
    w between -90 and 300
    """
    w, e = theta
    if 0.0 < e < 1.0 and -90.0 < w < 300.0:
        return 0.0
    return -np.inf


def log_probability(theta, g, gerr):
    """Log of probability
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, g, gerr)
