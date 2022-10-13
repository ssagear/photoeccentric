import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.constants as c
import scipy.special as sc
import batman

import emcee
import corner

from .utils import *
from .stellardensity import *
from .lcfitter import *


def get_b_from_i(inc, a_rs, e, w):
    """Gets impact parameter from inclincation.
    
    Parameters
    ----------
    inc: float
        Inclination (degrees)
    a_rs: float
        Semimajor axis in stellar radii (a/Rstar)
    e: float
        Eccentricity
    w: float
        Longitude of periastron/omega (degrees)

    Returns
    -------
    b: float
        Impact parameter
    """

    g = (1-e**2)/(1+e*np.sin(w*(np.pi/180)))
    b = a_rs*np.cos(inc*(np.pi/180))*g

    return b

def get_i_from_b(b, a_rs, e, w):
    """Gets inclination from impact parameter.
    
    Parameters
    ----------
    b: float
        Impact parameter
    a_rs: float
        Semimajor axis in stellar radii (a/Rstar)
    e: float
        Eccentricity
    w: float
        Longitude of periastron/omega (degrees)

    Returns
    -------
    inc: float
        Inclination (degrees)
    """

    g = (1+e*np.sin(w*(np.pi/180)))/(1-e**2)
    inc = np.arccos(b*(1./a_rs)*g)

    return inc


def get_T14(p, rprs, a_rs, i, e, w):
    """
    Calculates T14 (total transit duration, 1st to 4th contact).
    Assumes a circular orbit (e=0, w=0) if ecc_prior=False.
    If ecc_prior=True, e and w are required. T14 is multiplied by an eccentricity factor.

    Parameters
    ----------
    p: np.array
        Period (days)
    rprs: np.array
        Planet radius/stellar radius
    a_rs: np.array
        Semi-major axis (in stellar radii) (a/Rs)
    i: np.array
        Inclination (degrees)
    ecc: boolean
        Eccentricity taken into account? Default False
    e: float
        Eccentricity if ecc=True, default None
    w: float
        Longitude of periastron (degrees) if ecc=True, default None

    Returns
    -------
    T14: float
        Total transit duration (seconds)
    """


    rs_a = 1.0/a_rs # Rs/a - rstar in units of semimajor axis
    b = get_b_from_i(i, a_rs, e, w)
    print(b)

    T14_circ = (p/np.pi)*np.arcsin(rs_a*(np.sqrt(((1+rprs)**2)-b**2))/np.sin(i*(np.pi/180.0))) #Equation 14 in exoplanet textbook
    chidot = np.sqrt(1-e**2)/(1+e*np.sin(w*(np.pi/180.0))) #Equation 16 in exoplanet textbook
    T14 =  T14_circ*chidot

    return T14

def get_T23(p, rprs, a_rs, i, e, w):
    """
    Calculates T23 (full transit duration, 1st to 4th contact).
    Assumes a circular orbit (e=0, w=0) if ecc_prior=False.
    If ecc_prior=True, e and w are required. T23 is multiplied by an eccentricity factor.

    Parameters
    ----------
    p: np.array
        Period (days)
    rprs: np.array
        Planet radius/stellar radius
    a_rs: np.array
        Semi-major axis (in stellar radii) (a/Rs)
    i: np.array
        Inclination (degrees)
    ecc: boolean
        Eccentricity taken into account? Default False
    e: float
        Eccentricity if ecc=True, default None
    w: float
        Longitude of periastron (degrees) if ecc=True, default None

    Returns
    -------
    T23: float
        Full transit time (seconds)
    """

    rs_a = 1.0/a_rs # Rs/a - rstar in units of semimajor axis
    b = get_b_from_i(i, a_rs, e, w)

    T23_circ = (p/np.pi)*np.arcsin(rs_a*(np.sqrt(((1-rprs)**2)-b**2))/np.sin(i*(np.pi/180.0))) #Equation 14 in exoplanet textbook

    chidot = np.sqrt(1-e**2)/(1+e*np.sin(w*(np.pi/180.0))) #Equation 16 in exoplanet textbook
    T23 = T23_circ*chidot

    return T23

def get_rho_circ(rprs, T14, T23, p):
    """Returns stellar density, assuming a perfectly circular planetary orbit.

    Parameters
    ----------
    rprs: float
        Planet radius/stellar radii
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

    p = p*86400.
    T14 = T14*86400.
    T23 = T23*86400.

    if T14 >= T23:
        rho_circ = (((2*(delta**(0.25)))/np.sqrt(T14**2-T23**2))**3)*((3*p)/(c.G*(c.pi**2)))
    else:
        rho_circ = np.nan

    return rho_circ

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

def get_g_distribution(rhos, per_dist, rprs_dist, T14_dist, T23_dist):
    """Gets g distribution for a KOI.

    Parameters
    ----------
    rhos: np.array
        Density histogram
    per_dist: np.array
        Best-fit period (days)
    rprs_dist: np.array
        Best-fit rp/rs
    T14_dist: np.array
        Total transit duration (seconds) calculated from best-fit planet parameters
    T23_dist: np.array
        Full transit duration (seconds) calculated from best-fit planet parameters

    Returns
    -------
    gs: np.array
        g distribution for star/planet.
    rho_circ: np.array
        Density distribution assuming a circular orbit
    """

    gs = np.zeros((len(rhos)))
    rho_circ = np.zeros(len(rhos))

    #for element in histogram for star:
    for j in range(len(rhos)):

        per_dist[j] = per_dist[j]#*86400.

        rho_circ[j] = get_rho_circ(rprs_dist[j], T14_dist[j], T23_dist[j], per_dist[j])

        g = get_g(rho_circ[j], rhos[j])
        gs[j] = g

    return gs, rho_circ

def get_inclination(b, a_rs):
    """Get inclination (in degrees) from an impact parameter and semi-major axis (on stellar radius).

    Parameters
    ----------
    b: float
        Impact parameter
    a_rs: float
        Ratio of semi-major axis to stellar radius (a/Rs)

    Returns
    -------
    i: float
        Inclination (degrees)
    """

    i_radians = np.arccos(b*(1./a_rs))
    i = i_radians*(180./np.pi)
    return i

def stellar_params_from_archive(df, kep_name):
    """Get stellar parameters for the host of a KOI from exoplanet archive (downloaded data).

    Parameters
    ----------
    df: pandas.DataFrame
        dataframe of exop. archive downloaded data
    kep_name: str
        Kepler name of planet

    Returns
    -------
    smass: float
        Stellar mass (solar mass)
    srad: float
        Stellar radius (solar radius)
    limbdark_mod: str
        Limb darkening model
    ldm_c1: float
        Limb darkening coefficient 1
    ldm_c2: float
        Limb darkening coefficient 2

    """
    smass = float(df.loc[df['kepler_name'] == kep_name].koi_smass) #stellar mass (/solar mass)
    smass_uerr = float(df.loc[df['kepler_name'] == kep_name].koi_smass_err1) #stellar mass upper error
    smass_lerr = float(df.loc[df['kepler_name'] == kep_name].koi_smass_err2) #stellar mass lower error

    srad = float(df.loc[df['kepler_name'] == kep_name].koi_srad) #stellar radius (/solar radius)
    srad_uerr = float(df.loc[df['kepler_name'] == kep_name].koi_srad_err1) #stellar radius upper error
    srad_lerr = float(df.loc[df['kepler_name'] == kep_name].koi_srad_err2) #stellar radius lower error

    limbdark_mod = str(df.loc[df['kepler_name'] == kep_name].koi_limbdark_mod) #LDM Model
    ldm_c2 = float(df.loc[df['kepler_name'] == kep_name].koi_ldm_coeff2) #LDM coef 2
    ldm_c1 = float(df.loc[df['kepler_name'] == kep_name].koi_ldm_coeff1) #LDM coef 1

    return smass, smass_uerr, smass_lerr, srad, srad_uerr, srad_lerr, limbdark_mod, ldm_c1, ldm_c2


def zscore(dat, mean, sigma):
    """Calculates zscore of a data point in (or outside of) a dataset
    zscore: how many sigmas away is a value from the mean of a dataset?

    Parameters
    ----------
    dat: float
        Data point
    mean: float
        Mean of dataset
    sigma: flaot
        Sigma of dataset

    """
    zsc = (dat-mean)/sigma
    return zsc

def get_a_rs(rhos, periods):
    """Gets a/Rs guess based on orbital period, density & Kepler's 3rd law

    Parameters
    ----------
    rhos: np.array
        Stellar density array
    periods: np.array
        Periods array

    Returns
    -------
    a_rs: np.array
        a/Rs array calculated from periods, rhos

    """

    a_rs = np.zeros(len(rhos))

    for i in range(len(rhos)):
        per_iter = periods[i]*86400
        rho_iter = rhos[i]

        a_rs[i] = ((per_iter**2)*((c.G*rho_iter)/(3*np.pi)))**(1.0/3.0)

    return a_rs
