import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.constants as c
from tqdm import tqdm
import batman

import emcee
import corner

from .stellardensity import *
from .spectroscopy import *


def normalize_flux(flux_u, flux_err_u):
    """Normalizes flux array."""

    fmed = np.nanmedian(flux_u)
    flux = flux_u/fmed
    flux_err = flux_err_u/fmed

    return flux, flux_err


def bls(time, nflux):
    """Applies astropy Box Least-Squares to light curve to fit period

    Parameters
    ----------
    time: np.array
        Time array
    nflux: np.array
        Flux array


    Returns
    -------
    per_guess: float
        Period fit from BLS
    """

    from astropy.timeseries import BoxLeastSquares
    import astropy.units as u

    mod = BoxLeastSquares(time*u.day, nflux, dy=0.01)
    periodogram = mod.autopower(0.2, objective="snr")
    per_guess = np.asarray(periodogram.period)[int(np.median(np.argmax(periodogram.power)))]

    return per_guess

def get_period_dist(time, flux, pmin, pmax, size):
    """Periods is an array of periods to test."""

    from astropy.timeseries import BoxLeastSquares
    import astropy.units as u

    model = BoxLeastSquares(time*u.day, flux, dy=0.01)

    periods = np.linspace(pmin, pmax, 1000) * u.day
    periodogram = model.power(periods, 0.2)
    normpower = periodogram.power/np.sum(periodogram.power)

    periodPDF = np.random.choice(periods, size=size, p=normpower)

    return periodPDF


def planetlc(time, per, rp, ars, e, inc, w):
    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = 0.                        #time of inferior conjunction
    params.per = per                      #orbital period
    params.rp = rp                        #planet radius (in units of stellar radii)
    params.a = ars                          #semi-major axis (in units of stellar radii)
    params.inc = inc                      #orbital inclination (in degrees)
    params.ecc = e
    params.w = w                          #longitude of periastron (in degrees)
    #params.limb_dark = "linear"
    #params.u = [0.3]
    params.limb_dark = "quadratic"
    params.u = [0.1, 0.3]
    #params.limb_dark = "uniform"
    #params.u = []

    #times to calculate light curve
    m = batman.TransitModel(params, time)

    flux = m.light_curve(params)

    return flux

def planetlc_fitter(time, per, rp, ars, inc):
    """Always assumes e=0.
    w is NOT a free parameter."""

    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = 0                        #time of inferior conjunction
    params.per = per                     #orbital period
    params.rp = rp                        #planet radius (in units of stellar radii)
    params.a = a                          #semi-major axis (in units of stellar radii)
    params.inc = inc                      #orbital inclination (in degrees)
    params.ecc = 0.0                      #eccentricity
    params.w = 0.0                        #longitude of periastron (in degrees)
    #params.limb_dark = "linear"
    #params.u = [0.3]
    params.limb_dark = "quadratic"
    params.u = [0.5, 0.2]
    #params.limb_dark = "uniform"
    #params.u = []

    #times to calculate light curve
    m = batman.TransitModel(params, time)

    flux = m.light_curve(params)

    return flux


def integratedlc(time, per, rp, ars, e, inc, w):

    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = 0                        #time of inferior conjunction
    params.per = per                      #orbital period
    params.rp = rp                        #planet radius (in units of stellar radii)
    params.a = ars                          #semi-major axis (in units of stellar radii)
    params.inc = inc                      #orbital inclination (in degrees)
    params.ecc = e                      #eccentricity
    params.w = w                       #longitude of periastron (in degrees)
    #params.limb_dark = "linear"
    #params.u = [0.3]
    params.limb_dark = "quadratic"
    params.u = [0.5, 0.2]
    #params.limb_dark = "uniform"
    #params.u = []

    #times to calculate light curve
    ptime = get_ptime(time, get_mid(time), 29)

    m = batman.TransitModel(params, ptime)

    pflux = m.light_curve(params)
    flux = array_integrated(pflux, 29)

    return flux

def integratedlc_fitter(time, per, rp, ars, inc):

    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = 0                        #time of inferior conjunction
    params.per = per                      #orbital period
    params.rp = rp                        #planet radius (in units of stellar radii)
    params.a = ars                          #semi-major axis (in units of stellar radii)
    params.inc = inc                      #orbital inclination (in degrees)
    params.ecc = 0.0                      #eccentricity
    params.w = 0.0                       #longitude of periastron (in degrees)
    #params.limb_dark = "linear"
    #params.u = [0.3]
    params.limb_dark = "quadratic"
    params.u = [0.5, 0.2]
    #params.limb_dark = "uniform"
    #params.u = []

    #times to calculate light curve
    ptime = get_ptime(time, get_mid(time), 29)

    m = batman.TransitModel(params, ptime)

    pflux = m.light_curve(params)
    flux = array_integrated(pflux, 29)

    return flux


def get_mid(time):
    """Gets approximately 1/2 of cadence time."""

    return (time[1]-time[0])/2.

def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.nanargmin((np.abs(array - value)))
    return array[idx]

def get_transit_cutout(transitmid, ncadences, time, flux, flux_err):

    """Gets cutout with n cadences before and after transit.
    transitmid and time have same units.

    Parameters
    ----------
    transitmid: float
        Transit mid-time
    ncadences: int
        Number of cadences before and after transit mid-time.
    time: np.array
        Time
    flux: np.array
        Flux

    Returns
    -------
    t1: np.array
        Cutout time
    f1: np.array
        Cutout flux
    fe1: np.array
        Cutout flux error

    """


    tindex = int(np.where(time == find_nearest(time, transitmid))[0])

    t1 = time[tindex-ncadences:tindex+ncadences] - transitmid
    f1 = flux[tindex-ncadences:tindex+ncadences]
    fe1 = flux_err[tindex-ncadences:tindex+ncadences]

    return t1, f1, fe1


def get_transit_cutout_full(transitmids, ncadences, time, flux, flux_err):

    """Removes out of transit data from light curve.
    n cadences before and after transit for full light curve.
    transitmid and time must have the same units.

    Parameters
    ----------
    transitmid: array
        All transit mid-times
    ncadences: int
        Number of cadences before and after transit mid-time.
    time: np.array
        Time
    flux: np.array
        Flux

    Returns
    -------
    t1: np.array
        Cutout time
    f1: np.array
        Cutout flux
    fe1: np.array
        Cutout flux error

    """

    t1 = []
    f1 = []
    fe1 = []

    for tmid in transitmids:

        tindex = int(np.where(time == find_nearest(time, tmid))[0])

        t1.append(time[tindex-ncadences:tindex+ncadences])
        f1.append(flux[tindex-ncadences:tindex+ncadences])
        fe1.append(flux_err[tindex-ncadences:tindex+ncadences])

    t1, f1, fe1 = [np.array(x).flatten() for x in [t1, f1, fe1]]

    return t1, f1, fe1

def get_ptime(time, mid, num):
    """Time, get_mid, number of vals"""

    eti = []

    for i in range(len(time)):
        ettemp = np.linspace(time[i]-mid, time[i]+mid, num, endpoint=True)
        ettemp = list(ettemp)

        eti.append(ettemp)

    ptime = np.array([item for sublist in eti for item in sublist])

    return ptime

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def array_integrated(arr, nint):

    arr = list(divide_chunks(arr, nint))
    arr = np.mean(np.array(arr), axis=1)

    return arr

def mode(dist):

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

    n, bins = np.histogram(dist, bins=np.linspace(np.nanmin(dist), np.nanmax(dist), 100))
    mode = bins[np.nanargmax(n)]
    return mode
