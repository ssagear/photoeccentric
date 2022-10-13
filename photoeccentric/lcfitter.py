import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .utils import *
from .stellardensity import *




def do_linfit(time, flux, flux_err, transitmid, nbuffer, nlinfit, cadence=0.0201389, odd=False):

    """For a segment of a Kepler light curve with a transit,
    fit a line to the out-of-transit data and subtract.

    Parameters:
    ----------
    time: np.array
        Time array of entire light curve
    flux: np.array
        Flux array of entire light curve (normalized to 1)
    flux_err: np.array
        Flux error array of entire light curve (normalized to 1)
    transitmid: float
        Mid-time of transit (same units as time: BJD, BJD-X, etc.)
    nbuffer: int
        Number of out-of-transit data points to keep before and after transit
    nlinfit: int
        Number of out-of-transit data points to use in linear fit (from each end of cutout light curve)

    Returns:
    -------
    t1bjd: np.array
        Time cutout in BJD
    fnorm: np.array
        Flux cutout - linear fit
    fe1: np.array
        Flux error cutout

    """

    nbuffer = int(nbuffer)
    nlinfit = int(nlinfit)

    # Find closest Kepler time stamp to transit mid-timess
    tindex = int(np.where(time == find_nearest(time, transitmid))[0])
    # Time array of cutout: phase 0 at tmid
    t1 = np.array(time[int(tindex-nbuffer):int(tindex+nbuffer+1)]) - transitmid
    # Time array of cutout: BJD
    t1bjd = np.array(time[int(tindex-nbuffer):int(tindex+nbuffer+1)])


    # Flux array of cutout
    f1 = np.array(flux[int(tindex-nbuffer):int(tindex+nbuffer+1)])
    # Flux error array of cutout
    fe1 = np.array(flux_err[int(tindex-nbuffer):int(tindex+nbuffer+1)])


    if np.any(np.array([j-i for i, j in zip(t1bjd[:-1], t1bjd[1:])]) > 5*cadence): # If the midpoint lies during a gap in the data
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    if odd==False:
        # Do linear fit to OOT data
        idx = np.isfinite(t1) & np.isfinite(f1)
        m, b = np.polyfit(np.concatenate((t1[idx][:nlinfit], t1[idx][-nlinfit:])), np.concatenate((f1[idx][:nlinfit], f1[idx][-nlinfit:])), 1)

    if odd==True:
        # Do linear fit to OOT data
        idx = np.isfinite(t1) & np.isfinite(f1)
        m, b = np.polyfit(np.concatenate((t1[idx][:nlinfit-1], t1[idx][-nlinfit:])), np.concatenate((f1[idx][:nlinfit-1], f1[idx][-nlinfit:])), 1)


    # Subtract linear fit from LC
    linfit = m*t1 + b
    #plt.cla
    #plt.errorbar(t1, f1, yerr=fe1)
    #plt.plot(t1, linfit)
    #plt.show()
    #fnorm = (f1-linfit)+1
    fnorm = f1/linfit

    #plt.cla()
    #plt.errorbar(t1, fnorm, yerr=fe1)
    #plt.show()

    return m, b, t1bjd, t1, fnorm, fe1


def do_cubfit(time, flux, flux_err, transitmid, nbuffer, nlinfit, cadence=0.0201389, odd=False, custom_data=None, midpoint=1):

    """For a segment of a Kepler light curve with a transit,
    fit a cubic polynomial to the out-of-transit data and subtract.

    Parameters:
    ----------
    time: np.array
        Time array of entire light curve
    flux: np.array
        Flux array of entire light curve (normalized to 1)
    flux_err: np.array
        Flux error array of entire light curve (normalized to 1)
    transitmid: float
        Mid-time of transit (same units as time: BJD, BJD-X, etc.)
    nbuffer: int
        Number of out-of-transit data points to keep before and after transit
    nlinfit: int
        Number of out-of-transit data points to use in cubic fit (from each end of cutout light curve)
    custom_data: custom data to fit to. should have same time array.

    Returns:
    -------
    t1bjd: np.array
        Time cutout in BJD
    fnorm: np.array
        Flux cutout - linear fit
    fe1: np.array
        Flux error cutout

    """

    nbuffer = int(nbuffer)
    nlinfit = int(nlinfit)

    if custom_data is None:

        # Find closest Kepler time stamp to transit mid-timess
        tindex = int(np.where(time == find_nearest(time, transitmid))[0])
        # Time array of cutout: phase 0 at tmid
        t1 = np.array(time[int(tindex-nbuffer):int(tindex+nbuffer+1)]) - transitmid
        # Time array of cutout: BJD
        t1bjd = np.array(time[int(tindex-nbuffer):int(tindex+nbuffer+1)])


        # Flux array of cutout
        f1 = np.array(flux[int(tindex-nbuffer):int(tindex+nbuffer+1)])
        # Flux error array of cutout
        fe1 = np.array(flux_err[int(tindex-nbuffer):int(tindex+nbuffer+1)])


        if np.any(np.array([j-i for i, j in zip(t1bjd[:-1], t1bjd[1:])]) > 5*cadence): # If the midpoint lies during a gap in the data
            return np.nan, np.nan, np.nan, np.nan, np.nan

        if odd==False:
            # Do linear fit to OOT data
            idx = np.isfinite(t1) & np.isfinite(f1)
            z = np.polyfit(np.concatenate((t1[idx][:nlinfit], t1[idx][-nlinfit:])), np.concatenate((f1[idx][:nlinfit], f1[idx][-nlinfit:])), 3)

        if odd==True:
            # Do linear fit to OOT data
            idx = np.isfinite(t1) & np.isfinite(f1)
            z = np.polyfit(np.concatenate((t1[idx][:nlinfit-1], t1[idx][-nlinfit:])), np.concatenate((f1[idx][:nlinfit-1], f1[idx][-nlinfit:])), 3)

        # Subtract linear fit from LC
        cubfit = np.poly1d(z)
        fnorm = f1/cubfit(t1)

    else:

        # Find closest Kepler time stamp to transit mid-timess
        tindex = int(np.where(time == find_nearest(time, transitmid))[0])
        # Time array of cutout: phase 0 at tmid
        t1 = np.array(time[int(tindex-nbuffer):int(tindex+nbuffer+1)]) - transitmid
        # Time array of cutout: BJD
        t1bjd = np.array(time[int(tindex-nbuffer):int(tindex+nbuffer+1)])


        # Flux array of cutout
        f1 = np.array(flux[int(tindex-nbuffer):int(tindex+nbuffer+1)])
        c1 = np.array(custom_data[int(tindex-nbuffer):int(tindex+nbuffer+1)])
        # Flux error array of cutout
        fe1 = np.array(flux_err[int(tindex-nbuffer):int(tindex+nbuffer+1)])


        if np.any(np.array([j-i for i, j in zip(t1bjd[:-1], t1bjd[1:])]) > 4*cadence): # If the midpoint lies during a gap in the data
            return np.nan, np.nan, np.nan, np.nan, np.nan

        if odd==False:
            # Do linear fit to OOT data
            idx = np.isfinite(t1) & np.isfinite(f1)
            z = np.polyfit(np.concatenate((t1[idx][:nlinfit], t1[idx][-nlinfit:])), np.concatenate((c1[idx][:nlinfit], c1[idx][-nlinfit:])), 3)

        if odd==True:
            # Do linear fit to OOT data
            idx = np.isfinite(t1) & np.isfinite(f1)
            z = np.polyfit(np.concatenate((t1[idx][:nlinfit-1], t1[idx][-nlinfit:])), np.concatenate((c1[idx][:nlinfit-1], c1[idx][-nlinfit:])), 3)


        # Subtract linear fit from LC
        cubfit = np.poly1d(z)
        fnorm = f1/cubfit(t1)

    return z, t1bjd, t1, fnorm, fe1




def cutout_no_linfit(time, flux, flux_err, transitmid, nbuffer, cadence=0.0208333):
    """For a segment of a Kepler light curve with a transit,
    fit a line to the out-of-transit data and subtract.

    Parameters:
    ----------
    time: np.array
        Time array of entire light curve
    flux: np.array
        Flux array of entire light curve (normalized to 1)
    flux_err: np.array
        Flux error array of entire light curve (normalized to 1)
    transitmid: float
        Mid-time of transit (same units as time: BJD, BJD-X, etc.)

    Returns:
    -------
    t1bjd: np.array
        Time cutout in BJD
    fnorm: np.array
        Flux cutout - linear fit
    fe1: np.array
        Flux error cutout

    """

    #print('Cutout start')
    nbuffer = int(nbuffer)

    # Find closest Kepler time stamp to transit mid-timess
    tindex = int(np.where(time == find_nearest(time, transitmid))[0])
    # Time array of cutout: phase 0 at tmid
    t1 = np.array(time[int(tindex-nbuffer):int(tindex+nbuffer+1)]) - transitmid
    # Time array of cutout: BJD
    t1bjd = np.array(time[int(tindex-nbuffer):int(tindex+nbuffer+1)])


    #print('Between time and flux')
    # Flux array of cutout
    f1 = np.array(flux[int(tindex-nbuffer):int(tindex+nbuffer+1)])
    # Flux error array of cutout
    fe1 = np.array(flux_err[int(tindex-nbuffer):int(tindex+nbuffer+1)])


    if np.any(np.array([j-i for i, j in zip(t1bjd[:-1], t1bjd[1:])]) > 4*cadence): # If the midpoint lies during a gap in the data
        #print('Gap')
        return np.nan, np.nan, np.nan, np.nan

    #print(t1, f1)

    #print('cutout end')

    return t1bjd, t1, f1, fe1



def planetlc(time, per, rp, ars, e, inc, w):

    import batman

    params = batman.TransitParams()
    params.t0 = 0.
    params.per = per
    params.rp = rp
    params.a = ars
    params.inc = inc
    params.ecc = e
    params.w = w
    params.limb_dark = "quadratic"
    params.u = [0.1, 0.3]

    m = batman.TransitModel(params, time)

    flux = m.light_curve(params)

    return flux

def planetlc_fitter(time, per, rp, ars, inc):
    """Always assumes e=0.
    w is NOT a free parameter."""

    import batman

    params = batman.TransitParams()
    params.t0 = 0
    params.per = per
    params.rp = rp
    params.a = a
    params.inc = inc
    params.ecc = 0.0
    params.w = 0.0
    params.limb_dark = "quadratic"
    params.u = [0.5, 0.2]

    m = batman.TransitModel(params, time)

    flux = m.light_curve(params)

    return flux


def integratedlc(time, per, rp, ars, e, inc, w, t0, calc_ptime=True, ptime=None):

    import batman

    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = t0                        #time of inferior conjunction
    params.per = per                      #orbital period
    params.rp = rp                        #planet radius (in units of stellar radii)
    params.a = ars                          #semi-major axis (in units of stellar radii)
    params.inc = inc                      #orbital inclination (in degrees)
    params.ecc = e                      #eccentricity
    params.w = w                       #longitude of periastron (in degrees)
    params.limb_dark = "quadratic"
    params.u = [0.5, 0.2]

    if calc_ptime==True:
        ptime = get_ptime(time, get_mid(time), 29)

    m = batman.TransitModel(params, ptime)

    pflux = m.light_curve(params)
    flux = array_integrated(pflux, 29)

    return flux

def integratedlc_fitter(time, per, rp, ars, inc, t0, calc_ptime=True, ptime=None):

    import batman

    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = t0                        #time of inferior conjunction
    params.per = per                      #orbital period
    params.rp = rp                        #planet radius (in units of stellar radii)
    params.a = ars                          #semi-major axis (in units of stellar radii)
    params.inc = inc                      #orbital inclination (in degrees)
    params.ecc = 0.0                      #eccentricity
    params.w = 0.0                       #longitude of periastron (in degrees)
    params.limb_dark = "quadratic"
    params.u = [0.5, 0.2]

    if calc_ptime==True:
        ptime = get_ptime(time, get_mid(time), 29)

    m = batman.TransitModel(params, ptime)

    pflux = m.light_curve(params)
    flux = array_integrated(pflux, 29)

    return flux



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
