import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_lc_files(KIC, KICs, lcpath):

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
    """Gets approximately 1/2 of cadence time."""

    return (time[1]-time[0])/2.

def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.nanargmin((np.abs(array - value)))
    return array[idx]


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

def mode(dist, bins=500):

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

    n, bins = np.histogram(dist, bins=np.linspace(np.nanmin(dist), np.nanmax(dist), bins))
    mode = bins[np.nanargmax(n)]
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

#
# def fit_keplc_emcee(KOI, transitmpt, time, flux, flux_err, nwalk, nsteps, ndiscard, nbuffer, spectplanets, muirhead_comb):
#
#     """
#     One-step long-cadence light curve fitting with `emcee`.
#
#     Parameters:
#     ----------
#     KOI: int
#         KOI # of planet
#     midpoints: list
#         Transit mid-times (BJD).
#     time: np.array
#         Time stamps of stitched long-cadence LCs.
#     flux: np.array
#         Normalized flux (to 1) of stitched long-cadence LCs.
#     flux_err: np.array
#         Normalized flux errors (to 1) of stitched long-cadence LCs.
#     period: float
#         Orbital period (days)
#     nwalk: int
#         Number of MCMC walkers
#     nsteps: int
#         Number of MCMC steps
#     ndiscard: int
#         Number of MCMC steps to discard (post-burn-in)
#     nbuffer: int
#         Number of out-of-transit data points to keep before and after transit
#     spectplanets: pd.DataFrame
#         Spectroscopy data from Muirhead et al. (2013) for targets
#     muirhead_comb: pd.DataFrame
#         Combined stellar params and Gaia data for targets
#
#     Returns:
#     -------
#     ms: list
#         Linear fit slopes in order.
#     bs: list
#         Linear fit y-intercepts in order.
#     timesBJD: list
#         Time stamps of transit segments, BJD.
#     timesPhase: list
#         Time stamps of transit segments, phase (midpoint=0).
#     fluxNorm: list
#         Normalized, linfit-subtracted flux of transit segments.
#     fluxErrs: list
#         Normalized flux errors of transit segments.
#     rpDists: list
#         Fit Rp/Rs distributions.
#     arsDists: list
#         Fit a/Rs distributions.
#     incDists: list
#         Fit inclination distributions.
#     t0Dists: lsit
#         Fit t0 distributions.
#
#
#     """
#
#     import emcee
#     import os
#
#     print('dfkjsaflajsbflasjbfalsjb')
#
#     smass_kg = 1.9885e30  # Solar mass (kg)
#     srad_m = 696.34e6 # Solar radius (m)
#
#     # Get alt IDs
#     kepid = get_KIC(KOI, muirhead_comb)
#     kepname = spectplanets.loc[spectplanets['kepid'] == kepid].kepler_name.values[0]
#     kepoiname = spectplanets.loc[spectplanets['kepid'] == float(kepid)].kepoi_name.values[0]
#     print('kepoiname', kepoiname)
#     #kepoiname = kepoiname.replace('.01', '.01')
#
#     # Define directory to save
#     direct = 'ph_segfits/' + str(KOI) + '_emcee/'
#     if not os.path.exists(direct):
#         os.mkdir(direct)
#
#     # Read in isohrone fits from csv
#     isodf = pd.read_csv("datafiles/isochrones/iso_lums_" + str(kepid) + ".csv")
#
#
#     # Stellar params from isochrone
#     mstar = isodf["mstar"].mean()
#     mstar_err = isodf["mstar"].std()
#     rstar = isodf["radius"].mean()
#     rstar_err = isodf["radius"].std()
#     # Planet params from archive
#     period, period_uerr, period_lerr, rprs, rprs_uerr, rprs_lerr, _, a_uerr_arc, a_lerr_arc, inc, _, _ = planet_params_from_archive(spectplanets, kepoiname)
#
#     # Calculate a/Rs to ensure that it's consistent with the spectroscopy/Gaia stellar density.
#     a_rs = calc_a(period*86400.0, mstar*smass_kg, rstar*srad_m)
#     a_rs_err = np.mean((a_uerr_arc, a_lerr_arc))
#
#     print('Stellar mass (Msun): ', mstar, 'Stellar radius (Rsun): ', rstar)
#     print('Period (Days): ', period, 'Rp/Rs: ', rprs)
#     print('a/Rs: ', a_rs)
#     print('i (deg): ', inc)
#
#     # Inital guess: period, rprs, a/Rs, i, w
#     p0 = [period, rprs, a_rs, inc, transitmpt]
#
#     mid = get_mid(time)
#     ptime = get_ptime(time, mid, 29)
#     arrlen = (nsteps-ndiscard)*nwalk
#
#     # EMCEE Transit Model Fitting
#     res,reserrs, pdist, rdist, adist, idist, t0dist = mcmc_fitter(p0, time, ptime, flux, flux_err, nwalk, nsteps, ndiscard, 'X', 'X', direct, plot_Tburnin=False, plot_Tcorner=False)
#
#     # Create a light curve with the fit parameters
#     mcmcfit = integratedlc_fitter(t1, mode(pdist), mode(rdist), mode(adist), mode(idist), mode(t0dist))
#     truefit = integratedlc_fitter(t1, period, rprs, a_rs, inc, 0)
#
#     return res, reserrs, pdist, rdist, adist, idist, t0dist



def calc_a_from_rho(period, rho_star):

    import scipy.constants as c
    a_rs = (((period*86400.0)**2)*((c.G*rho_star)/(3*c.pi)))**(1./3.)
    return a_rs


def get_cdf(dist, nbins=250):

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

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if len(np.where(array == array[idx])[0]) == 1:
        return int(np.where(array == array[idx])[0])
    else:
        return int(np.where(array == array[idx])[0][0])
