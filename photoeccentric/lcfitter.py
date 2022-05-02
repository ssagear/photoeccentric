import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .utils import *
from .stellardensity import *


def tfit_log_likelihood(theta, time, ptime, flux, flux_err):
    """
    Transit fit emcee function

    model = integratedlc_fitter()
    gerr = sigma of g distribution

    """

    per, rp, a, inc, t0 = theta

    model = integratedlc_fitter(time, per, rp, a, inc, t0, calc_ptime=False, ptime=ptime)

    sigma2 = flux_err ** 2

    return -0.5 * np.sum((flux - model) ** 2 / sigma2 + np.log(sigma2))

def tfit_log_prior(theta):
    """
    Transit fit emcee function

    e must be between 0 and 1
    w must be between -90 and 300

    """
    per, rp, a, inc, t0 = theta
    if 0.0 < rp < 1.0 and np.arccos((1+rp)*(1./a))*(180./np.pi) < inc < 90.0 and a > 0.0:
        return 0.0
    return -np.inf

def tfit_log_probability(theta, time, ptime, flux, flux_err):
    """
    Transit fit emcee function
    """
    lp = tfit_log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + tfit_log_likelihood(theta, time, ptime, flux, flux_err)



def do_linfit(time, flux, flux_err, transitmid, nbuffer, nlinfit, odd=False):
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

    # Find closest Kepler time stamp to transit mid-timess
    tindex = int(np.where(time == find_nearest(time, transitmid))[0])
    # Time array of cutout: phase 0 at tmid
    t1 = np.array(time[tindex-nbuffer:tindex+nbuffer+1]) - transitmid
    # Time array of cutout: BJD
    t1bjd = np.array(time[tindex-nbuffer:tindex+nbuffer+1])


    # Flux array of cutout
    f1 = np.array(flux[tindex-nbuffer:tindex+nbuffer+1])
    # Flux error array of cutout
    fe1 = np.array(flux_err[tindex-nbuffer:tindex+nbuffer+1])

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
    fnorm = (f1-linfit)+1
    fnorm = f1/linfit

    return m, b, t1bjd, t1, fnorm, fe1




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

    # Find closest Kepler time stamp to transit mid-timess
    tindex = int(np.where(time == find_nearest(time, transitmid))[0])
    if abs(time-find_nearest(time, transitmid)) < 1.*cadence: # if the nearest time stamp is more than a cadence away from the midpoint, the midpoint lies in a gap.
        return np.nan, np.nan, np.nan, np.nan, np.nan


    # Time array of cutout: phase 0 at tmid
    t1 = np.array(time[tindex-nbuffer:tindex+nbuffer+1]) - transitmid
    # Time array of cutout: BJD
    t1bjd = np.array(time[tindex-nbuffer:tindex+nbuffer+1])


    # Flux array of cutout
    f1 = np.array(flux[tindex-nbuffer:tindex+nbuffer+1])
    # Flux error array of cutout
    fe1 = np.array(flux_err[tindex-nbuffer:tindex+nbuffer+1])

    return t1bjd, t1, f1, fe1


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


def mcmc_fitter(guess_transit, time, nflux, flux_err, nwalk, nsteps, ndiscard, directory, plot_Tburnin=True, plot_Tcorner=True, backend=True, reset_backend=True):
    """One-step MCMC transit fitting with `photoeccentric`.

    NB: (nsteps-ndiscard)*nwalkers must equal the length of rho_star.

    Parameters
    ----------
    guess_transit: np.array
        Initial guess: [period, Rp/Rs, a/Rs, i (deg)]
    time: np.array
        Time axis of light curve to fit
    nflux: np.arrayss
        Flux of light curve to fit
    flux_err: np.array
        Flux errors of light curve to fit
    nwalk: int
        Number of `emcee` walkers
    nsteps: int
        Number of `emcee` steps to run
    ndiscard: int
        Number of `emcee` steps to discard (after burn-in)
    directory: str
        Full directory path to save plots
    plot_Tburnin: boolean, default True
        Plot burn-in and save to directory
    plot_Tcorner: boolean, default True
        Plot corner plot and save to directory

    Returns
    -------
    results: list
        Fit results [period, Rp/Rs, a/Rs, i]
    results_errs: list
        Fit results errors [period err, Rp/Rs err, a/Rs err, i err]
    rprs_dist: np.array
        MCMC Rp/Rs distribution
    ars_dist: np.array
        MCMC a/Rs distributions
    i_dist: np.array
        MCMC i distribution

    """

    import emcee

    solnx = (guess_transit[0], guess_transit[1], guess_transit[2], guess_transit[3], guess_transit[4])
    pos = solnx + 1e-4 * np.random.randn(nwalk, 5)
    nwalkers, ndim = pos.shape

    ptime = get_ptime(time, get_mid(time), 29)

    if backend==True:
        filename = directory + "_sampler.h5"
        backend = emcee.backends.HDFBackend(filename)
    if reset_backend==True:
        backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, tfit_log_probability, args=(time, ptime, nflux, flux_err), threads=4, backend=backend)
    sampler.run_mcmc(pos, nsteps, progress=True);
    samples = sampler.get_chain()

    if plot_Tburnin==True:
        fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
        labels = ["period", "rprs", "a/Rs", "i", "t0"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");
        fig.savefig(directory + 'lcfit_burnin.png')
        plt.close(fig)

    flat_samples = sampler.get_chain(discard=ndiscard, thin=1, flat=True)

    if plot_Tcorner==True:
        fig = corner.corner(flat_samples, labels=labels);
        fig.savefig(directory + 'transit_corner.png')
        plt.close(fig)

    results = []
    results_errs = []

    for i in range(ndim):
        results.append(np.percentile(flat_samples[:,i], 50))
        results_errs.append(np.mean((np.percentile(flat_samples[:,i], 16), np.percentile(flat_samples[:,i], 84))))

    per_dist = flat_samples[:,0]
    rprs_dist = flat_samples[:,1]
    ars_dist = flat_samples[:,2]
    i_dist = flat_samples[:,3]
    t0_dist = flat_samples[:,4]

    return results, results_errs, per_dist, rprs_dist, ars_dist, i_dist, t0_dist
