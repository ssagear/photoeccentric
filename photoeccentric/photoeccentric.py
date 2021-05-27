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
from .lcfitter import *


def get_T14(p, rprs, a_rs, i, ecc_prior=False, e=None, w=None):
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
    nf = len(p)

    T14 = np.zeros(nf)
    rs_a = np.zeros(nf)
    b = np.zeros(nf)

    chidot = np.zeros(nf)

    for j in range(nf):

        p[j] = p[j]#*86400

        rs_a[j] = 1.0/a_rs[j]                  # Rs/a - rstar in units of semimajor axis
        b[j] = a_rs[j]*np.cos(i[j]*(np.pi/180.0))   # convert i to radians

        T14[j] = (p[j]/np.pi)*np.arcsin(rs_a[j]*(np.sqrt(((1+rprs[j])**2)-b[j]**2))/np.sin(i[j]*(np.pi/180.0))) #Equation 14 in exoplanet textbook

        if ecc_prior==True:
            chidot[j] = np.sqrt(1-e[j]**2)/(1+e[j]*np.sin(w[j]*(np.pi/180.0))) #Equation 16 in exoplanet textbook
            return T14[j]*chidot[j]

    return T14

def get_T23(p, rprs, a_rs, i, ecc_prior=False, e=None, w=None):
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

    nf = len(p)

    T23 = np.zeros(nf)
    rs_a = np.zeros(nf)
    b = np.zeros(nf)

    chidot = np.zeros(nf)

    for j in range(nf):

        p[j] = p[j]#*86400

        rs_a[j] = 1.0/a_rs[j]                  # Rs/a - rstar in units of semimajor axis
        b[j] = a_rs[j]*np.cos(i[j]*(np.pi/180.0))   # convert i to radians

        T23[j] = (p[j]/np.pi)*np.arcsin(rs_a[j]*(np.sqrt(((1-rprs[j])**2)-b[j]**2))/np.sin(i[j]*(np.pi/180.0))) #Equation 14 in exoplanet textbook

        if ecc_prior==True:
            chidot[j] = np.sqrt(1-e[j]**2)/(1+e[j]*np.sin(w[j]*(np.pi/180.0))) #Equation 16 in exoplanet textbook
            return T23[j]*chidot[j]

    return T23

def calc_a(period, smass, srad):
    """Calculates semi-major axis from planet period and stellar mass
    Kepler's 3rd law

    Parameters
    ----------
    period: float
        Planet period (SECONDS)
    smass: float
        Stellar mass (KG)
    srad: float
        Stellar radius (METERS)

    Returns
    -------
    a: float
        a/Rs: Semi-major axis of planet's orbit (units of stellar radii)
    """

    a = np.cbrt((period**2*c.G*smass)/(4*np.pi**2*srad**3)) # a on rstar

    return a

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

def planet_params_from_archive(df, kep_name):
    """Get stellar parameters for the host of a KOI from exoplanet archive (downloaded data).

    Parameters
    ----------
    df: pandas.DataFrame
        dataframe of exop. archive downloaded data
    kep_name: str
        Kepler name of planet

    Returns
    -------
    period: float
        Orbital period (days)
    rprs: float
        Planet radius (stellar radii)
    a: float
        Semi-major axis (stellar radii)
    e: float
        Eccentricity
    w: float
        Longitude of periastron (degrees)
    i: float
        Inclination (degrees)

    """

    period = float(df.loc[df['kepler_name'] == kep_name].koi_period) #period (days)
    period_uerr = float(df.loc[df['kepler_name'] == kep_name].koi_period_err1) #period upper error (days)
    period_lerr = float(df.loc[df['kepler_name'] == kep_name].koi_period_err2) #period lower error (days)

    rprs = float(df.loc[df['kepler_name'] == kep_name].koi_ror) #planet rad/stellar rad
    rprs_uerr = float(df.loc[df['kepler_name'] == kep_name].koi_ror_err1) #planet rad upper error (days)
    rprs_lerr = float(df.loc[df['kepler_name'] == kep_name].koi_ror_err2) #planet rad lower error (days)

    a_rs = float(df.loc[df['kepler_name'] == kep_name].koi_dor) #semi-major axis/r_star (a on Rstar)
    a_rs_uerr = float(df.loc[df['kepler_name'] == kep_name].koi_dor_err1) #semi-major axis/r_star upper error
    a_rs_lerr = float(df.loc[df['kepler_name'] == kep_name].koi_dor_err2) #semi-major axis/r_star upper error

    i = float(df.loc[df['kepler_name'] == kep_name].koi_incl) #inclination (degrees)

    e = float(df.loc[df['kepler_name'] == kep_name].koi_eccen) #eccentricity (assumed 0)
    w = float(df.loc[df['kepler_name'] == kep_name].koi_longp) #longtitude of periastron (assumed 0)

    return period, period_uerr, period_lerr, rprs, rprs_uerr, rprs_lerr, a_rs, a_rs_uerr, a_rs_lerr, i, e, w

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

def log_likelihood(theta, g, gerr):
    """Log of likelihood
    model = g(e,w)
    gerr = sigma of g distribution
    """
    w, e = theta
    model = (1+e*np.sin(w*(np.pi/180.)))/np.sqrt(1-e**2)
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

def photo_init(time, per, rp, a, e, inc, w, noise=0.000005):
    """Creates fake transit based on input planet parameters.

    Parameters
    ----------
    time: np.array
        Time over which to evaluate transit flux
    per: float
        Orbital peiod (days)
    rp: float
        Planet radius (units of stellar radii)
    a: float
        Semi-major axis (units of stellar radii)
    e: float
        Eccentricity
    inc: float
        Inclination (degrees)
    w: float
        Longitude of periastron (degrees)
    noise: float, default 0.000005
        Sigma of Gaussian noise to be added to light curve flux

    Returns
    -------
    nflux: np.array
        Flux array with noise
    flux_err: np.array
        Flux errors

    """

    # Calculate flux from transit model
    flux = integratedlc(time, per, rp, a, e, inc, w)

    # Adding some gaussian noise
    noise = np.random.normal(0,noise,len(time))
    nflux = flux+noise

    flux_err = np.array([0.00005]*len(nflux))

    return nflux, flux_err

def tfit_log_likelihood(theta, time, ptime, flux, flux_err):
    """
    Transit fit emcee function

    model = ph.integratedlc_fitter()
    gerr = sigma of g distribution

    """

    per, rp, a, inc = theta

    model = integratedlc_fitter(time, per, rp, a, inc)
    sigma2 = flux_err ** 2

    return -0.5 * np.sum((flux - model) ** 2 / sigma2 + np.log(sigma2))

def tfit_log_prior(theta):
    """
    Transit fit emcee function

    e must be between 0 and 1
    w must be between -90 and 300

    """
    per, rp, a, inc = theta
    if 0.0 < rp < 1.0 and 0.0 < inc < 90.0 and a > 0.0:
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

def mcmc_fitter(guess_transit, time, ptime, nflux, flux_err, nwalk, nsteps, ndiscard, e, w, directory, plot_Tburnin=True, plot_Tcorner=True):
    """One-step MCMC transit fitting with `photoeccentric`.

    NB: (nsteps-ndiscard)*nwalkers must equal the length of rho_star.

    Parameters
    ----------
    guess_transit: np.array
        Initial guess: [Rp/Rs, a/Rs, i (deg)]
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
    e: float
        True eccentricity (just to name directory)
    w: float
        True longitude of periastron (just to name directory)
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

    solnx = (guess_transit[0], guess_transit[1], guess_transit[2], guess_transit[3])
    pos = solnx + 1e-4 * np.random.randn(nwalk, 4)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, tfit_log_probability, args=(time, ptime, nflux, flux_err), threads=4)
    sampler.run_mcmc(pos, nsteps, progress=True);
    samples = sampler.get_chain()

    if plot_Tburnin==True:
        fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
        labels = ["period", "rprs", "a/Rs", "i"]
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

    return results, results_errs, per_dist, rprs_dist, ars_dist, i_dist

def photo_fit(time, ptime, nflux, flux_err, guess_transit, guess_ew, rho_star, e, w, directory, nwalk, nsteps, ndiscard, plot_transit=True, plot_burnin=True, plot_corner=True, plot_Tburnin=True, plot_Tcorner=True):

    """Fit eccentricity for a planet.

    Parameters
    ----------
    time: np.array
        Light curve time
    nflux: np.array
        Light curve flux
    flux_err: np.array
        Light curve flux errors
    guess_transit: np.array (length 4)
        Initial guess for MCMC transit fitting. Passed into mcmc_fitter().
    guess_ew: np.array (length 2)
        Initial guess for MCMC e and w fitting. [e guess, w guess]
    rho_star: np.array
        "True" stellar density distribution
    e: float
        True eccentricity (just to name plots)
    w: float
        True longitude of periastron (just to name plots)
    directory: str
        Directory to save plots
    nwalk: int
        Number of walkers
    nsteps: int
        Number of steps to run in MCMC. Passed into mcmc_fitter().
    ndiscard: int
        Number of steps to discard in MCMC. Passed into mcmc_fitter().
    plot_transit: boolean, default True
        Save transit light curve plot + fit in specified directory.

    Returns
    -------
    fite: float
        Best-fit eccentricity (mean of MCMC distribution)
    fitw: float
        Best-fit longitude of periastron (mean of MCMC distribution)
    gs: np.array
        "g" distribution for planet
    g_mean: float
        Mean of g distribution
    g_sigmas: list (length 2)
        [(-) sigma, (+) sigma] of g distribution
    zsc: list (length 2)
        Number of sigmas away [fit e, fit w] are from true [e, w]


    """

    # EMCEE Transit Model Fitting
    _, _, pdist, rdist, adist, idist = mcmc_fitter(guess_transit, time, ptime, nflux, flux_err, nwalk, nsteps, ndiscard, e, w, directory, plot_Tburnin=True, plot_Tcorner=True)

    p_f, perr_f = mode(pdist), get_sigmas(pdist)
    rprs_f, rprserr_f = mode(rdist), get_sigmas(rdist)
    a_f, aerr_f = mode(adist), get_sigmas(adist)
    i_f, ierr_f = mode(idist), get_sigmas(idist)

    # Create a light curve with the fit parameters
    # Boobooboo
    fit = integratedlc_fitter(time, p_f, rprs_f, a_f, i_f)

    if plot_transit==True:
        plt.cla()
        plt.errorbar(time, nflux, yerr=flux_err, c='blue', fmt='o', alpha=0.5, label='Original LC')
        plt.scatter(time, fit, c='red', alpha=1.0)
        plt.plot(time, fit, c='red', alpha=1.0, label='Fit LC')
        #plt.xlim(-0.1, 0.1)
        plt.legend()

        plt.savefig(directory + 'lightcurve_fitp' + str(p_f) + '_fitrprs' + str(rprs_f) + '_fitars' + str(a_f) + '_fiti' + str(i_f) + '.png')
        plt.close()

    print('Fit params:')
    print('Period (days): ', p_f)
    print('Rp/Rs: ', rprs_f)
    print('a/Rs: ', a_f)
    print('i (deg): ', i_f)

    T14dist = get_T14(pdist, rdist, adist, idist)
    T23dist = get_T23(pdist, rdist, adist, idist)

    gs, rho_c = get_g_distribution(rho_star, pdist, rdist, T14dist, T23dist)

    g_mean = mode(gs)
    g_sigma_min, g_sigma_plus = get_sigmas(gs)
    g_sigmas = [g_sigma_min, g_sigma_plus]

    #Guesses
    w_guess = guess_ew[1]
    e_guess = guess_ew[0]

    solnx = (w_guess, e_guess)
    pos = solnx + 1e-4 * np.random.randn(32, 2)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(g_mean, np.nanmean(g_sigmas)), threads=4)

    print('-------MCMC------')
    sampler.run_mcmc(pos, 5000, progress=True);
    flat_samples_e = sampler.get_chain(discard=1000, thin=15, flat=True)

    if plot_burnin==True:

        fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        labels = ["w", "e"]

        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");

        fig.savefig(directory + 'e_g_burnin.png')
        plt.close(fig)

    edist = flat_samples_e[:,1]
    wdist = flat_samples_e[:,0]

    fite = np.percentile(edist, 50)
    fitw = np.percentile(wdist, 50)

    mcmc_e = np.percentile(edist, [16, 50, 84])
    q_e = np.diff(mcmc_e)

    mcmc_w = np.percentile(wdist, [16, 50, 84])
    q_w = np.diff(mcmc_w)

    zsc_e = zscore(e, mcmc_e[1], np.mean(q_e))
    zsc_w = zscore(w, mcmc_w[1], np.mean(q_w))
    zsc = [zsc_e, zsc_w]

    if plot_corner==True:

        fig = corner.corner(flat_samples_e, labels=labels, show_titles=True, title_kwargs={"fontsize": 12}, truths=[w, e], quantiles=[0.16, 0.5, 0.84], plot_contours=True);
        fig.savefig(directory + 'corner_fit_e' + str(fite) + '_fit_w' + str(fitw) + '_fit_g' + str(g_mean) + '.png')
        plt.close(fig)

    return p_f, rprs_f, a_f, i_f, fite, fitw, edist, wdist, gs, g_mean, g_sigmas, zsc
