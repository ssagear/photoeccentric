import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_KIC(KOI, muirhead_comb):
    return muirhead_comb[muirhead_comb['KOI'] == str(KOI)].KIC.item()


def planet_params_from_archive(df, kepoiname):
    """Get stellar parameters for the host of a KOI from exoplanet archive (downloaded data).

    Parameters
    ----------
    df: pandas.DataFrame
        dataframe of exop. archive downloaded data
    kepoiname: str
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

    period = float(df.loc[df['kepoi_name'] == kepoiname].koi_period) #period (days)
    period_uerr = float(df.loc[df['kepoi_name'] == kepoiname].koi_period_err1) #period upper error (days)
    period_lerr = float(df.loc[df['kepoi_name'] == kepoiname].koi_period_err2) #period lower error (days)

    rprs = float(df.loc[df['kepoi_name'] == kepoiname].koi_ror) #planet rad/stellar rad
    rprs_uerr = float(df.loc[df['kepoi_name'] == kepoiname].koi_ror_err1) #planet rad upper error (days)
    rprs_lerr = float(df.loc[df['kepoi_name'] == kepoiname].koi_ror_err2) #planet rad lower error (days)

    a_rs = float(df.loc[df['kepoi_name'] == kepoiname].koi_dor) #semi-major axis/r_star (a on Rstar)
    a_rs_uerr = float(df.loc[df['kepoi_name'] == kepoiname].koi_dor_err1) #semi-major axis/r_star upper error
    a_rs_lerr = float(df.loc[df['kepoi_name'] == kepoiname].koi_dor_err2) #semi-major axis/r_star upper error

    i = float(df.loc[df['kepoi_name'] == kepoiname].koi_incl) #inclination (degrees)

    e = float(df.loc[df['kepoi_name'] == kepoiname].koi_eccen) #eccentricity (assumed 0)
    w = float(df.loc[df['kepoi_name'] == kepoiname].koi_longp) #longtitude of periastron (assumed 0)

    return period, period_uerr, period_lerr, rprs, rprs_uerr, rprs_lerr, a_rs, a_rs_uerr, a_rs_lerr, i, e, w


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

    import scipy.constants as c

    a = np.cbrt((period**2*c.G*smass)/(4*np.pi**2*srad**3)) # a on rstar

    return a


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

    import batman

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

    import batman

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
    #params.limb_dark = "linear"
    #params.u = [0.3]
    params.limb_dark = "quadratic"
    params.u = [0.5, 0.2]
    #params.limb_dark = "uniform"
    #params.u = []

    if calc_ptime==True:
        #times to calculate light curve
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
    #params.limb_dark = "linear"
    #params.u = [0.3]
    params.limb_dark = "quadratic"
    params.u = [0.5, 0.2]
    #params.limb_dark = "uniform"
    #params.u = []

    #times to calculate light curve
    if calc_ptime==True:
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

    #n, bins = np.histogram(dist, bins=np.linspace(np.nanmin(dist), np.nanmax(dist), 100))
    n, bins = np.histogram(dist, bins=np.linspace(np.nanmin(dist), np.nanmax(dist), bins))
    mode = bins[np.nanargmax(n)]
    return mode


###################################################################################################################

# Fit Kepler LCs


def remove_oot_data(time, flux, flux_err, midpoints):
    """Removes all out of transit data from a kepler or synthetic light curve"""

    ttime = []
    tflux = []
    tflux_err = []

    for i in range(len(midpoints)):
        m, b, t1bjd, t1, fnorm, fe1 = do_linfit(time, flux, flux_err, midpoints[i], 11, 5)
        ttime.append(t1bjd)
        tflux.append(fnorm)
        tflux_err.append(fe1)


    ttime = np.array(ttime).flatten()
    tflux = np.array(tflux).flatten()
    tflux_err = np.array(tflux_err).flatten()

    tflux = np.nan_to_num(tflux, nan=1.0)
    tflux_err = np.nan_to_num(tflux_err, nan=np.nanmedian(tflux_err))

    return ttime, tflux, tflux_err



def get_stitched_lcs(files, KIC):
    """Stitches Kepler LCs from a list of fits files downloaded from MAST.

    Parameters:
    ----------
    files: List

    KIC: float
        KOI of target

    Returns:
    -------
    hdus: list
        List of FITS hdus
    time:
    flux:
    flux_err:
    startimes:
    stoptimes:

    """

    from astropy.io import fits

    files = sorted(files)

    time = []
    flux = []
    flux_err = []
    hdus = []

    starttimes = []
    stoptimes = []

    for file in files:
        hdu = fits.open(file)

        time.append(list(hdu[1].data['TIME'] + hdu[1].header['BJDREFI'] + hdu[1].header['BJDREFF']))

        start = hdu[1].header['TSTART'] + hdu[1].header['BJDREFI'] + hdu[1].header['BJDREFF']
        starttimes.append(start)

        stop = hdu[1].header['TSTOP'] + hdu[1].header['BJDREFI'] + hdu[1].header['BJDREFF']
        stoptimes.append(stop)

        flux.append(list(hdu[1].data['PDCSAP_FLUX']/np.nanmedian(hdu[1].data['PDCSAP_FLUX'])))
        flux_err.append(list(hdu[1].data['PDCSAP_FLUX_ERR']/np.nanmedian(hdu[1].data['PDCSAP_FLUX'])))

        hdus.append(hdu)

        #hdu.close()

    return hdus, time, flux, flux_err, starttimes, stoptimes



def do_linfit(time, flux, flux_err, transitmid, nbuffer, nlinfit):
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
    t1 = np.array(time[tindex-nbuffer:tindex+nbuffer]) - transitmid
    # Time array of cutout: BJD
    t1bjd = np.array(time[tindex-nbuffer:tindex+nbuffer])

    # Flux array of cutout
    f1 = np.array(flux[tindex-nbuffer:tindex+nbuffer])
    # Flux error array of cutout
    fe1 = np.array(flux_err[tindex-nbuffer:tindex+nbuffer])

    # Do linear fit to OOT data
    idx = np.isfinite(t1) & np.isfinite(f1)
    m, b = np.polyfit(np.concatenate((t1[idx][:nlinfit], t1[idx][-nlinfit:])), np.concatenate((f1[idx][:nlinfit], f1[idx][-nlinfit:])), 1)

    # Subtract linear fit from LC
    linfit = m*t1 + b
    fnorm = (f1-linfit)+1

    return m, b, t1bjd, t1, fnorm, fe1


###################################################################################################################

def tfit_log_likelihood(theta, time, ptime, flux, flux_err):
    """
    Transit fit emcee function

    model = integratedlc_fitter()
    gerr = sigma of g distribution

    """

    per, rp, a, inc, t0 = theta

    model = integratedlc_fitter(time, per, rp, a, inc, t0)
    sigma2 = flux_err ** 2



    return -0.5 * np.sum((flux - model) ** 2 / sigma2 + np.log(sigma2))

def tfit_log_prior(theta):
    """
    Transit fit emcee function

    e must be between 0 and 1
    w must be between -90 and 300

    """
    per, rp, a, inc, t0 = theta
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

    import emcee

    solnx = (guess_transit[0], guess_transit[1], guess_transit[2], guess_transit[3], guess_transit[4])
    pos = solnx + 1e-4 * np.random.randn(nwalk, 5)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, tfit_log_probability, args=(time, ptime, nflux, flux_err), threads=4)
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

###############################################################################################



def tfit_noper_log_likelihood(theta, per, time, ptime, flux, flux_err):
    """
    Transit fit emcee function

    model = integratedlc_fitter()
    gerr = sigma of g distribution

    """

    rp, a, inc, t0 = theta

    model = integratedlc_fitter(time, per, rp, a, inc, t0)
    sigma2 = flux_err ** 2

    return -0.5 * np.sum((flux - model) ** 2 / sigma2 + np.log(sigma2))

def tfit_noper_log_prior(theta):
    """
    Transit fit emcee function

    e must be between 0 and 1
    w must be between -90 and 300

    """
    rp, a, inc, t0 = theta
    if 0.0 < rp < 1.0 and 0.0 < inc < 90.0 and a > 0.0:
        if -0.2 < t0 < 0.2:
            return 0.0
    return -np.inf

def tfit_noper_log_probability(theta, per, time, ptime, flux, flux_err):
    """
    Transit fit emcee function
    """
    lp = tfit_noper_log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + tfit_noper_log_likelihood(theta, per, time, ptime, flux, flux_err)


def mcmc_fitter_noper(guess_transit, per, time, ptime, nflux, flux_err, nwalk, nsteps, ndiscard, e, w, directory, plot_Tburnin=True, plot_Tcorner=True):
    """One-step MCMC transit fitting with `photoeccentric`, does not fit orbital period.
    Suitable for fitting a light curve with only one transit.
    Period must be known and entered into this function.

    NB: (nsteps-ndiscard)*nwalkers must equal the length of rho_star.

    Parameters
    ----------
    guess_transit: np.array
        Initial guess: [Rp/Rs, a/Rs, i (deg), t0]
    per: float
        Known period (days)
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
        Fit results [Rp/Rs, a/Rs, i]
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

    solnx = (guess_transit[0], guess_transit[1], guess_transit[2], guess_transit[3])
    pos = solnx + 1e-4 * np.random.randn(nwalk, 4)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, tfit_noper_log_probability, args=(per, time, ptime, nflux, flux_err), threads=4)
    sampler.run_mcmc(pos, nsteps, progress=True);
    samples = sampler.get_chain()

    if plot_Tburnin==True:
        fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
        labels = ["rprs", "a/Rs", "i", "t0"]
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

    rprs_dist = flat_samples[:,0]
    ars_dist = flat_samples[:,1]
    i_dist = flat_samples[:,2]
    t0_dist = flat_samples[:,3]

    return results, results_errs, rprs_dist, ars_dist, i_dist, t0_dist


###################################################################################################################


def fit_keplc_emcee(KOI, transitmpt, time, flux, flux_err, nwalk, nsteps, ndiscard, nbuffer, spectplanets, muirhead_comb):

    """
    One-step long-cadence light curve fitting with `emcee`.

    Parameters:
    ----------
    KOI: int
        KOI # of planet
    midpoints: list
        Transit mid-times (BJD).
    time: np.array
        Time stamps of stitched long-cadence LCs.
    flux: np.array
        Normalized flux (to 1) of stitched long-cadence LCs.
    flux_err: np.array
        Normalized flux errors (to 1) of stitched long-cadence LCs.
    period: float
        Orbital period (days)
    nwalk: int
        Number of MCMC walkers
    nsteps: int
        Number of MCMC steps
    ndiscard: int
        Number of MCMC steps to discard (post-burn-in)
    nbuffer: int
        Number of out-of-transit data points to keep before and after transit
    spectplanets: pd.DataFrame
        Spectroscopy data from Muirhead et al. (2013) for targets
    muirhead_comb: pd.DataFrame
        Combined stellar params and Gaia data for targets

    Returns:
    -------
    ms: list
        Linear fit slopes in order.
    bs: list
        Linear fit y-intercepts in order.
    timesBJD: list
        Time stamps of transit segments, BJD.
    timesPhase: list
        Time stamps of transit segments, phase (midpoint=0).
    fluxNorm: list
        Normalized, linfit-subtracted flux of transit segments.
    fluxErrs: list
        Normalized flux errors of transit segments.
    rpDists: list
        Fit Rp/Rs distributions.
    arsDists: list
        Fit a/Rs distributions.
    incDists: list
        Fit inclination distributions.
    t0Dists: lsit
        Fit t0 distributions.


    """

    import emcee
    import os

    print('dfkjsaflajsbflasjbfalsjb')

    smass_kg = 1.9885e30  # Solar mass (kg)
    srad_m = 696.34e6 # Solar radius (m)

    # Get alt IDs
    kepid = get_KIC(KOI, muirhead_comb)
    kepname = spectplanets.loc[spectplanets['kepid'] == kepid].kepler_name.values[0]
    kepoiname = spectplanets.loc[spectplanets['kepid'] == float(kepid)].kepoi_name.values[0]
    print('kepoiname', kepoiname)
    #kepoiname = kepoiname.replace('.01', '.01')

    # Define directory to save
    direct = 'ph_segfits/' + str(KOI) + '_emcee/'
    if not os.path.exists(direct):
        os.mkdir(direct)

    # Read in isohrone fits from csv
    isodf = pd.read_csv("datafiles/isochrones/iso_lums_" + str(kepid) + ".csv")


    # Stellar params from isochrone
    mstar = isodf["mstar"].mean()
    mstar_err = isodf["mstar"].std()
    rstar = isodf["radius"].mean()
    rstar_err = isodf["radius"].std()
    # Planet params from archive
    period, period_uerr, period_lerr, rprs, rprs_uerr, rprs_lerr, _, a_uerr_arc, a_lerr_arc, inc, _, _ = planet_params_from_archive(spectplanets, kepoiname)

    # Calculate a/Rs to ensure that it's consistent with the spectroscopy/Gaia stellar density.
    a_rs = calc_a(period*86400.0, mstar*smass_kg, rstar*srad_m)
    a_rs_err = np.mean((a_uerr_arc, a_lerr_arc))

    print('Stellar mass (Msun): ', mstar, 'Stellar radius (Rsun): ', rstar)
    print('Period (Days): ', period, 'Rp/Rs: ', rprs)
    print('a/Rs: ', a_rs)
    print('i (deg): ', inc)

    # Inital guess: period, rprs, a/Rs, i, w
    p0 = [period, rprs, a_rs, inc, transitmpt]

    mid = get_mid(time)
    ptime = get_ptime(time, mid, 29)
    arrlen = (nsteps-ndiscard)*nwalk

    # EMCEE Transit Model Fitting
    res,reserrs, pdist, rdist, adist, idist, t0dist = mcmc_fitter(p0, time, ptime, flux, flux_err, nwalk, nsteps, ndiscard, 'X', 'X', direct, plot_Tburnin=False, plot_Tcorner=False)

    # Create a light curve with the fit parameters
    mcmcfit = integratedlc_fitter(t1, mode(pdist), mode(rdist), mode(adist), mode(idist), mode(t0dist))
    truefit = integratedlc_fitter(t1, period, rprs, a_rs, inc, 0)

        # plt.cla()
        # offset = 2454900.0
        # plt.errorbar(t1bjd-offset, fnorm, yerr=fe1, c='blue', alpha=0.5, label='Original LC', fmt="o", capsize=0)
        # plt.scatter(t1bjd-offset, mcmcfit, c='red', alpha=1.0, label='Fit LC')
        # plt.plot(t1bjd-offset, mcmcfit, c='red', alpha=1.0)
        # plt.plot(t1bjd-offset, truefit, c='green', alpha=0.4)
        # plt.xlabel('BJD-2454900.0')

        # textstr = '\n'.join((
        # r'$\mathrm{Rp/Rs}=%.2f$' % (mode(rdist), ),
        # r'$\mathrm{a_rs}=%.2f$' % (mode(adist), ),
        # r'$\mathrm{i}=%.2f$' % (mode(idist), ),
        # r'$\mathrm{t0}=%.2f$' % (mode(t0dist), )))
        # plt.title(textstr)
        #
        # plt.legend()
        #
        # plt.savefig(direct + str(KOI) + 'segment' + str(ind) + 'fit.png')



    return res, reserrs, pdist, rdist, adist, idist, t0dist


def fit_keplc_dynesty(KOI, midpoints, time, flux, flux_err, pt, arrlen, nbuffer, spectplanets, muirhead_comb):

    """
    One-step long-cadence light curve fitting with `dynesty`.

    Parameters:
    ----------
    KOI: int
        KOI # of planet
    midpoints: list
        Transit mid-times (BJD).
    time: np.array
        Time stamps of stitched long-cadence LCs.
    flux: np.array
        Normalized flux (to 1) of stitched long-cadence LCs.
    flux_err: np.array
        Normalized flux errors (to 1) of stitched long-cadence LCs.
    period: float
        Orbital period (days)
    pt: list
        Prior transform parameters: [period width, period offset, Rp/Rs width, Rp/Rs offset,
        a/Rs width, a/Rs offset, inclination width, inclination offset, t0 width, t0 offset]
    arrlen: int
        Length of planet parameter distributions. Must match length of rho_star when fitting eccentricity.
    nbuffer: int
        Number of out-of-transit data points to keep before and after transit
    spectplanets: pd.DataFrame
        Spectroscopy data from Muirhead et al. (2013) for targets
    muirhead_comb: pd.DataFrame
        Combined stellar params and Gaia data for targets

    Returns:
    -------
    ms: list
        Linear fit slopes in order.
    bs: list
        Linear fit y-intercepts in order.
    timesBJD: list
        Time stamps of transit segments, BJD.
    timesPhase: list
        Time stamps of transit segments, phase (midpoint=0).
    fluxNorm: list
        Normalized, linfit-subtracted flux of transit segments.
    fluxErrs: list
        Normalized flux errors of transit segments.
    perDists: list
        Fit period distributions.
    rpDists: list
        Fit Rp/Rs distributions.
    arsDists: list
        Fit a/Rs distributions.
    incDists: list
        Fit inclination distributions.
    t0Dists: lsit
        Fit t0 distributions.


    """

    import dynesty
    import os
    import random

    smass_kg = 1.9885e30  # Solar mass (kg)
    srad_m = 696.34e6 # Solar radius (m)


    # Get alt IDs
    kepid = get_KIC(KOI, muirhead_comb)
    kepname = spectplanets.loc[spectplanets['kepid'] == kepid].kepler_name.values[0]
    kepoi_name = spectplanets.loc[spectplanets['kepid'] == kepid].kepoi_name.values[0]
    print(kepoi_name)

    # Define directory to save
    direct = 'ph_entire_tfits/' + str(KOI) + '_dynesty/'
    if not os.path.exists(direct):
        os.mkdir(direct)

    # Read in isochrone fits from csv
    isodf = pd.read_csv("datafiles/isochrones/iso_lums_" + str(kepid) + ".csv")

    # Stellar params from isochrone
    mstar = isodf["mstar"].mean()
    mstar_err = isodf["mstar"].std()
    rstar = isodf["radius"].mean()
    rstar_err = isodf["radius"].std()

    # Planet params from archive
    period, period_uerr, period_lerr, rprs, rprs_uerr, rprs_lerr, _, a_uerr_arc, a_lerr_arc, inc, _, _ = planet_params_from_archive(spectplanets, kepoi_name)

    # Calculate a/Rs to ensure that it's consistent with the spectroscopy/Gaia stellar density.
    a_rs = calc_a(period*86400.0, mstar*smass_kg, rstar*srad_m)
    a_rs_err = np.mean((a_uerr_arc, a_lerr_arc))

    #print('Stellar mass (Msun): ', mstar, 'Stellar radius (Rsun): ', rstar)
    #print('Period (Days): ', period, 'Rp/Rs: ', rprs)
    #print('a/Rs: ', a_rs)
    #print('i (deg): ', inc)


    """Define `dynesty` log likelihood and prior transform functions"""

    def tfit_loglike(theta):
        """
        Transit fit dynesty function

        model = integratedlc_fitter()
        gerr = sigma of g distribution

        """

        per, rprs, a_rs, inc, t0 = theta

        model = integratedlc_fitter(time, per, rprs, a_rs, inc, t0)
        sigma2 = flux_err ** 2

        return -0.5 * np.sum((flux - model) ** 2 / sigma2 + np.log(sigma2))


    def tfit_prior_transform(utheta):
        """Transforms samples `u` drawn from the unit cube to samples to those
        from our uniform prior for each variable."""

        uper, urp, ua, uinc, ut0 = utheta

        per = uper*pt[0]+pt[1]
        rprs = urp*pt[2]+pt[3]
        a_rs = ua*pt[4]+pt[5]
        inc = uinc*pt[6]+pt[7]
        t0 = ut0*pt[8]+pt[9]

        return per, rprs, a_rs, inc, t0

    dsampler = dynesty.DynamicNestedSampler(tfit_loglike, tfit_prior_transform, ndim=5, nlive=1500,
                                bound='balls', sample='rwalk', dlogz=0.5)

    dsampler.run_nested()
    dres = dsampler.results

    #Thinning distributions to size arrlen
    pdist = random.choices(dres.samples[:,0], k=arrlen)
    rdist = random.choices(dres.samples[:,1], k=arrlen)
    adist = random.choices(dres.samples[:,2], k=arrlen)
    idist = random.choices(dres.samples[:,3], k=arrlen)
    t0dist = random.choices(dres.samples[:,4], k=arrlen)

    per_f = mode(dres.samples[:,0])
    rprs_f = mode(dres.samples[:,1])
    a_f = mode(dres.samples[:,2])
    i_f = mode(dres.samples[:,3])
    t0_f = mode(dres.samples[:,4])


    """Plot and save transit fit"""

    # Create a light curve with the fit parameters
    nestedfit = integratedlc_fitter(time, per_f, rprs_f, a_f, i_f, t0_f)
    truefit = integratedlc_fitter(time, period, rprs, a_rs, inc, 0)

    plt.cla()
    offset = 2454900.0
    plt.errorbar(time, flux, yerr=flux_err, c='blue', alpha=0.5, label='Original LC', fmt="o", capsize=0)
    plt.scatter(time, nestedfit, c='red', alpha=1.0, label='Fit LC')
    plt.plot(time, nestedfit, c='red', alpha=1.0)
    plt.plot(time, truefit, c='green', alpha=0.4)
    plt.xlabel('BJD')
    plt.ylabel(' Relative Flux')

    np.savetxt('dists2.csv', np.transpose([pdist,rdist,adist,idist,t0dist]), delimiter=',')

    textstr = '\n'.join((
    r'$\mathrm{Period}=%.2f$' % (mode(pdist), ),
    r'$\mathrm{Rp/Rs}=%.2f$' % (mode(rdist), ),
    r'$\mathrm{a_rs}=%.2f$' % (mode(adist), ),
    r'$\mathrm{i}=%.2f$' % (mode(idist), ),
    r'$\mathrm{t0}=%.2f$' % (mode(t0dist), )))
    plt.title(textstr)

    plt.legend()

    plt.savefig(direct + str(KOI) + 'fit.png')


    return dres, pdist, rdist, adist, idist, t0dist



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


def ars_prior_photoeccentric(KOI, midpoints, time, flux, flux_err, pt, rho_star, arrlen, nbuffer, spectplanets, muirhead_comb):

    """
    One-step photoeccentric effect using an a/Rs prior from stellar density with `dynesty`.

    Parameters:
    ----------
    KOI: int
        KOI # of planet
    midpoints: list
        Transit mid-times (BJD).
    time: np.array
        Time stamps of stitched long-cadence LCs.
    flux: np.array
        Normalized flux (to 1) of stitched long-cadence LCs.
    flux_err: np.array
        Normalized flux errors (to 1) of stitched long-cadence LCs.
    period: float
        Orbital period (days)
    pt: list
        Prior transform parameters: [period width, period offset, Rp/Rs width, Rp/Rs offset, inclination width, inclination offset, t0 width, t0 offset]
    arrlen: int
        Length of planet parameter distributions. Must match length of rho_star when fitting eccentricity.
    nbuffer: int
        Number of out-of-transit data points to keep before and after transit
    spectplanets: pd.DataFrame
        Spectroscopy data from Muirhead et al. (2013) for targets
    muirhead_comb: pd.DataFrame
        Combined stellar params and Gaia data for targets

    Returns:
    -------
    ms: list
        Linear fit slopes in order.
    bs: list
        Linear fit y-intercepts in order.
    timesBJD: list
        Time stamps of transit segments, BJD.
    timesPhase: list
        Time stamps of transit segments, phase (midpoint=0).
    fluxNorm: list
        Normalized, linfit-subtracted flux of transit segments.
    fluxErrs: list
        Normalized flux errors of transit segments.
    perDists: list
        Fit period distributions.
    rpDists: list
        Fit Rp/Rs distributions.
    arsDists: list
        Fit a/Rs distributions.
    incDists: list
        Fit inclination distributions.
    t0Dists: lsit
        Fit t0 distributions.


    """

    import dynesty
    import os
    import random

    smass_kg = 1.9885e30  # Solar mass (kg)
    srad_m = 696.34e6 # Solar radius (m)

    ms = []
    bs = []
    timesBJD = []
    timesPhase = []
    fluxNorm = []
    fluxErrs = []
    perDists = []
    rpDists = []
    arsDists = []
    incDists = []
    t0Dists = []


    # Get alt IDs
    kepid = get_KIC(KOI, muirhead_comb)
    kepname = spectplanets.loc[spectplanets['kepid'] == kepid].kepler_name.values[0]

    # Define directory to save
    direct = 'ph_entire_tfits/' + str(KOI) + '_dynesty/'
    if not os.path.exists(direct):
        os.mkdir(direct)

    # Read in isochrone fits from csv
    isodf = pd.read_csv("datafiles/isochrones/iso_lums_" + str(kepid) + ".csv")

    # Stellar params from isochrone
    mstar = isodf["mstar"].mean()
    mstar_err = isodf["mstar"].std()
    rstar = isodf["radius"].mean()
    rstar_err = isodf["radius"].std()

    # Planet params from archive
    period, period_uerr, period_lerr, rprs, rprs_uerr, rprs_lerr, _, a_uerr_arc, a_lerr_arc, inc, _, _ = planet_params_from_archive(spectplanets, kepname)

    # Calculate a/Rs to ensure that it's consistent with the spectroscopy/Gaia stellar density.
    a_rs = calc_a(period*86400.0, mstar*smass_kg, rstar*srad_m)
    a_rs_err = np.mean((a_uerr_arc, a_lerr_arc))

    print('Stellar mass (Msun): ', mstar, 'Stellar radius (Rsun): ', rstar)
    print('Period (Days): ', period, 'Rp/Rs: ', rprs)
    print('a/Rs: ', a_rs)
    print('i (deg): ', inc)

    # Calculate a/Rs CDF prior from rho_star
    ars_prior_hist = calc_a_from_rho(period, rho_star)
    acdfx, acdfy = get_cdf(ars_prior_hist)

    ptime = get_ptime(time, get_mid(time), 29)

    """Define `dynesty` log likelihood and prior transform functions"""

    def tfit_loglike(theta):
        """
        Transit fit dynesty function

        model = integratedlc_fitter()
        gerr = sigma of g distribution

        """

        per, rprs, a_rs, e, inc, w, t0 = theta

        model = integratedlc(time, per, rprs, a_rs, e, inc, w, t0, calc_ptime=False, ptime=ptime)
        sigma2 = flux_err ** 2

        #print(per, rprs, a_rs, e, inc, w, t0)

        return -0.5 * np.sum((flux - model) ** 2 / sigma2 + np.log(sigma2))


    def tfit_prior_transform(utheta):
        """Transforms samples `u` drawn from the unit cube to samples to those
        from our uniform prior for each variable."""

        uper, urp, ua, ue, uinc, uw, ut0 = utheta

        per = uper*pt[0]+pt[1]
        rprs = urp*pt[2]+pt[3]
        a_rs = get_ppf_val(acdfx, acdfy, ua)
        e = ue*pt[4]+pt[5]
        inc = uinc*pt[6]+pt[7]
        w = uw*pt[8]+pt[9]
        t0 = ut0*pt[10]+pt[1]

        return per, rprs, a_rs, e, inc, w, t0

    dsampler = dynesty.DynamicNestedSampler(tfit_loglike, tfit_prior_transform, ndim=7, nlive=1000,
                                bound='multi', sample='unif')

    dsampler.run_nested()
    dres = dsampler.results

    # Thinning distributions to size arrlen
    pdist = random.choices(dres.samples[:,0], k=arrlen)
    rdist = random.choices(dres.samples[:,1], k=arrlen)
    adist = random.choices(dres.samples[:,2], k=arrlen)
    edist  = random.choices(dres.samples[:,3], k=arrlen)
    idist = random.choices(dres.samples[:,4], k=arrlen)
    wdist = random.choices(dres.samples[:,5], k=arrlen)
    t0dist = random.choices(dres.samples[:,6], k=arrlen)

    per_f = mode(dres.samples[:,0])
    rprs_f = mode(dres.samples[:,1])
    a_f = mode(dres.samples[:,2])
    e_f = mode(dres.samples[:,3])
    i_f = mode(dres.samples[:,4])
    w_f = mode(dres.samples[:,5])
    t0_f = mode(dres.samples[:,6])


    """Plot and save transit fit"""

    # Create a light curve with the fit parameters
    nestedfit = integratedlc(time, per_f, rprs_f, a_f, e_f, i_f, w_f, t0_f)
    truefit = integratedlc_fitter(time, period, rprs, a_rs, e, inc, w, 0)

    plt.cla()
    offset = 2454900.0
    plt.errorbar(time, flux, yerr=flux_err, c='blue', alpha=0.5, label='Original LC', fmt="o", capsize=0)
    plt.scatter(time, nestedfit, c='red', alpha=1.0, label='Fit LC')
    plt.plot(time, nestedfit, c='red', alpha=1.0)
    plt.plot(time, truefit, c='green', alpha=0.4)
    plt.xlabel('BJD')
    plt.ylabel(' Relative Flux')

    np.savetxt('dists3.csv', np.transpose([pdist,rdist,adist,edist,idist,wdist,t0dist]), delimiter=',')

    textstr = '\n'.join((
    r'$\mathrm{Period}=%.2f$' % (mode(pdist), ),
    r'$\mathrm{Rp/Rs}=%.2f$' % (mode(rdist), ),
    r'$\mathrm{a_rs}=%.2f$' % (mode(adist), ),
    r'$\mathrm{i}=%.2f$' % (mode(idist), ),
    r'$\mathrm{t0}=%.2f$' % (mode(t0dist), )))
    plt.title(textstr)

    plt.legend()

    plt.savefig(direct + str(KOI) + 'segment' + str(ind) + 'fit.png')


    return m, b, t1bjd, t1, fnorm, fe1, pdist, rdist, adist, edist, idist, wdist, t0dist

"""

####################################################################################################################
MCMC a/Rs Prior photoeccentric
####################################################################################################################

"""

def ars_prior_photoeccentric_mcmc(guess_transit, KOI, midpoints, time, flux, flux_err, rho_star, arrlen, nbuffer, spectplanets, muirhead_comb):

    """
    One-step photoeccentric effect using an a/Rs prior from stellar density with `emcee`.

    Parameters:
    ----------
    KOI: int
        KOI # of planet
    midpoints: list
        Transit mid-times (BJD).
    time: np.array
        Time stamps of stitched long-cadence LCs.
    flux: np.array
        Normalized flux (to 1) of stitched long-cadence LCs.
    flux_err: np.array
        Normalized flux errors (to 1) of stitched long-cadence LCs.
    period: float
        Orbital period (days)
    pt: list
        Prior transform parameters: [period width, period offset, Rp/Rs width, Rp/Rs offset, inclination width, inclination offset, t0 width, t0 offset]
    arrlen: int
        Length of planet parameter distributions. Must match length of rho_star when fitting eccentricity.
    nbuffer: int
        Number of out-of-transit data points to keep before and after transit
    spectplanets: pd.DataFrame
        Spectroscopy data from Muirhead et al. (2013) for targets
    muirhead_comb: pd.DataFrame
        Combined stellar params and Gaia data for targets

    Returns:
    -------
    ms: list
        Linear fit slopes in order.
    bs: list
        Linear fit y-intercepts in order.
    timesBJD: list
        Time stamps of transit segments, BJD.
    timesPhase: list
        Time stamps of transit segments, phase (midpoint=0).
    fluxNorm: list
        Normalized, linfit-subtracted flux of transit segments.
    fluxErrs: list
        Normalized flux errors of transit segments.
    perDists: list
        Fit period distributions.
    rpDists: list
        Fit Rp/Rs distributions.
    arsDists: list
        Fit a/Rs distributions.
    incDists: list
        Fit inclination distributions.
    t0Dists: lsit
        Fit t0 distributions.


    """
    import os
    import random

    smass_kg = 1.9885e30  # Solar mass (kg)
    srad_m = 696.34e6 # Solar radius (m)


    # Get alt IDs
    kepid = get_KIC(KOI, muirhead_comb)
    kepname = spectplanets.loc[spectplanets['kepid'] == kepid].kepler_name.values[0]

    # Define directory to save
    direct = 'ph_entire_tfits/' + str(KOI) + '_dynesty/'
    if not os.path.exists(direct):
        os.mkdir(direct)

    # Read in isochrone fits from csv
    isodf = pd.read_csv("datafiles/isochrones/iso_lums_" + str(kepid) + ".csv")

    # Stellar params from isochrone
    mstar = isodf["mstar"].mean()
    mstar_err = isodf["mstar"].std()
    rstar = isodf["radius"].mean()
    rstar_err = isodf["radius"].std()

    # Planet params from archive
    period, period_uerr, period_lerr, rprs, rprs_uerr, rprs_lerr, _, a_uerr_arc, a_lerr_arc, inc, _, _ = planet_params_from_archive(spectplanets, kepname)

    # Calculate a/Rs to ensure that it's consistent with the spectroscopy/Gaia stellar density.
    a_rs = calc_a(period*86400.0, mstar*smass_kg, rstar*srad_m)
    a_rs_err = np.mean((a_uerr_arc, a_lerr_arc))

    print('Stellar mass (Msun): ', mstar, 'Stellar radius (Rsun): ', rstar)
    print('Period (Days): ', period, 'Rp/Rs: ', rprs)
    print('a/Rs: ', a_rs)
    print('i (deg): ', inc)

    # Calculate a/Rs PDF prior from rho_star
    import scipy.stats
    ars_prior_hist = calc_a_from_rho(period, rho_star)
    hist = np.histogram(ars_prior_hist, bins=250)
    ars_pdf = scipy.stats.rv_histogram(hist)

    a_mean = mode(ars_prior_hist)
    a_std = np.std(ars_prior_hist)
    acdfx, acdfy = get_cdf(ars_prior_hist)


    """Define likelihood functions"""

    def tfit_log_likelihood_ars(theta, time, flux, flux_err):
        """
        Transit fit emcee function

        model = integratedlc_fitter()
        gerr = sigma of g distribution

        """

        per, rp, a, e, inc, w, t0 = theta

        model = integratedlc(time, per, rp, a, e, inc, w, t0)
        sigma2 = flux_err ** 2

        return -0.5 * np.sum((flux - model) ** 2 / sigma2 + np.log(sigma2))

    def tfit_log_prior_ars(theta):
        """
        Transit fit emcee function

        e must be between 0 and 1
        w must be between -90 and 300

        """
        from scipy.stats import norm
        per, rp, a, e, inc, w, t0 = theta
        if 0.0 < rp < 1.0 and 0.0 < e < 1.0 and 0.0 < inc < 90.0 and -90.0 < w < 270.0 and a > 0.0:
            return ars_pdf.pdf(a)
        return -np.inf

    def tfit_log_probability_ars(theta, time, flux, flux_err):
        """
        Transit fit emcee function
        """
        lp = tfit_log_prior_ars(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + tfit_log_likelihood_ars(theta, time, flux, flux_err)



    """Doing emcee fit"""

    import emcee
    import corner

    nsteps = 500
    ndiscard = 300

    solnx = (guess_transit[0], guess_transit[1], guess_transit[2], guess_transit[3], guess_transit[4], guess_transit[5], guess_transit[6])
    pos = solnx + 1e-4 * np.random.randn(64, 7)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, tfit_log_probability_ars, args=(time, flux, flux_err), threads=4)
    sampler.run_mcmc(pos, nsteps, progress=True);
    samples = sampler.get_chain()

    fig, axes = plt.subplots(7, figsize=(10, 7), sharex=True)
    labels = ["period", "rprs", "a/Rs", "e", "i", "w", "t0"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");
        fig.savefig('lcfit_burnin.png')
        plt.close(fig)

    flat_samples = sampler.get_chain(discard=ndiscard, thin=1, flat=True)

    fig = corner.corner(flat_samples, labels=labels);
    fig.savefig('transit_corner.png')
    plt.close(fig)

    pdist = flat_samples[:,0]
    rdist = flat_samples[:,1]
    adist = flat_samples[:,2]
    edist = flat_samples[:,3]
    idist = flat_samples[:,4]
    wdist = flat_samples[:,5]
    t0dist = flat_samples[:,6]


    """Plot and save transit fit"""

    # Create a light curve with the fit parameters
    mcmcfit = integratedlc(time, mode(pdist), mode(rdist), mode(adist), mode(edist), mode(idist), mode(wdist), mode(t0dist))
    truefit = integratedlc(time, period, rprs, a_rs, 0, inc, 90, 0)

    plt.cla()
    offset = 2454900.0
    plt.errorbar(time, flux, yerr=flux_err, c='blue', alpha=0.5, label='Original LC', fmt="o", capsize=0)
    plt.scatter(time, mcmcfit, c='red', alpha=1.0, label='Fit LC')
    plt.plot(time, mcmcfit, c='red', alpha=1.0)
    plt.plot(time, truefit, c='green', alpha=0.4)
    plt.xlabel('BJD')
    plt.ylabel(' Relative Flux')

    np.savetxt('dists3.csv', np.transpose([pdist,rdist,adist,edist,idist,wdist,t0dist]), delimiter=',')

    textstr = '\n'.join((
    r'$\mathrm{Period}=%.2f$' % (mode(pdist), ),
    r'$\mathrm{Rp/Rs}=%.2f$' % (mode(rdist), ),
    r'$\mathrm{a_rs}=%.2f$' % (mode(adist), ),
    r'$\mathrm{i}=%.2f$' % (mode(idist), ),
    r'$\mathrm{t0}=%.2f$' % (mode(t0dist), )))
    plt.title(textstr)

    plt.legend()

    plt.savefig(str(KOI) + 'arsfit.png')


    return pdist, rdist, adist, edist, idist, wdist, t0dist
