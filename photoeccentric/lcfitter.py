import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits

import scipy.constants as c
from tqdm import tqdm
import batman

import emcee
import corner

from astropy.timeseries import BoxLeastSquares

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

def integratedlc_fitter(time, per, rp, ars, inc, t0):

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

    # Find closest Kepler time stamp to transit mid-time
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

###################################################################################################################


def fit_keplc_dynesty(KOI, midpoints, time, flux, flux_err, period, pt, arrlen, nbuffer, spectplanets, muirhead_comb):

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
    ms, bs, timesBJD, timesPhase, fluxNorm, fluxErrs, perDists, rpDists, arsDists, incDists, t0Dists = ([], ) * 11

    # Get alt IDs
    kepid = get_KIC(KOI, muirhead_comb)
    kepname = spectplanets.loc[spectplanets['kepid'] == kepid].kepler_name.values[0]

    # Define directory to save
    direct = 'ph_segfits/' + str(KOI) + '_dynesty/'
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
    period, period_uerr, period_lerr, rprs, rprs_uerr, rprs_lerr, _, _, _, i, _, _ = ph.planet_params_from_archive(spectplanets, kepname)

    # Calculate a/Rs to ensure that it's consistent with the spectroscopy/Gaia stellar density.
    a_rs = ph.calc_a(period*86400.0, mstar*smass_kg, rstar*srad_m)
    a_rs_err = np.mean((a_uerr_arc, a_lerr_arc))

    print('Stellar mass (Msun): ', mstar, 'Stellar radius (Rsun): ', rstar)
    print('Period (Days): ', period, 'Rp/Rs: ', rprs)
    print('a/Rs: ', a_rs)
    print('i (deg): ', i)

    for ind in range(0, len(midpoints)):

        transitmid = midpoints[ind]

        m, b, t1bjd, t1, fnorm, fe1 = ph.do_linfit(time, flux, flux_err, transitmid, nbuffer, 5)

        fnorm = np.nan_to_num(fnorm, nan=1.0)
        fe1 = np.nan_to_num(fe1, nan=np.nanmedian(fe1))

        """Define `dynesty` log likelihood and prior transform functions"""

        def tfit_loglike(theta):
            """
            Transit fit dynesty function

            model = ph.integratedlc_fitter()
            gerr = sigma of g distribution

            """

            per, rprs, a_rs, inc, t0 = theta

            model = ph.integratedlc_fitter(t1, per, rprs, a_rs, inc, t0)
            sigma2 = fe1 ** 2

            return -0.5 * np.sum((fnorm - model) ** 2 / sigma2 + np.log(sigma2))


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

        dsampler = dynesty.DynamicNestedSampler(tfit_loglike, tfit_prior_transform, ndim=5, nlive=1000,
                                    bound='multi', sample='rwalk')

        dsampler.run_nested()
        dres = dsampler.results

        # Thinning distributions to size arrlen
        pdist = random.choices(dres.samples[:,0], k=arrlen)
        rdist = random.choices(dres.samples[:,1], k=arrlen)
        adist = random.choices(dres.samples[:,2], k=arrlen)
        idist = random.choices(dres.samples[:,3], k=arrlen)
        t0dist = random.choices(dres.samples[:,4], k=arrlen)

        per_f = ph.mode(dres.samples[:,0])
        rprs_f = ph.mode(dres.samples[:,1])
        a_f = ph.mode(dres.samples[:,2])
        i_f = ph.mode(dres.samples[:,3])
        t0_f = ph.mode(dres.samples[:,4])


        """Plot and save transit fit"""

        # Create a light curve with the fit parameters
        nestedfit = ph.integratedlc_fitter(t1, per_f, rprs_f, a_f, i_f, t0_f)
        truefit = ph.integratedlc_fitter(t1, period, rprs, a_rs, i, 0)

        plt.cla()
        offset = 2454900.0
        plt.errorbar(t1bjd-offset, fnorm, yerr=fe1, c='blue', alpha=0.5, label='Original LC', fmt="o", capsize=0)
        plt.scatter(t1bjd-offset, nestedfit, c='red', alpha=1.0, label='Fit LC')
        plt.plot(t1bjd-offset, nestedfit, c='red', alpha=1.0)
        plt.plot(t1bjd-offset, truefit, c='green', alpha=0.4)
        plt.xlabel('BJD-2454900.0')
        plt.ylabel(' Relative Flux')

        textstr = '\n'.join((
        r'$\mathrm{Period}=%.2f$' % (ph.mode(pdist), ),
        r'$\mathrm{Rp/Rs}=%.2f$' % (ph.mode(rdist), ),
        r'$\mathrm{a_rs}=%.2f$' % (ph.mode(adist), ),
        r'$\mathrm{i}=%.2f$' % (ph.mode(idist), ),
        r'$\mathrm{t0}=%.2f$' % (ph.mode(t0dist), )))
        plt.title(textstr)

        plt.legend()

        plt.savefig(direct + str(KOI) + 'segment' + str(ind) + 'fit.png')

        ms.append(m)
        bs.append(b)
        timesBJD.append(t1bjd)
        timesPhase.append(t1)
        fluxNorm.append(fnorm)
        fluxErrs.append(fe1)
        perDists.append(pdist)
        rpDists.append(rdist)
        arsDists.append(adist)
        incDists.append(idist)
        t0Dists.append(t0dist)


    return ms, bs, timesBJD, timesPhase, fluxNorm, fluxErrs, perDists, rpDists, arsDists, incDists, t0Dist
