import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits

# Using astropy BLS and scipy curve_fit to fit transit
from astropy.timeseries import BoxLeastSquares

import dynesty

# And importing `photoeccentric`
import photoeccentric as ph

# # Random stuff
import scipy.constants as c
import os

import random


def get_lc_files(KIC, KICs, lcpath):

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



def sigma_clip(time, flux, fluxerr, sig=4):
    for i in tqdm(range(len(flux))):
        if flux[i] > np.nanmean(flux) + sig*np.nanstd(flux) or flux[i] < np.nanmean(flux) - sig*np.nanstd(flux):
            time[i] = np.nan
            flux[i] = np.nan
            fluxerr[i] = np.nan

    return time, flux, fluxerr


def get_KIC(KOI, muirhead_comb):
    return muirhead_comb[muirhead_comb['KOI'] == str(KOI)].KIC.item()


def keplc_fitter(KOI):

    arrlen = 10000

    direct = 'KepLCtfits/' + str(KOI) + '_lcfit_results'

    smass_kg = 1.9885e30  # Solar mass (kg)
    srad_m = 696.34e6 # Solar radius (m)

    spectplanets = pd.read_csv('/blue/sarahballard/ssagear/datafiles/spectplanets.csv')
    muirhead_comb = pd.read_csv('/blue/sarahballard/ssagear/datafiles/muirhead_comb.csv')
    muirhead_comb_lums = pd.read_csv('/blue/sarahballard/ssagear/datafiles/muirhead_comb_lums.csv')
    muirheadKOIs = pd.read_csv('/blue/sarahballard/ssagear/datafiles/MuirheadKOIs.csv')

    KICs = np.sort(np.unique(np.array(muirhead_comb['KIC'])))
    KOIs = np.sort(np.unique(np.array(muirhead_comb['KOI'])))


    # Getting light curve files
    lcpath = '/blue/sarahballard/ssagear/sample_lcs/'
    kepid = get_KIC(KOI, muirhead_comb)

    files = get_lc_files(kepid, KICs, lcpath)

    # Getting stitched LCs
    hdus, t, f, fe, starts, stops = ph.get_stitched_lcs(files, float(kepid))

    # For now, not sigma clipping at All
    alltime_noclip = []
    allflux_noclip = []
    allfluxerr_noclip = []

    for sublist in t:
        for item in sublist:
            alltime_noclip.append(item)

    for sublist in f:
        for item in sublist:
            allflux_noclip.append(item)

    for sublist in fe:
        for item in sublist:
            allfluxerr_noclip.append(item)

    # Defining time, flux, fluxerr from (not) sigma clipped data

    time, flux, flux_err = np.array(alltime_noclip), np.array(allflux_noclip), np.array(allfluxerr_noclip)

    # Kepler name
    kepname = spectplanets.loc[spectplanets['kepid'] == kepid].kepler_name.values[0]

    # Get isochrones, mass, radii
    # Remember to copy isochrones to hpg
    isodf = pd.read_csv("datafiles/isochrones/iso_lums_" + str(kepid) + ".csv")
    mstar = isodf["mstar"].mean()
    mstar_err = isodf["mstar"].std()
    rstar = isodf["radius"].mean()
    rstar_err = isodf["radius"].std()


    period, period_uerr, period_lerr, rprs, rprs_uerr, rprs_lerr, a_arc, a_uerr_arc, a_lerr_arc, i, e_arc, w_arc = ph.planet_params_from_archive(spectplanets, kepname)

    # We calculate a_rs to ensure that it's consistent with the spec/Gaia stellar density.
    a_rs = ph.calc_a(period*86400.0, mstar*smass_kg, rstar*srad_m)
    a_rs_err = np.mean((a_uerr_arc, a_lerr_arc))

    print('Stellar mass (Msun): ', mstar, 'Stellar radius (Rsun): ', rstar)
    print('Period (Days): ', period, 'Rp/Rs: ', rprs)
    print('a/Rs: ', a_rs)
    print('i (deg): ', i)

    # Copy midpoints files
    # Get midpoint
    mpts = pd.read_csv('/blue/sarahballard/ssagear/datafiles/sample_tmidpoints.csv', comment='#')

    transitmpt = mpts.loc[mpts['KOI (star)'] == KOI]['Transit Epoch (BJD)'].values[0]
    starttime = stars[0]
    stoptime = stops[-1]

    midpoints = np.concatenate((np.arange(transitmpt, starttime, -period), np.arange(transitmpt, stoptime, period)))

    ttime, tflux, tflux_err = ph.remove_oot_data(time, flux, flux_err, midpoints)

    ### Prior transform and fitting ###
    priortransform = [3., period-1, 1., 0., 30., a_rs-15, 2., 88, 0.1, transitmpt]
    nbuffer = 11

    # Fitting Transits with Dynesty
    dres, pdist, rdist, adist, idist, t0dist = ph.fit_keplc_dynesty(KOI, midpoints, ttime, tflux, tflux_err, priortransform, arrlen, nbuffer, spectplanets, muirhead_comb)

    with open(direct + 'keptransitdres.pickle', 'wb') as f:
        pickle.dump(dres, f, pickle.HIGHEST_PROTOCOL)

    ### Saving Dists ###
    np.savetxt(direct + 'pdist.csv', pdist, delimiter=',')
    np.savetxt(direct + 'rdist.csv', rdist, delimiter=',')
    np.savetxt(direct + 'adist.csv', adist, delimiter=',')
    np.savetxt(direct + 'idist.csv', idist, delimiter=',')
    np.savetxt(direct + 't0dist.csv', t0dist, delimiter=',')


    # Corner Plot of Transit Fit
    fig, axes = dyplot.cornerplot(dres, labels=["period", "Rp/Rs", "a/Rs", "i", "t0"])
    plt.savefig(direct + 'keptransitcorner.png')

    per_f = ph.mode(pdist)
    rprs_f = ph.mode(rdist)
    a_f = ph.mode(adist)
    i_f = ph.mode(idist)
    t0_f = ph.mode(t0dist)

    # Create a light curve with the fit parameters
    fit1 = ph.integratedlc_fitter(ttime, per_f, rprs_f, a_f, i_f, t0_f)

    T14dist = ph.get_T14(pdist, rdist, adist, idist)
    T14errs = ph.get_sigmas(T14dist)

    T23dist = ph.get_T23(pdist, rdist, adist, idist)
    T23errs = ph.get_sigmas(T23dist)

    gs, rho_c = ph.get_g_distribution(rho_star, pdist, rdist, T14dist, T23dist)

    g_mean = ph.mode(gs)
    g_sigma = np.mean(np.abs(ph.get_sigmas(gs)))


    def loglike(theta):
        """The log-likelihood function."""

        w, e = theta

        model = (1+e*np.sin(w*(np.pi/180.)))/np.sqrt(1-e**2)
        sigma2 = g_sigma ** 2

        return -0.5 * np.sum((g_mean - model) ** 2 / sigma2 + np.log(sigma2))


    def betae_prior_transform(utheta):
        """Uniform eccentricity prior"""

        uw, ue = utheta
        w = 360.*uw-90.

        a, b = 0.867, 3.03
        e = scipy.stats.beta.ppf(ue, a, b)

        return w, e

    dsampler = dynesty.DynamicNestedSampler(loglike, betae_prior_transform, ndim=2, bound='multi', sample='rwalk')
    dsampler.run_nested()

    ewdres = dsampler.results

    edist  = random.choices(ewdres.samples[:,0], k=arrlen)
    wdist  = random.choices(ewdres.samples[:,1], k=arrlen)

    np.savetxt(direct + 'edist.csv', edist, delimiter=',')
    np.savetxt(direct + 'wdist.csv', wdist, delimiter=',')
    np.savetxt(direct + 'T14dist.csv', T14dist, delimiter=',')
    np.savetxt(direct + 'T23dist.csv', T23dist, delimiter=',')
    np.savetxt(direct + 'gdist.csv', gs, delimiter=',')

    with open(direct + 'kepewdres.pickle', 'wb') as f:
        pickle.dump(ewdres, f, pickle.HIGHEST_PROTOCOL)

    fig, axes = dyplot.cornerplot(dres, show_titles=True, title_kwargs={'y': 1.04}, labels=["w", "e"],
                                  fig=plt.subplots(2, 2, figsize=(8, 8)))

    plt.savefig(direct + 'kepewcorner.png')


    #return dres, pdist, rdist, adist, idist, t0dist, gs, edist, wdist, T14dist, T23dist, direct
