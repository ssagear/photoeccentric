import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import juliet

from .utils import *
from .stellardensity import *
from .lcfitter import *
from .eccentricity import *


def initialize_data(nkoi, muirhead_comb):
    """This is my own temp function"""
    """Remove when done running"""

    star = KeplerStar(int(np.floor(float(nkoi))))
    star.get_KIC(muirhead_comb)

    isodf = pd.read_csv("/Users/ssagear/Dropbox (UFL)/Research/MetallicityProject/HiPerGator/HPG_Replica/datafiles/iso_lums_" + str(star.KIC) + ".csv")
    direct = 'Local_emceeKepLCtfits/' + str(nkoi) + '_lcfit_results/'

    return direct, isodf


def run_fit_juliet(nkoi, isodf, spectplanets, muirhead_comb, muirheadKOIs, lcpath, direct, nbuffer, nlinfit, nlive=1000, nsupersample=29, exptimesupersample=0.0201389, show_plot=False, juliet_phase=False, std=1, lim=0.12):

    import os

    star = KeplerStar(int(np.floor(float(nkoi))))
    star.get_stellar_params(isodf)
    koi = KOI(nkoi, int(np.floor(float(nkoi))), isodf)
    koi.get_KIC(muirhead_comb)
    koi.planet_params_from_archive(spectplanets)
    koi.calc_a(koi.mstar, koi.rstar)

    KICs = np.sort(np.unique(np.array(muirhead_comb['KIC'])))
    KOIs = np.sort(np.unique(np.array(muirhead_comb['KOI'])))

    files = get_lc_files(koi.KIC, KICs, lcpath)

    #print(files)

    koi.get_stitched_lcs(files)
    koi.get_midpoints()
    print(nbuffer, nlinfit)
    koi.remove_oot_data(nbuffer, nlinfit)


    if juliet_phase==True:
        koi.phase_intransit = juliet.utils.get_phases(koi.time_intransit-2454900, koi.period, koi.epoch-2454900)

    plt.scatter(koi.phase_intransit, koi.flux_intransit, s=1, c='k')

    plt.savefig(direct + 'foldedtransit.png')
    if show_plot == True:
        plt.show()


    dataset, results = koi.do_tfit_juliet(direct, nsupersample=nsupersample, exptimesupersample=exptimesupersample, nlive=nlive)
    koi.calc_durations()
    koi.get_gs()
    koi.do_eccfit(direct)

    return dataset, results, koi.e_dist, koi.w_dist

def run_ttv_fit_juliet(nkoi, isodf, spectplanets, muirhead_comb, muirheadKOIs, lcpath, direct, nbuffer, nlinfit, nlive=1000, nsupersample=29, exptimesupersample=0.0201389, show_plot=False, juliet_phase=False, std=1, lim=0.12):

    import os

    star = KeplerStar(int(np.floor(float(nkoi))))
    star.get_stellar_params(isodf)
    koi = KOI(nkoi, int(np.floor(float(nkoi))), isodf)
    koi.get_KIC(muirhead_comb)
    koi.planet_params_from_archive(spectplanets)
    koi.calc_a(koi.mstar, koi.rstar)

    KICs = np.sort(np.unique(np.array(muirhead_comb['KIC'])))
    KOIs = np.sort(np.unique(np.array(muirhead_comb['KOI'])))

    files = get_lc_files(koi.KIC, KICs, lcpath)


    koi.get_stitched_lcs(files)
    koi.get_midpoints()
    koi.remove_oot_data(nbuffer, nlinfit)


    if juliet_phase==True:
        koi.phase_intransit = juliet.utils.get_phases(koi.time_intransit-2454900, koi.period, koi.epoch-2454900)

    plt.scatter(koi.phase_intransit, koi.flux_intransit, s=1, c='k')

    plt.savefig(direct + 'foldedtransit.png')
    if show_plot == True:
        plt.show()

    dataset, results = koi.do_ttv_tfit_juliet(direct, nsupersample=nsupersample, exptimesupersample=exptimesupersample, nlive=nlive)
    koi.calc_durations()
    koi.get_gs()
    koi.do_eccfit(direct)

    return dataset, results, koi.e_dist, koi.w_dist



def run_multi_tfit_juliet(nkois, isodf, spectplanets, muirhead_comb, muirheadKOIs, lcpath, direct, nbuffer, nlinfit, nsupersample=29, exptimesupersample=0.0201389, nlive=1000, show_plot=False):

    import os

    star = KeplerStar(int(np.floor(float(nkois[0]))))
    star.get_stellar_params(isodf)
    multikoi = multiKOI(nkois, int(np.floor(float(nkois[0]))), isodf)

    multikoi.get_KIC(muirhead_comb)
    multikoi.planet_params_from_archive(spectplanets)
    multikoi.calc_a(multikoi.mstar, multikoi.rstar)

    KICs = np.sort(np.unique(np.array(muirhead_comb['KIC'])))
    KOIs = np.sort(np.unique(np.array(muirhead_comb['KOI'])))

    files = get_lc_files(multikoi.KIC, KICs, lcpath)

    multikoi.get_stitched_lcs(files)
    multikoi.get_midpoints()
    print(len(multikoi.midpoints))
    multikoi.remove_oot_data(nbuffer, nlinfit)
    print(len(multikoi.midpoints), '/')

    print(multikoi.periods)

    dataset, results = multikoi.do_multi_tfit_juliet(direct, nsupersample=None, exptimesupersample=None)
    multikoi.calc_durations()
    multikoi.get_gs()
    multikoi.do_eccfit(direct)

    return dataset, results, multikoi.e_dist, multikoi.w_dist


def run_multi_ttv_tfit_juliet(nkois, isodf, spectplanets, muirhead_comb, muirheadKOIs, lcpath, direct, nbuffer, nlinfit, nsupersample=29, exptimesupersample=0.0201389, nlive=1000, show_plot=False):

    import os

    star = KeplerStar(int(np.floor(float(nkois[0]))))
    star.get_stellar_params(isodf)
    multikoi = multiKOI(nkois, int(np.floor(float(nkois[0]))), isodf)

    multikoi.get_KIC(muirhead_comb)
    multikoi.planet_params_from_archive(spectplanets)
    multikoi.calc_a(multikoi.mstar, multikoi.rstar)

    KICs = np.sort(np.unique(np.array(muirhead_comb['KIC'])))
    KOIs = np.sort(np.unique(np.array(muirhead_comb['KOI'])))

    files = get_lc_files(multikoi.KIC, KICs, lcpath)

    multikoi.get_stitched_lcs(files)
    multikoi.get_midpoints()
    print(len(multikoi.midpoints))
    multikoi.remove_oot_data(nbuffer, nlinfit)
    print(len(multikoi.midpoints), '/')

    print(multikoi.periods)

    dataset, results = multikoi.do_multi_ttv_tfit_juliet(direct, nsupersample=None, exptimesupersample=None)
    multikoi.calc_durations()
    multikoi.get_gs()
    multikoi.do_eccfit(direct)

    return dataset, results, multikoi.e_dist, multikoi.w_dist




def remove_oot_data_multi(time, flux,flux_err, midpoints, nbuffer, nlinfit):

    tbjd = []
    tnorm = []
    fl = []
    fr = []

    for i in range(len(midpoints)):

        try:
            m, b, t1bjd, t1, fnorm, fe1 = do_linfit(time, flux, flux_err, midpoints[i], nbuffer, nlinfit)
            if np.isnan(fnorm).any()==False:
                tbjd.append(t1bjd)
                tnorm.append(t1)
                fl.append(fnorm)
                fr.append(fe1)

        except TypeError:
            continue

    time_intransit = np.array(tbjd).flatten()
    phase_intransit = np.array(tnorm).flatten()
    flux_intransit = np.array(fl).flatten()
    fluxerr_intransit = np.array(fr).flatten()

    return time_intransit, phase_intransit, flux_intransit, fluxerr_intransit


class KeplerStar:

    def __init__(self, StarKOI):
        self.StarKOI = StarKOI

    def get_stellar_params(self, isodf):

        self.isodf = isodf
        self.mstar, self.mstar_err, self.rstar, self.rstar_err = read_stellar_params(self.isodf)
        self.rho_star_dist, self.mass, self.radius, self.rho_star = get_rho_star(self.mstar, self.mstar_err, self.rstar, self.rstar_err, arrlen=1000)

    def get_KIC(self, muirhead_comb):
        "Gets KIC number for KOI (star)."
        self.KIC = muirhead_comb[muirhead_comb['KOI'] == str(int(self.StarKOI))].KIC.item()


class KOI(KeplerStar):

    def __init__(self, nkoi, StarKOI, isodf):

        super().__init__(StarKOI)
        super().get_stellar_params(isodf)

        self.nkoi = nkoi

        if len(str(self.nkoi)) == 4:
            self.kepoiname = 'K0000' + str(self.nkoi)
        if len(str(self.nkoi)) == 5:
            self.kepoiname = 'K000' + str(self.nkoi)
        elif len(str(self.nkoi)) == 6:
            self.kepoiname = 'K00' + str(self.nkoi)
        elif len(str(self.nkoi)) == 7:
            self.kepoiname = 'K0' + str(self.nkoi)
        else:
            self.kepoiname = None


    def get_KIC(self, muirhead_comb):
        "Gets KIC number for KOI (star)."
        self.KIC = muirhead_comb[muirhead_comb['KOI'] == str(int(np.floor(float(self.nkoi))))].KIC.item()

    def planet_params_from_archive(self, df):
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

        self.period = float(df.loc[df['kepoi_name'] == self.kepoiname].koi_period) #period (days)
        self.period_uerr = float(df.loc[df['kepoi_name'] == self.kepoiname].koi_period_err1) #period upper error (days)
        self.period_lerr = float(df.loc[df['kepoi_name'] == self.kepoiname].koi_period_err2) #period lower error (days)

        self.rprs = float(df.loc[df['kepoi_name'] == self.kepoiname].koi_ror) #planet rad/stellar rad
        self.rprs_uerr = float(df.loc[df['kepoi_name'] == self.kepoiname].koi_ror_err1) #planet rad upper error (days)
        self.rprs_lerr = float(df.loc[df['kepoi_name'] == self.kepoiname].koi_ror_err2) #planet rad lower error (days)

        self.a_rs = float(df.loc[df['kepoi_name'] == self.kepoiname].koi_dor) #semi-major axis/r_star (a on Rstar)
        self.a_rs_uerr = float(df.loc[df['kepoi_name'] == self.kepoiname].koi_dor_err1) #semi-major axis/r_star upper error
        self.a_rs_lerr = float(df.loc[df['kepoi_name'] == self.kepoiname].koi_dor_err2) #semi-major axis/r_star upper error

        self.i = float(df.loc[df['kepoi_name'] == self.kepoiname].koi_incl) #inclination (degrees)

        self.e = float(df.loc[df['kepoi_name'] == self.kepoiname].koi_eccen) #eccentricity (assumed 0)
        self.w = float(df.loc[df['kepoi_name'] == self.kepoiname].koi_longp) #longtitude of periastron (assumed 0)

        self.epoch = float(df.loc[df['kepoi_name'] == self.kepoiname].koi_time0) #transit epoch (BJD)

        self.dur = float(df.loc[df['kepoi_name'] == self.kepoiname].koi_duration) #transit epoch (BJD)

    def calc_a(self, smass, srad):
        """Calculates semi-major axis from planet period and stellar mass
        Kepler's 3rd law

        Parameters
        ----------
        period: float
            Planet period (SECONDS)
        smass: float
            Stellar mass (Msol)
        srad: float
            Stellar radius (Rsol)

        Returns
        -------
        a: float
            a/Rs: Semi-major axis of planet's orbit (units of stellar radii)
        """

        import scipy.constants as c

        smass = smass*1.9885e30  # Solar mass (kg)
        srad = srad*696.34e6 # Solar radius (m)
        period = self.period*86400.0

        a = np.cbrt((period**2*c.G*smass)/(4*np.pi**2*srad**3)) # a on rstar

        self.a_rs = a

        return a

    def get_stitched_lcs(self, files):
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

            hdu.close()

        self.hdus = hdus

        self.time = np.array([element for sublist in time for element in sublist])
        self.flux = np.array([element for sublist in flux for element in sublist])
        self.flux_err = np.array([element for sublist in flux_err for element in sublist])


        self.starttimes = starttimes
        self.stoptimes = stoptimes




    def normalize_flux(self):
        """Normalizes flux array."""

        fmed = np.nanmedian(self.flux)
        self.flux = self.flux/fmed
        self.flux_err = self.flux_err/fmed


    def get_midpoints(self):

        starttime = np.min(self.starttimes)
        stoptime = np.max(self.stoptimes)

        midpoints = np.concatenate((np.arange(self.epoch, starttime, -self.period), np.arange(self.epoch, stoptime, self.period)))
        midpoints = np.sort(midpoints)
        midpoints = np.unique(midpoints)

        self.midpoints = midpoints


    def remove_oot_data(self, nbuffer, nlinfit, std=1e6, lim=1e6):

        tbjd = []
        tnorm = []
        fl = []
        fr = []

        for i in range(len(self.midpoints)):
        #for i in range(10):

            try:
                m, b, t1bjd, t1, fnorm, fe1 = do_linfit(self.time, self.flux, self.flux_err, self.midpoints[i], nbuffer, nlinfit)

                if np.isnan(fnorm).any() == False:
                    if np.std(np.abs(t1)) > std or np.any(np.abs(t1)>lim):
                        continue
                    tbjd.append(list(t1bjd))
                    tnorm.append(list(t1))
                    fl.append(list(fnorm))
                    fr.append(list(fe1))

            except TypeError:
                continue

        self.time_intransit = np.array([x for y in tbjd for x in y])
        self.phase_intransit = np.array([x for y in tnorm for x in y])
        self.flux_intransit = np.array([x for y in fl for x in y])
        self.fluxerr_intransit = np.array([x for y in fr for x in y])


    def do_tfit(self, nwalk, nsteps, ndiscard, directory, custom_guess=None, plot_Tburnin=False, plot_Tcorner=False, backend=True, reset_backend=True):

        if custom_guess is not None:
            guess_transit = custom_guess
        else:
            guess_transit=[self.period, self.rprs, self.a_rs, self.i, self.epoch]

        self.tfit_results, self.tfit_results_errs, self.per_dist, self.rprs_dist, self.ars_dist, self.i_dist, self.t0_dist = mcmc_fitter(guess_transit, self.time_intransit, self.flux_intransit, self.fluxerr_intransit, nwalk, nsteps, ndiscard, directory, plot_Tburnin=plot_Tburnin, plot_Tcorner=plot_Tcorner, backend=backend, reset_backend=reset_backend)

        self.tfit_period = mode(self.per_dist)
        self.tfit_rprs = mode(self.rprs_dist)
        self.tfit_ars = mode(self.ars_dist)
        self.tfit_i = mode(self.per_dist)
        self.tfit_t0 = mode(self.t0_dist)

    def calc_durations(self):

        self.T14_dist = get_T14(self.per_dist, self.rprs_dist, self.ars_dist, self.i_dist)
        self.T14_errs = get_sigmas(self.T14_dist)

        self.T23_dist = get_T23(self.per_dist, self.rprs_dist, self.ars_dist, self.i_dist)
        self.T23_errs = get_sigmas(self.T23_dist)


    def get_gs(self):


        self.g_dist, self.rho_circ = get_g_distribution(self.rho_star_dist, self.per_dist, self.rprs_dist, self.T14_dist, self.T23_dist)

        self.g_mean = mode(self.g_dist)
        self.g_sigma = np.nanmean(np.abs(get_sigmas(self.g_dist)))

    def do_eccfit(self, direct, arrlen=1000, savecsv=False, savepickle=True):

        import dynesty
        import scipy
        import random
        import pickle

        def loglike(theta):
            """The log-likelihood function."""

            w, e = theta

            model = (1+e*np.sin(w*(np.pi/180.)))/np.sqrt(1-e**2)
            sigma2 = self.g_sigma ** 2

            return -0.5 * np.sum((self.g_mean - model) ** 2 / sigma2 + np.log(sigma2))

        def betae_prior_transform(utheta):
            """Beta-distribution eccentricity prior"""

            uw, ue = utheta
            w = 360.*uw-90.

            a, b = 0.867, 3.03
            e = scipy.stats.beta.ppf(ue, a, b)

            return w, e

        dsampler = dynesty.DynamicNestedSampler(loglike, betae_prior_transform, ndim=2, bound='multi', sample='rwalk')
        dsampler.run_nested()

        ewdres = dsampler.results

        edist  = random.choices(ewdres.samples[:,1], k=arrlen)
        wdist  = random.choices(ewdres.samples[:,0], k=arrlen)

        self.e_dist = edist
        self.w_dist = wdist
        self.e = mode(edist)
        self.w = mode(wdist)

        if savecsv == True:
            if direct is not None:
                np.savetxt(direct + 'edist.csv', edist, delimiter=',')
                np.savetxt(direct + 'wdist.csv', wdist, delimiter=',')
                np.savetxt(direct + 'T14dist.csv', T14dist, delimiter=',')
                np.savetxt(direct + 'T23dist.csv', T23dist, delimiter=',')
                np.savetxt(direct + 'gdist.csv', gs, delimiter=',')
            else:
                print('You must specify a directory to save csv files.')

        if savepickle == True:
            with open(direct + 'kepewdres.pickle', 'wb') as f:
                pickle.dump(ewdres, f, pickle.HIGHEST_PROTOCOL)



    def do_tfit_juliet(self, direct, nsupersample=29, exptimesupersample=0.0201389, nlive=1000, priors=None, reset_out=False):

        import juliet

        times, fluxes, fluxes_error = {},{},{}
        times['KEPLER'], fluxes['KEPLER'], fluxes_error['KEPLER'] = self.time_intransit, self.flux_intransit, self.fluxerr_intransit

        if priors==None:
            priors = {}

            params = ['P_p1','t0_p1','p_p1','b_p1','q1_KEPLER','q2_KEPLER','ecc_p1','omega_p1',\
                          'a_p1', 'mdilution_KEPLER', 'mflux_KEPLER', 'sigma_w_KEPLER']

            dists = ['normal','normal','normal','uniform','uniform','uniform','fixed','fixed',\
                             'truncatednormal', 'fixed', 'normal', 'loguniform']

            hyperps = [[self.period,0.01], [self.epoch,0.1], [self.rprs, 0.005], [0.,1.], [0., 1.], [0., 1.], 0.0, 90., [self.a_rs, 0.5*self.a_rs, 0., 200.], 1.0, [0.,0.1], [0.1, 1000.]]

            # Populate the priors dictionary:
            for param, dist, hyperp in zip(params, dists, hyperps):
                priors[param] = {}
                priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

            # Perform juliet fit:
            if reset_out==True:
                import os
                if os.path.exists(direct):
                    os.rmdir(direct)


            if nsupersample==None or exptimesupersample==None:
                dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, yerr_lc = fluxes_error, out_folder = direct)
            else:
                dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, yerr_lc = fluxes_error, out_folder = direct, lc_instrument_supersamp = ['KEPLER'], lc_n_supersamp = [nsupersample], lc_exptime_supersamp = [exptimesupersample])
            print("Fitting KOI " + str(self.nkoi))
            results = dataset.fit(ecclim=0., n_live_points=nlive)

            self.per_dist = results.posteriors['posterior_samples']['P_p1']
            self.rprs_dist = results.posteriors['posterior_samples']['p_p1']
            self.ars_dist = results.posteriors['posterior_samples']['a_p1']
            self.i_dist = np.arccos(results.posteriors['posterior_samples']['b_p1']*(1./self.ars_dist))*(180./np.pi)
            self.t0_dist = results.posteriors['posterior_samples']['t0_p1']

            self.tfit_period = mode(self.per_dist)
            self.tfit_rprs = mode(self.rprs_dist)
            self.tfit_ars = mode(self.ars_dist)
            self.tfit_i = mode(self.i_dist)
            self.tfit_t0 = mode(self.t0_dist)

            return dataset, results

    def do_ttv_tfit_juliet(self, direct, time=0., flux=0., flux_err=0., mptsintransit=0., nsupersample=29, exptimesupersample=0.0201389, nlive=1000, priors=None, reset_out=False):

        import juliet

        times, fluxes, fluxes_error = {},{},{}
        if type(time) == float:
            times['KEPLER'], fluxes['KEPLER'], fluxes_error['KEPLER'] = self.time_intransit, self.flux_intransit, self.fluxerr_intransit
        else:
            times['KEPLER'], fluxes['KEPLER'], fluxes_error['KEPLER'] = time, flux, flux_err

        if priors==None:
            priors = {}

            params = ['p_p1','b_p1','q1_KEPLER','q2_KEPLER','ecc_p1','omega_p1',\
                          'a_p1', 'mdilution_KEPLER', 'mflux_KEPLER', 'sigma_w_KEPLER']

            dists = ['normal','uniform','uniform','uniform','fixed','fixed',\
                             'truncatednormal', 'fixed', 'normal', 'loguniform']

            hyperps = [[self.rprs, 0.005], [0.,1.], [0., 1.], [0., 1.], 0.0, 90., [self.a_rs, 0.5*self.a_rs, 0., 200.], 1.0, [0.,0.1], [0.1, 1000.]]

            ttvparams = []
            ttvdists = []
            ttvhyperps = []

            if type(mptsintransit)==float:
                mptsintransit = self.midpoints
            for mpt in range(1, len(mptsintransit)+1):
                ttvparams.append('T_p1_KEPLER_' + str(mpt))
                ttvdists.append('truncatednormal')
                ttvhyperps.append([self.midpoints[mpt-1], 0.2, self.midpoints[mpt-1]-1.0, self.midpoints[mpt-1]+1.0])

            params = params + ttvparams
            dists = dists +ttvdists
            hyperps = hyperps + ttvhyperps

            # Populate the priors dictionary:
            for param, dist, hyperp in zip(params, dists, hyperps):
                priors[param] = {}
                priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

            # Perform juliet fit:
            if reset_out==True:
                import os
                if os.path.exists(direct):
                    os.rmdir(direct)


            if nsupersample==None or exptimesupersample==None:
                dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, yerr_lc = fluxes_error, out_folder = direct)
            else:
                dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, yerr_lc = fluxes_error, out_folder = direct, lc_instrument_supersamp = ['KEPLER'], lc_n_supersamp = [nsupersample], lc_exptime_supersamp = [exptimesupersample])
            print("Fitting KOI " + str(self.nkoi))
            results = dataset.fit(ecclim=0., n_live_points=nlive)

            self.per_dist = results.posteriors['posterior_samples']['P_p1']
            self.rprs_dist = results.posteriors['posterior_samples']['p_p1']
            self.ars_dist = results.posteriors['posterior_samples']['a_p1']
            self.i_dist = np.arccos(results.posteriors['posterior_samples']['b_p1']*(1./self.ars_dist))*(180./np.pi)
            self.t0_dist = results.posteriors['posterior_samples']['t0_p1']

            self.tfit_period = mode(self.per_dist)
            self.tfit_rprs = mode(self.rprs_dist)
            self.tfit_ars = mode(self.ars_dist)
            self.tfit_i = mode(self.i_dist)
            self.tfit_t0 = mode(self.t0_dist)

            return dataset, results


class multiKOI(KeplerStar):

    """For multi-planet systems, fitting rho instead of a/Rs"""

    def __init__(self, nkois, StarKOI, isodf):

        super().__init__(StarKOI)
        super().get_stellar_params(isodf)

        # nkois is a list of KOIs in system in order of shortest-longest period
        # e.g. [248.03, 248.01, 248.02, 248.04]
        self.nkois = nkois
        self.nplanets = len(self.nkois)
        self.kepoinames = []

        for nkoi in self.nkois:
            if len(str(nkoi)) == 4:
                self.kepoinames.append('K0000' + str(nkoi))
            if len(str(nkoi)) == 5:
                self.kepoinames.append('K000' + str(nkoi))
            elif len(str(nkoi)) == 6:
                self.kepoinames.append('K00' + str(nkoi))
            elif len(str(nkoi)) == 7:
                self.kepoinames.append('K0' + str(nkoi))
            else:
                self.kepoinames.append(None)


    def get_KIC(self, muirhead_comb):
        "Gets KIC number for KOI (star)."
        self.KIC = muirhead_comb[muirhead_comb['KOI'] == str(int(np.floor(float(self.nkois[0]))))].KIC.item()

    def planet_params_from_archive(self, df):
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

        self.periods, self.periods_uerr, self.periods_lerr, self.rprs, self.rprs_uerr, self.rprs_lerr, self.a_rs, self.a_rs_uerr, self.a_rs_lerr, self.incs, self.es, self.ws, self.epochs, self.durs = [], [], [], [], [], [], [], [], [], [], [], [], [], []

        for p in range(self.nplanets):

            self.periods.append(float(df.loc[df['kepoi_name'] == self.kepoinames[p]].koi_period)) #period (days)
            self.periods_uerr.append(float(df.loc[df['kepoi_name'] == self.kepoinames[p]].koi_period_err1)) #period upper error (days)
            self.periods_lerr.append(float(df.loc[df['kepoi_name'] == self.kepoinames[p]].koi_period_err2)) #period lower error (days)

            self.rprs.append(float(df.loc[df['kepoi_name'] == self.kepoinames[p]].koi_ror)) #planet rad/stellar rad
            self.rprs_uerr.append(float(df.loc[df['kepoi_name'] == self.kepoinames[p]].koi_ror_err1)) #planet rad upper error (days)
            self.rprs_lerr.append(float(df.loc[df['kepoi_name'] == self.kepoinames[p]].koi_ror_err2)) #planet rad lower error (days)

            self.a_rs.append(float(df.loc[df['kepoi_name'] == self.kepoinames[p]].koi_dor)) #semi-major axis/r_star (a on Rstar)
            self.a_rs_uerr.append(float(df.loc[df['kepoi_name'] == self.kepoinames[p]].koi_dor_err1)) #semi-major axis/r_star upper error
            self.a_rs_lerr.append(float(df.loc[df['kepoi_name'] == self.kepoinames[p]].koi_dor_err2)) #semi-major axis/r_star upper error

            self.incs.append(float(df.loc[df['kepoi_name'] == self.kepoinames[p]].koi_incl)) #inclination (degrees)

            self.es.append(float(df.loc[df['kepoi_name'] == self.kepoinames[p]].koi_eccen)) #eccentricity (assumed 0)
            self.ws.append(float(df.loc[df['kepoi_name'] == self.kepoinames[p]].koi_longp)) #longtitude of periastron (assumed 0)

            self.epochs.append(float(df.loc[df['kepoi_name'] == self.kepoinames[p]].koi_time0)) #transit epoch (BJD)

            self.durs.append(float(df.loc[df['kepoi_name'] == self.kepoinames[p]].koi_duration)) #transit epoch (BJD)

    def calc_a(self, smass, srad):
        """Calculates semi-major axis from planet period and stellar mass
        Kepler's 3rd law

        Parameters
        ----------
        period: float
            Planet period (SECONDS)
        smass: float
            Stellar mass (Msol)
        srad: float
            Stellar radius (Rsol)

        Returns
        -------
        a: float
            a/Rs: Semi-major axis of planet's orbit (units of stellar radii)
        """

        import scipy.constants as c

        smass = smass*1.9885e30  # Solar mass (kg)
        srad = srad*696.34e6 # Solar radius (m)

        for p in range(self.nplanets):
            period = self.periods[p]*86400.0
            a = np.cbrt((period**2*c.G*smass)/(4*np.pi**2*srad**3)) # a on rstar
            self.a_rs[p] = a

    def get_stitched_lcs(self, files):
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

            hdu.close()

        self.hdus = hdus

        self.time = np.array([element for sublist in time for element in sublist])
        self.flux = np.array([element for sublist in flux for element in sublist])
        self.flux_err = np.array([element for sublist in flux_err for element in sublist])

        # plt.plot(time, flux)
        # plt.show()

        # tnans = np.argwhere(np.isnan(time))
        # fnans = np.argwhere(np.isnan(flux))
        # fenans = np.argwhere(np.isnan(flux_err))
        #
        # nans = np.concatenate((tnans, fnans, fenans))
        # self.time = np.delete(time, nans)
        # self.flux = np.delete(flux, nans)
        # self.flux_err = np.delete(flux_err, nans)


        self.starttimes = starttimes
        self.stoptimes = stoptimes




    def normalize_flux(self):
        """Normalizes flux array."""

        fmed = np.nanmedian(self.flux)
        self.flux = self.flux/fmed
        self.flux_err = self.flux_err/fmed


    def get_midpoints(self):

        starttime = min(self.starttimes)
        stoptime = max(self.stoptimes)

        self.midpoints = []

        for p in range(self.nplanets):
            midpoints = np.concatenate((np.arange(self.epochs[p], starttime, -self.periods[p]), np.arange(self.epochs[p], stoptime, self.periods[p])))
            midpoints = np.sort(midpoints)
            midpoints = list(np.unique(midpoints))
            self.midpoints.append(midpoints)

        self.divided_midpoints = self.midpoints

        self.midpoints = [y for x in self.midpoints for y in x]


    def remove_oot_data(self, nbuffer, nlinfit):

        tbjd = []
        tnorm = []
        fl = []
        fr = []

        for idx in range(1, len(self.midpoints)):
            if (self.midpoints[idx]-self.midpoints[idx-1]) < 0.08333:
                self.midpoints[idx-1] = np.nan
                self.midpoints[idx] = np.nan

        self.midpoints = np.array(self.midpoints)
        self.midpoints = self.midpoints[~np.isnan(self.midpoints)]

        for i in range(len(self.midpoints)):

            try:
                m, b, t1bjd, t1, fnorm, fe1 = do_linfit(self.time, self.flux, self.flux_err, self.midpoints[i], nbuffer, nlinfit)

                if np.isnan(fnorm).any() == False:
                    tbjd.append(list(t1bjd))
                    tnorm.append(list(t1))
                    fl.append(list(fnorm))
                    fr.append(list(fe1))

            except TypeError:
                continue

        self.time_intransit = np.array([x for y in tbjd for x in y])
        self.phase_intransit = np.array([x for y in tnorm for x in y])
        self.flux_intransit = np.array([x for y in fl for x in y])
        self.fluxerr_intransit = np.array([x for y in fr for x in y])


    def calc_durations(self):

        self.T14_dists, self.T14_errs = [], []
        self.T23_dists, self.T23_errs = [], []

        for idx in range(self.nplanets):

            self.T14_dists.append(get_T14(self.per_dists[idx], self.rprs_dists[idx], self.a_dists[idx], self.i_dists[idx]))
            self.T14_errs.append(get_T14(self.per_dists[idx], self.rprs_dists[idx], self.a_dists[idx], self.i_dists[idx]))

            self.T23_dists.append(get_T23(self.per_dists[idx], self.rprs_dists[idx], self.a_dists[idx], self.i_dists[idx]))
            self.T23_errs.append(get_T23(self.per_dists[idx], self.rprs_dists[idx], self.a_dists[idx], self.i_dists[idx]))


    def get_gs(self):

        self.g_dists, self.g_means, self.g_sigmas = [], [], []

        for idx in range(self.nplanets):

            gdist, rhocirc = get_g_distribution(self.rho_star_dist, self.per_dists[idx], self.rprs_dists[idx], self.T14_dists[idx], self.T23_dists[idx])
            self.g_dists.append(gdist)

            self.g_means.append(mode(gdist))
            self.g_sigmas.append(np.nanmean(np.abs(get_sigmas(gdist))))

    def do_eccfit(self, direct, arrlen=1000, savecsv=False, savepickle=True):

        import dynesty
        import scipy
        import random
        import pickle

        def loglike(theta):
            """The log-likelihood function."""

            w, e = theta

            model = (1+e*np.sin(w*(np.pi/180.)))/np.sqrt(1-e**2)
            sigma2 = self.g_sigmas[idx] ** 2

            return -0.5 * np.sum((self.g_means[idx] - model) ** 2 / sigma2 + np.log(sigma2))

        def betae_prior_transform(utheta):
            """Uniform eccentricity prior"""

            uw, ue = utheta
            w = 360.*uw-90.

            a, b = 0.867, 3.03
            e = scipy.stats.beta.ppf(ue, a, b)

            return w, e

        self.e_dist, self.w_dist = [], []
        self.e, self.w = [], []


        for idx in range(self.nplanets):

            dsampler = dynesty.DynamicNestedSampler(loglike, betae_prior_transform, ndim=2, bound='multi', sample='rwalk')
            dsampler.run_nested()

            ewdres = dsampler.results

            edist  = random.choices(ewdres.samples[:,1], k=arrlen)
            wdist  = random.choices(ewdres.samples[:,0], k=arrlen)

            self.e_dist.append(edist)
            self.w_dist.append(wdist)
            self.e.append(mode(edist))
            self.w.append(mode(wdist))

            if savecsv == True:
                if direct is not None:
                    np.savetxt(direct + 'edist_p' + str(idx+1) + '.csv', edist, delimiter=',')
                    np.savetxt(direct + 'wdist_p' + str(idx+1) + '.csv', wdist, delimiter=',')
                    np.savetxt(direct + 'T14dist_p' + str(idx+1) + '.csv', T14dist, delimiter=',')
                    np.savetxt(direct + 'T23dist_p' + str(idx+1) + '.csv', T23dist, delimiter=',')
                    np.savetxt(direct + 'gdist_p' + str(idx+1) + '.csv', gs, delimiter=',')
                else:
                    print('You must specify a directory to save csv files.')

            if savepickle == True:
                with open(direct + 'kepewdres_p' + str(idx+1) + '.pickle', 'wb') as f:
                    pickle.dump(ewdres, f, pickle.HIGHEST_PROTOCOL)

    def do_multi_tfit_juliet(self, direct, nsupersample=29, exptimesupersample=0.0201389, nlive=1000, priors=None, reset_out=False):

        import juliet
        import pickle

        times, fluxes, fluxes_error = {},{},{}
        times['KEPLER'], fluxes['KEPLER'], fluxes_error['KEPLER'] = self.time_intransit, self.flux_intransit, self.fluxerr_intransit

        if priors==None:

            priors = {}

            params = []
            dists = []
            hyperps = []

            for planet_number in range(1, self.nplanets+1):
                params.append('P_p' + str(planet_number))
                dists.append('normal')
                hyperps.append([self.periods[planet_number-1],0.1])
                params.append('t0_p' + str(planet_number))
                dists.append('normal')
                hyperps.append([self.epochs[planet_number-1],0.1])
                params.append('p_p' + str(planet_number))
                dists.append('normal')
                hyperps.append([self.rprs[planet_number-1],0.005])
                params.append('a_p' + str(planet_number))
                dists.append('truncatednormal')
                hyperps.append([self.a_rs[planet_number-1],0.5*self.a_rs[planet_number-1], 0., 200.])
                params.append('b_p' + str(planet_number))
                dists.append('uniform')
                hyperps.append([0.,1.])
                params.append('ecc_p' + str(planet_number))
                dists.append('fixed')
                hyperps.append(0.)
                params.append('omega_p' + str(planet_number))
                dists.append('fixed')
                hyperps.append(90.)

            params = params + ['q1_KEPLER','q2_KEPLER', 'mdilution_KEPLER', 'mflux_KEPLER', 'sigma_w_KEPLER']
            dists = dists + ['uniform','uniform', 'fixed', 'normal', 'loguniform']
            hyperps = hyperps + [[0., 1.], [0., 1.], 1.0, [0.,0.1], [0.1, 1000.]]

            # Populate the priors dictionary:
            for param, dist, hyperp in zip(params, dists, hyperps):
                priors[param] = {}
                priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

            # Perform juliet fit:
            if reset_out==True:
                import os
                if os.path.exists(direct):
                    os.rmdir(direct)

            if nsupersample==None or exptimesupersample==None:
                dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, yerr_lc = fluxes_error, out_folder = direct)
            else:
                dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, yerr_lc = fluxes_error, out_folder = direct, lc_instrument_supersamp = ['KEPLER'], lc_n_supersamp = [nsupersample], lc_exptime_supersamp = [exptimesupersample])

            print("Fitting KOI " + str(self.StarKOI))
            results = dataset.fit(ecclim=0., n_live_points=nlive)

            with open(direct + 'results.pickle', 'wb') as f:
                pickle.dump(results, f)

            with open(direct + 'dataset.pickle', 'wb') as f:
                pickle.dump(dataset, f)


            self.per_dists, self.rprs_dists, self.a_dists, self.i_dists, self.t0_dists = [], [], [], [], []
            self.tfit_periods, self.tfit_rprs, self.tfit_as, self.tfit_is, self.tfit_t0s = [], [], [], [], []

            for planet_number in range(1, self.nplanets+1):

                self.per_dists.append(results.posteriors['posterior_samples']['P_p' + str(planet_number)])
                self.rprs_dists.append(results.posteriors['posterior_samples']['p_p' + str(planet_number)])
                self.a_dists.append(results.posteriors['posterior_samples']['a_p' + str(planet_number)])
                self.i_dists.append(np.arccos(results.posteriors['posterior_samples']['b_p1']*(1./results.posteriors['posterior_samples']['a_p' + str(planet_number)]))*(180./np.pi))
                self.t0_dists.append(results.posteriors['posterior_samples']['t0_p' + str(planet_number)])

                self.tfit_periods.append(mode(results.posteriors['posterior_samples']['P_p' + str(planet_number)]))
                self.tfit_rprs.append(mode(results.posteriors['posterior_samples']['p_p' + str(planet_number)]))
                self.tfit_as.append(mode(results.posteriors['posterior_samples']['a_p' + str(planet_number)]))
                self.tfit_is.append(mode(np.arccos(results.posteriors['posterior_samples']['b_p1']*(1./results.posteriors['posterior_samples']['a_p' + str(planet_number)]))*(180./np.pi)))
                self.tfit_t0s.append(mode(results.posteriors['posterior_samples']['t0_p' + str(planet_number)]))

            return dataset, results


    def do_multi_ttv_tfit_juliet(self, direct, mptsintransit=0., nsupersample=29, exptimesupersample=0.0201389, nlive=1000, priors=None, reset_out=False):

        import juliet
        import pickle

        times, fluxes, fluxes_error = {},{},{}
        times['KEPLER'], fluxes['KEPLER'], fluxes_error['KEPLER'] = self.time_intransit, self.flux_intransit, self.fluxerr_intransit

        if priors==None:

            priors = {}

            params = []
            dists = []
            hyperps = []

            for planet_number in range(1, self.nplanets+1):
                params.append('P_p' + str(planet_number))
                dists.append('normal')
                hyperps.append([self.periods[planet_number-1],0.1])
                params.append('t0_p' + str(planet_number))
                dists.append('normal')
                hyperps.append([self.epochs[planet_number-1],0.1])
                params.append('p_p' + str(planet_number))
                dists.append('normal')
                hyperps.append([self.rprs[planet_number-1],0.005])
                params.append('a_p' + str(planet_number))
                dists.append('truncatednormal')
                hyperps.append([self.a_rs[planet_number-1],0.5*self.a_rs[planet_number-1], 0., 200.])
                params.append('b_p' + str(planet_number))
                dists.append('uniform')
                hyperps.append([0.,1.])
                params.append('ecc_p' + str(planet_number))
                dists.append('fixed')
                hyperps.append(0.)
                params.append('omega_p' + str(planet_number))
                dists.append('fixed')
                hyperps.append(90.)

                for mpt in range(1, len(self.divided_midpoints[planet_number-1])+1):
                    ttvparams.append('T_p' + str(planet_number) + '_KEPLER_' + str(mpt))
                    ttvdists.append('truncatednormal')
                    ttvhyperps.append([self.divided_midpoints[planet_number-1][mpt-1], 0.2, self.divided_midpoints[planet_number-1][mpt-1]-1.0, self.divided_midpoints[planet_number-1][mpt-1]+1.0])

            params = params + ['q1_KEPLER','q2_KEPLER', 'mdilution_KEPLER', 'mflux_KEPLER', 'sigma_w_KEPLER']
            dists = dists + ['uniform','uniform', 'fixed', 'normal', 'loguniform']
            hyperps = hyperps + [[0., 1.], [0., 1.], 1.0, [0.,0.1], [0.1, 1000.]]

            ttvparams = []
            ttvdists = []
            ttvhyperps = []

            #if type(mptsintransit)==float:
            mptsintransit = self.midpoints



            params = params + ttvparams
            dists = dists + ttvdists
            hyperps = hyperps + ttvhyperps

            # Populate the priors dictionary:
            for param, dist, hyperp in zip(params, dists, hyperps):
                priors[param] = {}
                priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

            # Perform juliet fit:
            if reset_out==True:
                import os
                if os.path.exists(direct):
                    os.rmdir(direct)

            if nsupersample==None or exptimesupersample==None:
                dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, yerr_lc = fluxes_error, out_folder = direct)
            else:
                dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, yerr_lc = fluxes_error, out_folder = direct, lc_instrument_supersamp = ['KEPLER'], lc_n_supersamp = [nsupersample], lc_exptime_supersamp = [exptimesupersample])

            print("Fitting KOI " + str(self.StarKOI))
            results = dataset.fit(ecclim=0., n_live_points=nlive)

            with open(direct + 'results.pickle', 'wb') as f:
                pickle.dump(results, f)

            with open(direct + 'dataset.pickle', 'wb') as f:
                pickle.dump(dataset, f)


            self.per_dists, self.rprs_dists, self.a_dists, self.i_dists, self.t0_dists = [], [], [], [], []
            self.tfit_periods, self.tfit_rprs, self.tfit_as, self.tfit_is, self.tfit_t0s = [], [], [], [], []

            for planet_number in range(1, self.nplanets+1):

                self.per_dists.append(results.posteriors['posterior_samples']['P_p' + str(planet_number)])
                self.rprs_dists.append(results.posteriors['posterior_samples']['p_p' + str(planet_number)])
                self.a_dists.append(results.posteriors['posterior_samples']['a_p' + str(planet_number)])
                self.i_dists.append(np.arccos(results.posteriors['posterior_samples']['b_p1']*(1./results.posteriors['posterior_samples']['a_p' + str(planet_number)]))*(180./np.pi))
                self.t0_dists.append(results.posteriors['posterior_samples']['t0_p' + str(planet_number)])

                self.tfit_periods.append(mode(results.posteriors['posterior_samples']['P_p' + str(planet_number)]))
                self.tfit_rprs.append(mode(results.posteriors['posterior_samples']['p_p' + str(planet_number)]))
                self.tfit_as.append(mode(results.posteriors['posterior_samples']['a_p' + str(planet_number)]))
                self.tfit_is.append(mode(np.arccos(results.posteriors['posterior_samples']['b_p1']*(1./results.posteriors['posterior_samples']['a_p' + str(planet_number)]))*(180./np.pi)))
                self.tfit_t0s.append(mode(results.posteriors['posterior_samples']['t0_p' + str(planet_number)]))

            return dataset, results
