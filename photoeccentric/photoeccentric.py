import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import juliet

from .utils import *
from .stellardensity import *
from .lcfitter import *
from .eccentricity import *


def run_fit_juliet(nkoi, isodf, spectplanets, muirhead_comb, muirheadKOIs, lcpath, direct, nbuffer, nlinfit, nlive=1000, nsupersample=29, exptimesupersample=0.0201389, show_plot=False, juliet_phase=False, std=1, lim=0.12):
    """One step transit + eccentricity fitting for Kepler light curves."""

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

    dataset, results = koi.do_tfit_juliet(direct, nsupersample=nsupersample, exptimesupersample=exptimesupersample, nlive=nlive)
    koi.calc_durations()
    koi.get_gs()
    koi.do_eccfit(direct)

    return dataset, results, koi.e_dist, koi.w_dist

class KeplerStar:

    def __init__(self, StarKOI):
        self.StarKOI = StarKOI

    def get_stellar_params(self, isodf):
        """Gets stellar parameters from set of consistent stellar isochrones."""

        self.isodf = isodf
        self.mstar, self.mstar_err, self.rstar, self.rstar_err = read_stellar_params(self.isodf)
        self.rho_star_dist, self.mass, self.radius, self.rho_star = get_rho_star(self.mstar, self.mstar_err, self.rstar, self.rstar_err, arrlen=1000)

    def get_KIC(self, muirhead_comb):
        """Gets KIC number for KOI (star)."""
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

        self.epoch = float(df.loc[df['kepoi_name'] == self.kepoiname].koi_time0)-2454900. #transit epoch (BJD)

        self.dur = float(df.loc[df['kepoi_name'] == self.kepoiname].koi_duration) #transit epoch (BJD)

        self.ld1 = float(df.loc[df['kepoi_name'] == self.kepoiname].koi_ldm_coeff1) #ld coefficient 1
        self.ld2 = float(df.loc[df['kepoi_name'] == self.kepoiname].koi_ldm_coeff2) #ld coefficient 2

        self.b = float(df.loc[df['kepoi_name'] == self.kepoiname].koi_impact) #impact parameter


    def calc_a(self, smass, srad):
        """Calculates semi-major axis from planet period and stellar mass using Kepler's 3rd law

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

    def get_stitched_lcs(self, files, cadence_combine=False, record_bounds=False):
        """Stitches Kepler LCs from a list of fits files downloaded from MAST.

        Parameters:
        ----------
        files: List

        KIC: float
            KOI of target

        cadence_combine: boolean
            True if light curve files include both short- and long-cadence data. False otherwise.

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

        if cadence_combine==False:
            files = sorted(files)

        elif cadence_combine==True:

            ordered_dates = sorted([filename.split('-')[1].split('_')[0] for filename in files])

            sorted_files = []
            for i in range(len(ordered_dates)):
                for j in range(len(files)):
                    if ordered_dates[i] in files[j]:
                        sorted_files.append(files[j])

            files = sorted_files

        time = []
        flux = []
        flux_err = []
        hdus = []

        starttimes = []
        stoptimes = []

        if record_bounds==True:
            bounds = []

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

            if record_bounds==True:
                bounds.append([start, stop])

            hdu.close()

        if record_bounds==True:
            self.bounds = np.array(bounds)-2454900.

        self.hdus = hdus

        self.time = np.array([element for sublist in time for element in sublist])-2454900.
        self.flux = np.array([element for sublist in flux for element in sublist])
        self.flux_err = np.array([element for sublist in flux_err for element in sublist])


        self.starttimes = np.array(starttimes)-2454900.
        self.stoptimes = np.array(stoptimes)-2454900.




    def normalize_flux(self):
        """Normalizes flux array."""

        fmed = np.nanmedian(self.flux)
        self.flux = self.flux/fmed
        self.flux_err = self.flux_err/fmed


    def get_midpoints(self):
        """Calculates transit midpoints within time bounds of Kepler light curve."""

        starttime = np.min(self.starttimes)
        stoptime = np.max(self.stoptimes)

        midpoints = np.concatenate((np.arange(self.epoch, starttime, -self.period), np.arange(self.epoch, stoptime, self.period)))
        midpoints = np.sort(midpoints)
        midpoints = np.unique(midpoints)

        self.midpoints = midpoints


    def remove_oot_data(self, nbuffer, linearfit=False, nlinfit=None, include_nans=False, delete_nan_transits=False, simultaneous_midpoints=None):
        """Removes out-of-transit segments of Kepler light curves.
        Fits a linear model to out-of-transit points immediately surrounding each transit.
        Subtracts the linear model from each transit cutout.

        Parameters
        ----------
        nbuffer: int
            Number of flux points before and after transit midpoint to include in transit cut-out.
            e.g. if nbuffer = 7, function will preserve 7 flux points before and after each transit midpoint and discard the rest of light curve.
        nlinfit: int
            Number of flux points from each end of transit cutout to use in linear fit.
            e.g. if nbuffer = 7 and nlinfit = 5, function will use the 10 outermost flux points for linear fit.
        include_nans: boolean, default False
            Include nans in in-transit data?
        delete_nan_transits: boolean, default False
            Delete entire transit if includes nan flux value?

        Returns
        -------
        None

        """

        tbjd = []
        tnorm = []
        fl = []
        fr = []
        mpintransit = []

        for i in range(len(self.midpoints)):

            try:
                if simultaneous_midpoints is not None:
                    near_sim_mpt = find_nearest(simultaneous_midpoints, self.midpoints[i])
                    if abs(self.midpoints[i] - near_sim_mpt) < 3*0.0416667: # If transit midpoint is close to a simultaneous transit, then don't include that transit.
                        continue

                if linearfit==True:
                    assert nlinfit is not None, "If performing a linear fit, you must define nlinfit."
                    m, b, t1bjd, t1, fnorm, fe1 = do_linfit(self.time, self.flux, self.flux_err, self.midpoints[i], nbuffer, nlinfit)
                    if np.isnan(m) and np.isnan(t1) and np.isnan(fnorm):
                        print('Gap midpoint')
                        continue
                elif linearfit==False:
                    t1bjd, t1, fnorm, fe1 = cutout_no_linfit(self.time, self.flux, self.flux_err, self.midpoints[i], nbuffer)

                if include_nans==False:

                    if np.isnan(fnorm).any() == False:
                        tbjd.append(list(t1bjd))
                        tnorm.append(list(t1))
                        fl.append(list(fnorm))
                        fr.append(list(fe1))
                        mpintransit.append(self.midpoints[i])

                    if np.isnan(fnorm).any() == True:

                        if delete_nan_transits == True:
                            continue

                        elif delete_nan_transits == False:
                            naninds = []
                            for idx in range(len(t1bjd)):
                                if np.isnan(fnorm[idx]):
                                    naninds.append(idx)


                            tbjd.append(list(np.delete(t1bjd, naninds)))
                            tnorm.append(list(np.delete(t1, naninds)))
                            fl.append(list(np.delete(fnorm, naninds)))
                            fr.append(list(np.delete(fe1, naninds)))
                            mpintransit.append(self.midpoints[i])

                elif include_nans==True:
                    tbjd.append(list(t1bjd))
                    tnorm.append(list(t1))
                    fl.append(list(fnorm))
                    fr.append(list(fe1))
                    mpintransit.append(self.midpoints[i])

            except TypeError:
                continue

        self.time_intransit = np.array([x for y in tbjd for x in y])
        self.phase_intransit = np.array([x for y in tnorm for x in y])
        self.flux_intransit = np.array([x for y in fl for x in y])
        self.fluxerr_intransit = np.array([x for y in fr for x in y])
        self.midpoints_intransit = np.array(mpintransit)



    def calc_durations(self):
        """After fitting circular period, Rp/Rs, a/Rs, and inclination, calculates full and total transit duration of circular transit using Winn (2010) Eqs. 14, 15."""

        self.T14_dist = get_T14(self.per_dist, self.rprs_dist, self.ars_dist, self.i_dist)
        self.T14_errs = get_sigmas(self.T14_dist)

        self.T23_dist = get_T23(self.per_dist, self.rprs_dist, self.ars_dist, self.i_dist)
        self.T23_errs = get_sigmas(self.T23_dist)


    def get_gs(self, custom_rho_star="None"):
        """Calculates g using Dawson & Johsnon (2012) Eq. 6."""

        if type(custom_rho_star)==str:
            self.g_dist, self.rho_circ = get_g_distribution(self.rho_star_dist, self.per_dist, self.rprs_dist, self.T14_dist, self.T23_dist)
        else:
            self.g_dist, self.rho_circ = get_g_distribution(custom_rho_star, self.per_dist, self.rprs_dist, self.T14_dist, self.T23_dist)

        g_bins = np.arange(0,5,0.05)
        self.g_mean = mode(self.g_dist, bin_type='arr', bins=g_bins)
        self.g_sigma = np.nanmean(np.abs(get_sigmas(self.g_dist)))

    def do_eccfit(self, direct, arrlen=1000, bound='multi', sample='rwalk', savecsv=False, savepickle=True):
        """One-step eccentricity & longitude of periastron fitting from g. Nested sampling w/ dynesty.
        Using Dawson & Johnson (2012) Eq. 4.

        Parameters
        ----------
        direct: str
            Output path.
        arrlen: int, default 1000
            Number of (e, omega) posterior points to save.
        bound: default 'multi'
            dynesty bound
        sample: default 'rwalk'
            dynesty sampling method
        savecsv: boolean, default False
            Save e, w, transit durations, and g distributions as csvs?
        savepickle: boolean, default True
            Save dynesty results as pickle file?

        Returns
        -------
        None

        """

        import dynesty
        import scipy
        import random
        import pickle

        def loglike(theta):
            """The log-likelihood function."""

            w, e = theta

            model = (1+e*np.sin(w*(np.pi/180.)))/np.sqrt(1-e**2)
            sigma2 = self.g_sigma ** 2

            llike = -0.5 * np.sum((self.g_mean - model) ** 2 / sigma2 + np.log(sigma2))

            return llike

        def prior_transform(utheta):
            """Uniform eccentricity prior"""

            uw, ue = utheta

            # UNIFORM PRIOR #
            ew = [ue, uw*2*np.pi]

            e = float(ew[0])
            w = float(ew[1])*(180./np.pi)

            return w, e


        dsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim=2, bound=bound, sample=sample)
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



    def do_tfit_juliet(self, direct, nsupersample=29, exptimesupersample=0.0201389, nlive=400, priors=None, reset_out=False):
        """One-step Kepler transit light curve fitting. Using Juliet w/ pymultinest. Only fits circular transits (e, w fixed at 0!).

        Parameters
        ----------
        direct: str
            Output path.
        nsupersample: int, default 29
            Number of flux points over which to supersample, default 29 (for Kepler long cadence.)
        exptimesupersample: float, default 0.0201389 (29 min for Kepler long cadence)
            Time over which to supersample, default
        nlive: int, default 400
            Number of pymultinest live points
        priors: default None
            Custom transit fitting priors for juliet fit
        reset_out: boolean, default False
            Reset output directory?

        Returns
        -------
        dataset: juliet object
            Juliet dataset
        results: juliet object
            Juliet results


        """
        import juliet

        times, fluxes, fluxes_error = {},{},{}
        times['KEPLER'], fluxes['KEPLER'], fluxes_error['KEPLER'] = self.time_intransit, self.flux_intransit, self.fluxerr_intransit

        if priors==None:
            priors = {}

            params = ['P_p1','t0_p1','p_p1','b_p1','q1_KEPLER','q2_KEPLER','ecc_p1','omega_p1',\
                          'a_p1', 'mdilution_KEPLER', 'mflux_KEPLER', 'sigma_w_KEPLER']

            dists = ['normal','normal','normal','uniform','uniform','uniform','fixed','fixed',\
                             'uniform', 'fixed', 'normal', 'loguniform']

            hyperps = [[self.period,0.001], [self.epoch,0.001], [self.rprs, 0.001], [0.,1.], [0., 1.], [0., 1.], 0.0, 90., [0., 200.], 1.0, [0.,0.1], [0.1, 1000.]]

            for param, dist, hyperp in zip(params, dists, hyperps):
                priors[param] = {}
                priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

            if reset_out==True:
                import os
                if os.path.exists(direct):
                    os.rmdir(direct)


            if nsupersample==None or exptimesupersample==None:
                dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, yerr_lc = fluxes_error, out_folder = direct)
            else:
                dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, yerr_lc = fluxes_error, out_folder = direct, lc_instrument_supersamp = ['KEPLER'], lc_n_supersamp = [nsupersample], lc_exptime_supersamp = [exptimesupersample])

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

    def do_ttv_tfit_juliet(self, direct, time=0., flux=0., flux_err=0., mptsintransit=0., nsupersample=29, exptimesupersample=0.0201389, nlive=400, priors=None, reset_out=False):
        """One-step Kepler transit light curve fitting, for transits with TTVs. Using Juliet w/ pymultinest. Only fits circular transits (e, w fixed at 0!).
        Fixes TTV midpoints. Uses Kepler catalog orbital period + errorbars to simluate a fit period distribution.

        Parameters
        ----------
        direct: str
            Output path.
        mptsintransit: array or float, default 0.
            Input irregular transit midpoints with TTVs. If 0, uses default, calculated midpoints.
        nsupersample: int, default 29
            Number of flux points over which to supersample, default 29 (for Kepler long cadence.)
        exptimesupersample: float, default 0.0201389 (29 min for Kepler long cadence)
            Time over which to supersample, default
        nlive: int, default 400
            Number of pymultinest live points
        priors: default None
            Custom transit fitting priors for juliet fit
        reset_out: boolean, default False
            Reset output directory?

        Returns
        -------
        dataset: juliet object
            Juliet dataset
        results: juliet object
            Juliet results

        """

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
                             'uniform', 'fixed', 'normal', 'loguniform']

            hyperps = [[self.rprs, 0.001], [0.,1.], [0., 1.], [0., 1.], 0.0, 90., [0., 200.], 1.0, [0.,0.1], [0.1, 1000.]]

            ttvparams = []
            ttvdists = []
            ttvhyperps = []

            if type(mptsintransit)==float:
                mptsintransit = self.midpoints
            for mpt in range(1, len(mptsintransit)+1):
                ttvparams.append('T_p1_KEPLER_' + str(mpt))
                ttvdists.append('fixed')
                ttvhyperps.append(self.midpoints[mpt-1])

            params = params + ttvparams
            dists = dists + ttvdists
            hyperps = hyperps + ttvhyperps

            for param, dist, hyperp in zip(params, dists, hyperps):
                priors[param] = {}
                priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

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

            self.per_dist = np.random.normal(self.period, np.mean((abs(self.period_uerr), abs(self.period_lerr))), size=len(results.posteriors['posterior_samples']['p_p1']))
            self.rprs_dist = results.posteriors['posterior_samples']['p_p1']
            self.ars_dist = results.posteriors['posterior_samples']['a_p1']
            self.i_dist = np.arccos(results.posteriors['posterior_samples']['b_p1']*(1./self.ars_dist))*(180./np.pi)

            self.tfit_period = mode(self.per_dist)
            self.tfit_rprs = mode(self.rprs_dist)
            self.tfit_ars = mode(self.ars_dist)
            self.tfit_i = mode(self.i_dist)

            return dataset, results
