import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .utils import *
from .stellardensity import *
from .lcfitter import *
from .eccentricity import *


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
        self.KIC = muirhead_comb[muirhead_comb['KOI'] == float(str(self.StarKOI) + '.01')].KIC.item()


class KOI(KeplerStar):

    def __init__(self, nkoi, StarKOI):

        super().__init__(StarKOI)
        # Temporary: only invocate if we want to use interpolated Dartmouth isochrones!
        #super().get_stellar_params(isodf)

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

        self.lightcurvefiles = False

    
    def download_planet_params(self):
        """Download stellar parameters for a KOI from the Exoplanet Archive.

        """
        url, content = from_exoarchive(KOI=self.kepoiname)

        self.KIC = float(content['kepid'].values) #KIC
        self.kepname = str(content['kepler_name'].values) #Kepler Name

        self.period = float(content['koi_period'].values) #period (days)
        self.period_uerr = float(content['koi_period_err1'].values) #period upper error (days)
        self.period_lerr = float(content['koi_period_err2'].values) #period lower error (days)

        self.rprs = float(content['koi_ror'].values) #planet rad/stellar rad
        self.rprs_uerr = float(content['koi_ror_err1'].values) #planet rad upper error (days)
        self.rprs_lerr = float(content['koi_ror_err1'].values) #planet rad lower error (days)

        self.a_rs = float(content['koi_dor'].values) #semi-major axis/r_star (a on Rstar)
        self.a_rs_uerr = float(content['koi_dor_err1'].values) #semi-major axis/r_star upper error
        self.a_rs_lerr = float(content['koi_dor_err1'].values) #semi-major axis/r_star upper error

        self.i = float(content['koi_incl'].values) #inclination (degrees)

        self.e = float(content['koi_eccen'].values) #eccentricity (assumed 0)
        self.w = float(content['koi_longp'].values) #longtitude of periastron (assumed 0)

        self.epoch = float(content['koi_time0'].values)-2454900. #transit epoch (BJD)

        self.dur = float(content['koi_duration'].values) #transit epoch (BJD)

        self.ld1 = float(content['koi_ldm_coeff1'].values) #ld coefficient 1
        self.ld2 = float(content['koi_ldm_coeff2'].values) #ld coefficient 2

        self.b = float(content['koi_impact'].values) #impact parameter

        return content

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

    def download_lightcurves(self):
        """Download Kepler lightcurves from MAST for a KOI."""
    
        table = kepler_from_mast('KIC ' + str(int(self.KIC)))
        lcs = get_timeseries_files('mastDownload.tar.gz')

        self.lightcurvefiles = lcs


    def get_stitched_lcs(self, files=None, cadence_combine=False, record_bounds=False):
        """Stitches Kepler LCs from a list of fits files downloaded from MAST.

        Parameters:
        ----------
        files: list
            List of FITS file paths containing light curves
        KIC: float
            KOI of target
        cadence_combine: boolean
            True if light curve files include both short- a nd long-cadence data. False otherwise.
        record_bounds: boolean
            True if recording start and end times of each light curve segment is desired. False otherwise.

        Returns:
        -------
        None

        """

        from astropy.io import fits

        if self.lightcurvefiles != False:
            files = self.lightcurvefiles

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
        uflux = []
        uflux_err = []
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

            uflux.append(list(hdu[1].data['PDCSAP_FLUX']))
            uflux_err.append(list(hdu[1].data['PDCSAP_FLUX_ERR']))

            hdus.append(hdu)

            if record_bounds==True:
                bounds.append([start, stop])



        if record_bounds==True:
            self.bounds = np.array(bounds)-2454900.

        self.hdus = hdus

        self.time_quarters = np.array(time)
        self.flux_quarters = np.array(flux)
        self.flux_err_quarters = np.array(flux_err)

        self.time = np.array([element for sublist in time for element in sublist])-2454900.
        self.flux = np.array([element for sublist in flux for element in sublist])
        self.flux_err = np.array([element for sublist in flux_err for element in sublist])

        self.uflux = np.array([element for sublist in uflux for element in sublist])
        self.uflux_err = np.array([element for sublist in uflux_err for element in sublist])


        self.starttimes = np.array(starttimes)-2454900.
        self.stoptimes = np.array(stoptimes)-2454900.

        hdu.close()


    def normalize_flux(self):
        """Normalizes self.flux array."""

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

    def sigma_clip_quarters(self, sigma=6, maxiters=1):
        """Sigma clips light curve by quarter.
        Number of sigmas and max # of iterations allowed can be specified"""

        t = []
        f = []
        fe = []

        for q in range(len(self.time_quarters)):

            qlctime = np.array(self.time_quarters[q])-2454900.
            qlcflux = np.array(self.flux_quarters[q])
            qlcflux_err = np.array(self.flux_err_quarters[q])

            # Sigma Clipping
            from astropy.stats import sigma_clip
            import numpy.ma as ma

            filtered_data = sigma_clip(qlcflux, sigma=sigma, maxiters=maxiters)
            qlcflux = np.delete(qlcflux, np.where(filtered_data.mask))
            qlctime = np.delete(qlctime, np.where(filtered_data.mask))
            qlcflux_err = np.delete(qlcflux_err, np.where(filtered_data.mask))

            t.append(qlctime)
            f.append(qlcflux)
            fe.append(qlcflux_err)

        self.time = np.array([y for x in t for y in x])
        self.flux = np.array([y for x in f for y in x])
        self.flux_err = np.array([y for x in fe for y in x])


    def delete_nans(self):
        """Deletes nans from self.time, self.flux, and self.fluxerr"""

        naninds = np.where(np.isnan(self.flux))
        self.time = np.delete(self.time, naninds)
        self.flux = np.delete(self.flux, naninds)
        self.flux_err = np.delete(self.flux_err, naninds)


    def get_simultaneous_transits(self, KOIS, files=None):
        """Gets a list of all transit midpoints for other transits in a system (specified as a list with KOIS)"""

        if self.lightcurvefiles is not None:
            files = self.lightcurvefiles

        self.simult_midpoints = []

        for nkoi in KOIS:

            koi = KOI(nkoi, int(np.floor(float(nkoi))))
            koi.download_planet_params()

            koi.get_stitched_lcs(files)
            koi.get_midpoints()

            self.simult_midpoints.append(koi.midpoints)




    def remove_oot_data(self, nbuffer, linearfit=False, cubicfit=False, custom_data=None, nlinfit=None, include_nans=False, delete_nan_transits=False, nan_limit=10, simultaneous_midpoints=None, simultaneous_threshold=None, return_intransit=True, cadence=0.0201389):
        """Removes out-of-transit segments of Kepler light curves.
        Fits a linear model to out-of-transit points immediately surrounding each transit.
        Subtracts the linear model from each transit cutout.

        Parameters
        ----------
        nbuffer: int
            Number of flux points before and after transit midpoint to include in transit cut-out.
            e.g. if nbuffer = 7, function will preserve 7 flux points before and after each transit midpoint and discard the rest of light curve.
        linearfit: boolean
            True if subtracting a linear fit to baseline. False otherwise
        cubicfit: boolean
            True if subtracting a cubic fit to baseline. False otherwise
        nlinfit: int
            Number of flux points from each end of transit cutout to use in linear fit.
            e.g. if nbuffer = 7 and nlinfit = 5, function will use the 10 outermost flux points for linear fit.
        include_nans: boolean, default False
            Include nans in in-transit data?
        delete_nan_transits: boolean, default False
            Delete entire transit if includes nan flux values > nan_limit?
        nan_limit: int
            Number of nans to allow in-transit before deleting entire transit.
        simultaneous_midpoints: list
            Transit midpoints for other planets in system.
        simultaneous_threshold: float
            Overlap threshold for removing simultaneous transits. If any transit midpoint is closer than [simultaneous_threshold] to a member of [simultaneous_midpoints], that transit is discarded.
        return_intransit: boolean
            Save the in-transit midpoint times?
        cadence: float
            Cadence of the light curve data

        Returns
        -------
        None

        """

        tbjd = []
        tnorm = []
        fl = []
        fr = []
        mpintransit = []

        from tqdm import tqdm
        for i in tqdm(range(len(self.midpoints))):
            try:
                if simultaneous_midpoints is not None:
                    assert simultaneous_threshold is not None, "If removing simultaneous transits, you need to define 'simultaneous_threshold'"
                    near_sim_mpt = find_nearest(simultaneous_midpoints, self.midpoints[i])
                    if abs(self.midpoints[i] - near_sim_mpt) < simultaneous_threshold: # If transit midpoint is close to a simultaneous transit, then don't include that transit.
                        continue

                if linearfit==True:
                    assert nlinfit is not None, "If performing a linear fit, you must define nlinfit."

                    m, b, t1bjd, t1, fnorm, fe1 = do_linfit(self.time, self.flux, self.flux_err, self.midpoints[i], nbuffer, nlinfit, cadence=cadence)
                    if np.isnan(m) and np.isnan(t1) and np.isnan(fnorm):
                        continue

                elif cubicfit==True:
                    assert nlinfit is not None, "If performing a cubic fit, you must define nlinfit."

                    z, t1bjd, t1, fnorm, fe1 = do_cubfit(self.time, self.flux, self.flux_err, self.midpoints[i], nbuffer, nlinfit, custom_data=custom_data, cadence=cadence, midpoint=i)
                    if np.isnan(z).all() and np.isnan(t1) and np.isnan(fnorm):
                        continue

                else:
                    t1bjd, t1, fnorm, fe1 = cutout_no_linfit(self.time, self.flux, self.flux_err, self.midpoints[i], nbuffer)

                    if np.isnan(t1).all():
                        continue

                    if return_intransit==False:
                        midind = int((len(t1bjd) - 1)/2)
                        dur = int(nbuffer-nlinfit)
                        t1bjd = np.delete(t1bjd, [range(midind-dur,midind+dur+1)])
                        t1 = np.delete(t1, [range(midind-dur,midind+dur+1)])
                        fnorm = np.delete(fnorm, [range(midind-dur,midind+dur+1)])
                        fe1 = np.delete(fe1, [range(midind-dur,midind+dur+1)])


                if include_nans==False:

                    if np.count_nonzero(np.isnan(fnorm)) <= nan_limit:
                        tbjd.append(list(t1bjd))
                        tnorm.append(list(t1))
                        fl.append(list(fnorm))
                        fr.append(list(fe1))
                        mpintransit.append(self.midpoints[i])

                    if np.count_nonzero(np.isnan(fnorm)) > nan_limit:

                        if delete_nan_transits == True:
                            #print('Deleted')
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



    def do_tfit_exoplanet(self, tune=1000, draw=1000, save_trace=False, direct='ExampleDir', oversample=29, exptime=0.0201389, optimize=False):
        """Transit light curve fitting with exoplanet.

        Parameters
        ----------
        save_trace: boolean
            Save output trace to directory?
        direct: str
            Output path.
        oversample: int, default 29
            Number of flux points over which to supersample, default 29 min (Kepler long cadence.)
        exptime: float, default 0.0201389
            Time over which to supersample, default 29 min (Kepler long cadence)
        optimize: boolean
            Optimize and start at map_soln?

        Returns
        -------
        trace:
            pymc3 trace object

        """
        import pymc3 as pm
        import pymc3_ext as pmx
        import exoplanet as xo
        import arviz as az
        import aesara_theano_fallback.tensor as tt
        from functools import partial

        u1 = self.ld1
        u2 = self.ld2

        def lds(u1, u2):
            q1 = (u1+u2)**2
            q2 = 0.5*u1*(u1+u2)**(-1)
            return q1, q2

        q1, q2 = lds(u1,u2)
        q1err = np.mean((abs(q1-lds(u1-0.05,u2-0.05)[0]), abs(lds(u1+0.05,u2+0.05)[0]-q1)))
        q2err = np.mean((abs(lds(u1-0.05,u2-0.05)[1]-q2), abs(q2-lds(u1+0.05,u2+0.05)[1])))


        with pm.Model() as model:

            # Shared orbital parameters
            period = pm.Uniform("period", lower=self.period-0.05, upper=self.period+0.05, testval=self.period)
            t0 = pm.Uniform("t0", lower=self.epoch-0.05, upper=self.epoch+0.05, testval=self.epoch)

            b = pm.Uniform("b", lower=-1.0, upper=1.0, testval=self.b)

            q = [pm.Normal("q1", mu=q1, sd=q1err), pm.Normal("q2", mu=q2, sd=q2err)]
            u = [reverse_ld_coeffs("quadratic", q[0], q[1])[0], reverse_ld_coeffs("quadratic", q[0], q[1])[1]]

            rho_star = pm.Normal("rho_star", mu=self.rho_star*1.408, sd=self.rho_star_err*1.408)
            r_star = self.rstar

            ror = pm.Uniform("ror", lower=0.001, upper=0.2, testval=self.rprs)
            r_pl = pm.Deterministic("r_pl", ror * r_star)

            ecs = pmx.UnitDisk("ecs", shape=(2, 1), testval=0.01*np.ones((2, 1)))
            ecc = pm.Deterministic("ecc", tt.sum(ecs**2, axis=0))
            omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))

            # The flux zero point
            mean = 1.0

            # Set up a Keplerian orbit for the planets
            orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b, ror=ror, rho_star=rho_star, r_star=r_star, ecc=ecc, omega=omega)

            light_curves = xo.LimbDarkLightCurve(u[0], u[1]).get_light_curve(orbit=orbit, r=r_pl, t=self.time, texp=exptime, oversample=oversample)
            light_curve = pm.math.sum(light_curves, axis=-1) + mean

            pm.Deterministic("light_curves", light_curves)
            pm.Normal("obs", mu=light_curve, sd=self.flux_err, observed=self.flux)

            map_soln = None

            if optimize==True:
                map_soln = model.test_point
                print('Optimizing...')
                map_soln = pmx.optimize(map_soln)
                print('Done Optimizing')


        with model:
            trace = pmx.sample(
                tune=tune,
                draws=draw,
                cores=1,
                chains=2,
                start=map_soln,
                target_accept=0.90,
                return_inferencedata=True)

        return trace
