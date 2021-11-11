import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .utils import *

def get_kepID(hdul):
    """Pulls KIC IDs from Kepler-Gaia dataset.

    Parameters
    ----------
    hdul: astropy.io.fits.hdu.hdulist.HDUList
        Astropy hdulist from FITS file of Kepler-Gaia dataset

    Returns
    -------
    kepID_lst: np.array
        Array of KIC IDs for entire Kepler-Gaia dataset
    """
    kepID_lst = []
    for i in tqdm(range(len(hdul[1].data))):
        kepID_lst.append(hdul[1].data[i][96])
    kepID_lst = np.asarray(kepID_lst)

    return np.asarray(kepID_lst)



def get_masses(hdul):
    """Pulls stellar masses from Kepler-Gaia dataset.

    Parameters
    ----------
    hdul: astropy.io.fits.hdu.hdulist.HDUList
        Astropy hdulist from FITS file of Kepler-Gaia dataset

    Returns
    -------
    mass_lst: np.array
        Array of masses for entire Kepler-Gaia dataset
    masserr1_lst: np.array
        Array of mass (-) errors for entire Kepler-Gaia dataset
    masserr2_lst: np.array
        Array of mass (+) errors for entire Kepler-Gaia dataset
    """
    mass_lst = []
    masserr1_lst = []
    masserr2_lst = []

    for i in tqdm(range(len(hdul[1].data))):
        mass_lst.append(hdul[1].data[i][122])
        masserr1_lst.append(hdul[1].data[i][123])
        masserr2_lst.append(hdul[1].data[i][124])
    mass_lst = np.asarray(mass_lst)
    masserr1_lst = np.asarray(masserr1_lst)
    masserr2_lst = np.asarray(masserr2_lst)

    return mass_lst, masserr1_lst, masserr2_lst


def get_radii(hdul, mission='Kepler'):
    """Pulls stellar radii from Kepler-Gaia dataset.

    Parameters
    ----------
    hdul: astropy.io.fits.hdu.hdulist.HDUList
        Astropy hdulist from FITS file of Kepler-Gaia dataset
    mission: 'Kepler' or 'Gaia'
        Mission from which radii are measured

    Returns
    -------
    rad_lst: np.array
        Array of radii for entire Kepler-Gaia dataset
    raderr1_lst: np.array
        Array of radius (-) errors for entire Kepler-Gaia dataset
    raderr2_lst: np.array
        Array of radius (+) errors for entire Kepler-Gaia dataset
    """

    rad_lst = []
    raderr1_lst = []
    raderr2_lst = []

    for i in tqdm(range(len(hdul[1].data))):

        if mission=='Kepler':

            #Kepler radii
            rad_lst.append(hdul[1].data[i][119])
            raderr1_lst.append(hdul[1].data[i][120])
            raderr2_lst.append(hdul[1].data[i][121])

        elif mission=='Gaia':

            #Gaia radii
            rad_lst.append(hdul[1].data[i][88])
            raderr1_lst.append(hdul[1].data[i][89])
            raderr2_lst.append(hdul[1].data[i][90])

        else:
            raise KeyError("Invalid mission")

    rad_lst = np.asarray(rad_lst)
    raderr1_lst = np.asarray(raderr1_lst)
    raderr2_lst = np.asarray(raderr2_lst)

    return rad_lst, raderr1_lst, raderr2_lst


def get_logg(hdul):
    """Pulls stellar log(g) from Kepler-Gaia dataset.

    Parameters
    ----------
    hdul: astropy.io.fits.hdu.hdulist.HDUList
        Astropy hdulist from FITS file of Kepler-Gaia dataset

    Returns
    -------
    rad_lst: np.array
        Array of radii for entire Kepler-Gaia dataset
    raderr1_lst: np.array
        Array of radius (-) errors for entire Kepler-Gaia dataset
    raderr2_lst: np.array
        Array of radius (+) errors for entire Kepler-Gaia dataset
    """
    logg_lst = []
    loggerr1_lst = []
    loggerr2_lst = []

    for i in tqdm(range(len(hdul[1].data))):
        logg_lst.append(hdul[1].data[i][111])
        loggerr1_lst.append(hdul[1].data[i][112])
        loggerr2_lst.append(hdul[1].data[i][113])

    logg_lst = np.asarray(logg_lst)
    loggerr1_lst = np.asarray(loggerr1_lst)
    loggerr2_lst = np.asarray(loggerr2_lst)

    return logg_lst, loggerr1_lst, loggerr2_lst


def get_nan_indices(masses, radii, radii2, logg):
    """Removes stars with nans in any of stellar masses, radii, and log(g).
    If one or more entry is nan, the whole star is discarded.

    Parameters
    ----------
    masses: np.ndarray
        Masses of all stars in dataset
    radii: np.ndarray
        Radii of all stars in dataset from Kepler
    radii2: np.ndarray
        Radii of all stars in dataset from Gaia
    logg: np.ndarray
        log(g)s of all stars in dataset

    Returns
    -------
    nan_i = list
        Indices of stars with nan values
    """

    #Indices of stars with nan values (to be tossed)
    nan_i = []

    for i in range(len(masses)):
        if np.isnan(masses[i]) or np.isnan(radii[i]) or np.isnan(radii2[i]):
            nan_i.append(i)

    return nan_i


def remove_nans(nan_indices, stellar_prop):
    """Removes nans from array of stellar property.

    Notes: nan_indices is the output from get_nan_indices()
    Run remove_nans() on every array input in get_nan_indices().

    Parameters
    ----------
    nan_indices: list
        Indices of nan values to be removed
    stellar_prop: np.ndarray
        Array of stellar property (masses, radii, etc) to remove nans from

    Returns
    -------
    stellar_prop_nonans: np.ndarray
        Array of stellar property without nans.

    """
    stellar_prop_nonans = np.delete(stellar_prop, nan_i)
    return stellar_prop_nonans

def density(mass, radius):
    """Get density of sphere given mass and radius.

    Parameters
    ----------
    mass: float
        Mass of sphere (kg)
    radius: float
        Radius of sphere (m)

    Returns
    rho: float
        Density of sphere (kg*m^-3)
    """

    rho = mass/((4.0/3.0)*np.pi*radius**3)
    return rho


def asymmetric_gaussian(mean, sigma_minus, sigma_plus, nvals):
    """Generates an asymmetric Gaussian distribution based on a mean and 2 different sigmas (one (-) and one (+))
    Made by generating 2 symmetric Gaussians with different sigmas and sticking them together at the mean.
    The integral of the resulting Gaussian is 1.

    Parameters
    ----------
    mean: float
        Mean of distribution
    sigma_minus: float
        Negative sigma of distribtion
    sigma_plus: float
        Positive sigma of distribtion
    nvals: int
        Number of values

    Results
    -------
    dist: np.ndarray
        Asymmetric Gaussian distribution, length nvals
    """

    left = np.random.normal(mean, abs(sigma_minus), nvals+200)
    right = np.random.normal(mean, abs(sigma_plus), nvals+200)
    dist_left = left[left<mean]
    dist_right = right[right>=mean]
    dist = np.concatenate((dist_right, dist_left), axis=None)

    np.random.shuffle(dist)

    while len(dist) > nvals:
        dist = np.delete(dist, [np.random.randint(0, len(dist)-1)])
    else:
        return dist

def solar_density():
    """Gets solar density in kg m^-3.

    Parameters
    ----------
    None

    Returns
    -------
    sol_density: float
        Solar density in kg m^-3
    """
    sol_density = ((1.*1.989e30)/((4./3.)*np.pi*1.**3*696.34e6**3))
    return sol_density


def read_stellar_params(isodf):
    mstar = isodf["mstar"].mean()
    mstar_err = isodf["mstar"].std()
    rstar = isodf["radius"].mean()
    rstar_err = isodf["radius"].std()

    return mstar, mstar_err, rstar, rstar_err



def get_rho_star(mstar, mstar_err, rstar, rstar_err, arrlen=1000):

    density, mass, radius = find_density_dist_symmetric(mstar, mstar_err, rstar, rstar_err, npoints=arrlen)

    mean_density = mode(density)

    return density, mass, radius, mean_density



def find_density_dist_symmetric(mass, masserr, radius, raderr, npoints):
    """Gets symmetric stellar density distribution for stars.
    Symmetric stellar density distribution = Gaussian with same sigma on each end.

    Parameters
    ----------
    mass: float
        Mean stellar mass (solar mass)
    masserr: np.ndarray
        Sigma of mass (solar mass)
    radius: float
        Mean stellar radius (solar radii)
    raderr: np.ndarray
        Sigma of radius (solar radii)
    npoints: int

    Returns
    -------
    rho_dist: np.ndarray
        Array of density distributions for each star in kg/m^3
        Length npoints
    mass_dist: np.ndarray
        Array of symmetric Gaussian mass distributions for each star in kg
        Length npoints
    rad_dist: np.ndarray
        Array of symmetric Gaussian radius distributions for each star in m
        Length 100npoints0
    """

    smass_kg = 1.9885e30  # Solar mass (kg)
    srad_m = 696.34e6     # Solar radius (m)

    rho_dist = np.zeros(npoints)

    mass_dist = np.random.normal(mass*smass_kg, masserr*smass_kg, npoints)
    rad_dist = np.random.normal(radius*srad_m, raderr*srad_m, npoints)


    #Add each density point to rho_temp (for each star)
    for point in range(len(mass_dist)):
        rho_dist[point] = density(mass_dist[point], rad_dist[point])


    return rho_dist, mass_dist, rad_dist



def find_density_dist_asymmetric(ntargs, masses, masserr1, masserr2, radii, raderr1, raderr2, logg):
    """Gets asymmetric stellar density distribution for stars, based on asymmetric mass and radius errorbars.
    Asymmetric stellar density distribution = Gaussian with different sigmas on each end.

    Parameters
    ----------
    ntargs: int
        Number of stars to get distribution for
    masses: np.ndarray
        Array of stellar masses (solar mass)
    masserr1: np.ndarray
        Array of (-) sigma_mass (solar mass)
    masserr2: np.ndarray
        Array of (+) sigma_mass (solar mass)
    radii: np.ndarray
        Array of stellar radii (solar radii)
    raderr1: np.ndarray
        Array of (-) sigma_radius (solar radii)
    raderr2: np.ndarray
        Array of (+) sigma_radius (solar radii)
    logg: np.ndarray
        Array of log(g)s

    Returns
    -------
    rho_dist: np.ndarray
        Array of density distributions for each star
        Each element length 1000
    mass_dist: np.ndarray
        Array of asymmetric Gaussian mass distributions for each star
        Each element length 1000
    rad_dist: np.ndarray
        Array of asymmetric Gaussian radius distributions for each star
        Each element length 1000
    """

    rho_dist = np.zeros((ntargs, 1000))
    mass_dist = np.zeros((ntargs, 1000))
    rad_dist = np.zeros((ntargs, 1000))

    #star: indexing star
    #point: indexing PDF point for star
    for star in tqdm(range(ntargs)):

        rho_temp = np.zeros(1200)
        mass_temp = np.zeros(1200)
        rad_temp = np.zeros(1200)

        #####
        mass_temp = asymmetric_gaussian(masses[star], masserr2[star], masserr1[star])
        #len 1200
        rad_temp = asymmetric_gaussian(radii[star], raderr2[star], raderr1[star])
        #len 1200
        #####

        #for j from 0 to 1200
        #for each point in individual star PDF
        #Adding each density point to rho_temp (specific to this star)
        for point in range(len(mass_temp)-1):
            #if (mass_temp[point] >= 0. and rad_temp[point] >= 0):
            if True:
                rho_temp[point] = density(mass_temp[point], rad_temp[point], sol_density)

        #Now rho_temp is a n-long array with this star. We want it to be 1000-long exactly

        while len(rho_temp) > 1000:
            temp_ind = np.random.randint(0, len(rho_temp)-1)
            rho_temp = np.delete(rho_temp, temp_ind)
            mass_temp = np.delete(mass_temp, temp_ind)
            rad_temp = np.delete(rad_temp, temp_ind)
        else:
            rho_dist[star] = rho_temp
            mass_dist[star] = mass_temp
            rad_dist[star] = rad_temp


    return rho_dist, mass_dist, rad_dist




def fit_isochrone_lum(data, isochrones, gaia_lum=True, source='Muirhead', lum_source='Gaia', lums=None):
    """Pulls isochrones where effective temperature, mass, radius, and luminosity fall within 1-sigma errorbars from an observed star.

   Parameters
   ----------
   data: pandas.DataFrame
       Spectroscopic data + Kepler/Gaia for n stars in one table. (muirhead_comb or muirhead_lamost)
   isochrones: pandas.DataFrame
       Isochrones table. (isochrones)
   source: 'Muirhead' or 'LAMOST' (default 'Muirhead')
       Source for Teffs

   Returns
   -------
   iso_fits_final: pandas.DataFrame()
       All isochrones that are consistent with this star based on spectroscopy and Gaia luminosity.
   """

    from tqdm import tqdm

    iso_fits = pd.DataFrame()

    wide = 1

    if source=='Muirhead':
        Teff_range = [float(data.Teff)-wide*float(data.eTeff), float(data.Teff)+wide*float(data.ETeff)]

    elif source=='LAMOST':
        Teff_range = [float(data.TEFF_AP)-float(data.TEFF_AP_ERR), float(data.TEFF_AP)+float(data.TEFF_AP_ERR)]

    # Muirhead
    Mstar_range = [float(data.Mstar)-wide*float(data.e_Mstar), float(data.Mstar)+wide*float(data.e_Mstar)]
    # Muirhead
    Rstar_range = [float(data.Rstar)-wide*float(data.e_Rstar), float(data.Rstar)+wide*float(data.e_Rstar)]

    # Gaia
    lum_range = [float(data.lum_percentile_lower), float(data.lum_percentile_upper)]

    if lum_source=='custom':
        lum_range=lums

    print(Teff_range)
    print(Rstar_range)
    print(lum_range)

    if np.isnan(lum_range[0]) or np.isnan(lum_range[1]):
        #print(lum_range)
        print("No Gaia Lums")

        for j in tqdm(range(len(isochrones))):
            if Teff_range[0] < 10**isochrones.logt[j] < Teff_range[1] and Mstar_range[0] < isochrones.mstar[j] < Mstar_range[1] and Rstar_range[0] < isochrones.radius[j] < Rstar_range[1]:
                iso_fits = iso_fits.append(isochrones.loc[[j]])

    else:
        print("Gaia Lums")

        templums = []

        if gaia_lum==True:
            for j in tqdm(range(len(isochrones))):
                #print(10**isochrones.logl_ls[j])
                if Teff_range[0] < 10**isochrones.logt[j] < Teff_range[1] and Mstar_range[0] < isochrones.mstar[j] < Mstar_range[1] and Rstar_range[0] < isochrones.radius[j] < Rstar_range[1] and lum_range[0] < 10**isochrones.logl_ls[j] < lum_range[1]:
                    iso_fits = iso_fits.append(isochrones.loc[[j]])
            print(len(iso_fits))

        if gaia_lum==False:
            for j in tqdm(range(len(isochrones))):
                if Teff_range[0] < 10**isochrones.logt[j] < Teff_range[1] and Mstar_range[0] < isochrones.mstar[j] < Mstar_range[1] and Rstar_range[0] < isochrones.radius[j] < Rstar_range[1]:
                    iso_fits = iso_fits.append(isochrones.loc[[j]])
                    templums.append(10**isochrones.logl_ls[j])
            print(len(iso_fits))
            print(np.nanmin(templums), np.nanmax(templums))

    #iso_fits['KIC'] = stellarobs['KIC']
    #iso_fits['KOI'] = stellarobs['KOI']

    return iso_fits

def solar_density():
    #Define Solar density for 1 Msol and 1 Rsol in kg/m3
    sol_density = ((1.*1.989e30)/((4./3.)*np.pi*1.**3*696.34e6**3))
    return sol_density

def find_density_dist(mass_dist, rad_dist, norm=None):

    rho_dist = np.zeros(len(mass_dist))
    #for point from 0 to len(isochrones)
    #Adding each density point to rho_dist
    for point in range(len(mass_dist)):
        rho_dist[point] = density(mass_dist[point], rad_dist[point], norm=norm)

    return rho_dist

def density(mass, radius, norm=None):
    """Mass in solar density
    Radius in solar density
    sol_density in kg/m^3"""

    if norm==None:
        return ((mass)/((4./3.)*np.pi*radius**3))
    else:
        return ((mass)/((4./3.)*np.pi*radius**3))/float(norm)

def iso_lists(path):

    from glob import glob

    iso_lst = []
    files = glob.glob(path)
    for f in files:
        iso_lst.append(pd.read_csv(f))
    return iso_lst


def dict_rhos(isos):
    dict = {}
    for i in range(len(isos)):
        kic = isos[i].KIC[0]
        dict["{0}".format(kic)] = find_density_dist(np.array(isos[i].mstar), np.array(isos[i].radius))
    return dict


def rho_dict_to_csv(rho_dict, filename):
    pd.DataFrame.from_dict(rho_dict, orient='index').transpose().to_csv(filename, index=False)


def get_cdf(dist, nbins=100):
    counts, bin_edges = np.histogram(dist, bins=nbins, range=(np.min(dist), np.max(dist)))
    cdf = np.cumsum(counts)
    cdf = cdf/np.max(cdf)
    return bin_edges[1:], cdf

# def find_nearest_index(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return int(np.where(array == array[idx])[0])

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if len(np.where(array == array[idx])[0]) == 1:
        return int(np.where(array == array[idx])[0])
    else:
        return int(np.where(array == array[idx])[0][0])

def find_sigma(x, cdf, sign):
    med = x[find_nearest_index(cdf, 0.5)]
    if sign == "-":
        sigma = x[find_nearest_index(cdf, 0.16)] - med
    elif sign == "+":
        sigma = x[find_nearest_index(cdf, 0.84)] - med
    return sigma

def plot_cdf(x, cdf):
    plt.plot(x, cdf)
    plt.axvline(x=x[find_nearest_index(cdf, 0.5)], c='r', label='Median')
    plt.axvline(x=x[find_nearest_index(cdf, 0.5)]+find_sigma(x, cdf, "-"), c='blue', label='- sigma')
    plt.axvline(x=x[find_nearest_index(cdf, 0.5)]+find_sigma(x, cdf, "+"), c='orange', label='+ sigma')
    plt.legend()
    plt.xlabel('Density')


def iterate_stars(rho_dict):
    for key, val in rho_dict.items():
        x, cdf = get_cdf(rho_dict[key])
        sigma_minus = find_sigma(x, cdf, "-")
        sigma_plus = find_sigma(x, cdf, "+")

        df = pd.DataFrame({"x" : x, "cdf" : cdf, "sigma_minus": sigma_minus, "sigma_plus": sigma_plus})
        df.to_csv("cdfs/cdf_lum/cdf_lum_" + str(key) + ".csv", index=False)
