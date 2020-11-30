
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm

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


# def asymmetric_gaussian(mean, sigma_minus, sigma_plus):
#     """Generates an asymmetric Gaussian distribution based on a mean and 2 different sigmas (one (-) and one (+))
#     Made by generating 2 symmetric Gaussians with different sigmas and sticking them together at the mean.
#     The integral of the resulting Gaussian is 1.
#
#     Parameters
#     ----------
#     mean: float
#         Mean of distribution
#     sigma_minus: float
#         Negative sigma of distribtion
#     sigma_plus: float
#         Positive sigma of distribtion
#
#     Results
#     -------
#     dist: np.ndarray
#         Asymmetric Gaussian distribution, length 1200
#     """
#
#     left = np.random.normal(mean, abs(sigma_minus), 1300)
#     right = np.random.normal(mean, abs(sigma_plus), 1300)
#     dist_left = left[left<mean]
#     dist_right = right[right>=mean]
#     dist = np.concatenate((dist_right, dist_left), axis=None)
#
#     np.random.shuffle(dist)
#
#     while len(dist) > 1200:
#         dist = np.delete(dist, [np.random.randint(0, len(dist)-1)])
#     else:
#         return dist

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

def density(mass, radius, norm_to=None):
    """Get density of sphere given mass and radius.

    Parameters
    ----------
    mass: float
        Mass of sphere (kg)
    radius: float
        Radius of sphere (m)
    norm: float, default None
        Value to normalize to (kg m^-3)
    """

    if norm==None:
        return ((mass*1.989e30)/((4./3.)*np.pi*radius**3*696.34e6**3))
    else:
        return ((mass*1.989e30)/((4./3.)*np.pi*radius**3*696.34e6**3))/float(norm)


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


def find_density_dist_symmetric(ntargs, masses, masserr, radii, raderr):
    """Gets symmetric stellar density distribution for stars.
    Symmetric stellar density distribution = Gaussian with same sigma on each end.

    Parameters
    ----------
    ntargs: int
        Number of stars to get distribution for
    masses: np.ndarray
        Array of stellar masses (solar mass)
    masserr: np.ndarray
        Array of sigma_mass (solar mass)
    radii: np.ndarray
        Array of stellar radii (solar radii)
    raderr: np.ndarray
        Array of sigma_radius (solar radii)

    Returns
    -------
    rho_dist: np.ndarray
        Array of density distributions for each star
        Each element length 1000
    mass_dist: np.ndarray
        Array of symmetric Gaussian mass distributions for each star
        Each element length 1000
    rad_dist: np.ndarray
        Array of symmetric Gaussian radius distributions for each star
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
        mass_temp = np.random.normal(masses[star], masserr[star], 1200)
        #len 1200
        rad_temp = np.random.normal(radii[star], raderr[star], 1200)
        #len 1200
        #####

        #for j from 0 to 1200
        #for each point in individual star PDF
        #Adding each density point to rho_temp (specific to this star)
        for point in range(len(mass_temp)):
            #if mass_dist[point] >= 0. and rad_dist[point] >= 0:
            if True:
                rho_temp[point] = density(mass_temp[point], rad_temp[point], sol_density())

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
