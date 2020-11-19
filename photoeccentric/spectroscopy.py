import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
import glob


def fit_isochrone(data, isochrones):
    """Pulls isochrones where effective temperature, mass, and radius fall within 1-sigma errorbars from an observed star.

       Parameters
       ----------
       data: pd.DataFrame
            Spectroscopic data + Kepler/Gaia data for n stars in one table. (`muirhead_comb`)
       isochrones: pd.DataFrame
            Table of isochrones (models). (`isochrones`)

       Returns
       -------
       iso_fits_final: list of pd.DataFrame
            Each element of list returend is a table of the isochrones that fit this star (index matches) based ONLY on spectroscopy.
       """

    iso_fits_final = list()

    #test each star in spectroscopy sample:
    for i in tqdm(range(len(muirhead_comb))):

        iso_fits = pd.DataFrame()

        Teff_range = [data.Teff[i]-data.eTeff[i], data.Teff[i]+data.ETeff[i]]
        Mstar_range = [data.Mstar[i]-data.e_Mstar[i], data.Mstar[i]+data.e_Mstar[i]]
        Rstar_range = [data.Rstar[i]-data.e_Rstar[i], data.Rstar[i]+data.e_Rstar[i]]

        #test each stellar model to see if it falls within error bars:
        for j in range(len(isochrones)):
            if Teff_range[0] < 10**isochrones.logt[j] < Teff_range[1] and Mstar_range[0] < isochrones.mstar[j] < Mstar_range[1] and Rstar_range[0] < isochrones.radius[j] < Rstar_range[1]:
                iso_fits = iso_fits.append(isochrones.loc[[j]])

        iso_fits['KIC'] = muirhead_comb['KIC'][i]
        iso_fits['KOI'] = muirhead_comb['KOI'][i]

        iso_fits_final.append(iso_fits)

    return iso_fits_final


def fit_isochrone_lum(data, isochrones):
    """Pulls isochrones where effective temperature, mass, radius, and luminosity fall within 1-sigma errorbars from an observed star.

       Parameters
       ----------
       data: pd.DataFrame
            Spectroscopic data + Kepler/Gaia data for n stars in one table. (`muirhead_comb`)
       isochrones: pd.DataFrame
            Table of isochrones (models). (`isochrones`)

       Returns
       -------
       iso_fits_final: list of pd.DataFrame
            Each element of list returend is a table of the isochrones that fit this star (index matches) based on spectroscopy AND Gaia luminosity.
       """

    iso_fits_final = list()

    #for i in tqdm(range(len(muirhead_comb))):
    for i in range(1):

        iso_fits = pd.DataFrame()

        Teff_range = [data.Teff[i]-data.eTeff[i], data.Teff[i]+data.ETeff[i]]
        Mstar_range = [data.Mstar[i]-data.e_Mstar[i], data.Mstar[i]+data.e_Mstar[i]]
        Rstar_range = [data.Rstar[i]-data.e_Rstar[i], data.Rstar[i]+data.e_Rstar[i]]
        lum_range = [data.lum_val[i]-data.lum_percentile_lower[i], data.lum_val[i]+data.lum_percentile_lower[i]]

        for j in range(len(isochrones)):
            if Teff_range[0] < 10**isochrones.logt[j] < Teff_range[1] and Mstar_range[0] < isochrones.mstar[j] < Mstar_range[1] and Rstar_range[0] < isochrones.radius[j] < Rstar_range[1] and lum_range[0] < 10**isochrones.logl_ls[j] < lum_range[1]:
                iso_fits = iso_fits.append(isochrones.loc[[j]])

        iso_fits['KIC'] = muirhead_comb['KIC'][i]
        iso_fits['KOI'] = muirhead_comb['KOI'][i]

        iso_fits_final.append(iso_fits)

    return iso_fits_final



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
        return ((mass*1.989e30)/((4./3.)*np.pi*radius**3*696.34e6**3))
    else:
        return ((mass*1.989e30)/((4./3.)*np.pi*radius**3*696.34e6**3))/float(norm)

def iso_lists(path):
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

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(np.where(array == array[idx])[0])

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
