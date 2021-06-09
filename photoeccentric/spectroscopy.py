import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
import glob


def fit_isochrone_lum(data, stellarobs, isochrones, gaia_lum=True, source='Muirhead'):
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

    iso_fits = pd.DataFrame()

    if source=='Muirhead':
        Teff_range = [float(data.Teff)-float(data.eTeff), float(data.Teff)+float(data.ETeff)]
    elif source=='LAMOST':
        Teff_range = [float(data.TEFF_AP)-float(data.TEFF_AP_ERR), float(data.TEFF_AP)+float(data.TEFF_AP_ERR)]

    Mstar_range = [float(data.Mstar)-float(data.e_Mstar), float(data.Mstar)+float(data.e_Mstar)]
    Rstar_range = [float(data.Rstar)-float(data.e_Rstar), float(data.Rstar)+float(data.e_Rstar)]
    lum_range = [float(data.lum_val)-float(data.lum_percentile_lower), float(data.lum_val)+float(data.lum_percentile_lower)]

    for j in tqdm(range(len(isochrones))):
        if gaia_lum==True:
            if Teff_range[0] < 10**isochrones.logt[j] < Teff_range[1] and Mstar_range[0] < isochrones.mstar[j] < Mstar_range[1] and Rstar_range[0] < isochrones.radius[j] < Rstar_range[1] and lum_range[0] < 10**isochrones.logl_ls[j] < lum_range[1]:
                iso_fits = iso_fits.append(isochrones.loc[[j]])

        if gaia_lum==False:
            if Teff_range[0] < 10**isochrones.logt[j] < Teff_range[1] and Mstar_range[0] < isochrones.mstar[j] < Mstar_range[1] and Rstar_range[0] < isochrones.radius[j] < Rstar_range[1]:
                iso_fits = iso_fits.append(isochrones.loc[[j]])

    iso_fits['KIC'] = stellarobs['KIC']
    iso_fits['KOI'] = stellarobs['KOI']

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
