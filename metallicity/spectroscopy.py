import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm


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
