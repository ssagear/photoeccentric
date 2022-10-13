import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def from_exoarchive(KOI=0, KIC=0, KepName=0):
    """Downloads data from exoplanet archive"""
    
    from urllib.request import urlopen
    from io import StringIO
    
    if KOI==0 and KIC==0 and KepName==0:
        print('You need to specify either KOI or KIC or KepName.')
        return

    if (KOI!=0 and KIC!=0) or (KOI!=0 and KepName!=0) or (KepName!=0 and KIC!=0):
        print('You can only specify one of KOI, KIC or KepName.')
        return
    
    if KOI!=0:
        if 'K' in KOI:
            nkoi = KOI
        elif 'K' not in KOI:
            if len(str(KOI)) == 4:
                nkoi = 'K0000' + str(KOI)
            if len(str(KOI)) == 5:
                nkoi = 'K000' + str(KOI)
            elif len(str(KOI)) == 6:
                nkoi = 'K00' + str(KOI)
            elif len(str(KOI)) == 7:
                nkoi = 'K0' + str(KOI)

        url_string = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&where=kepoi_name%20like%20%27" + nkoi + '%27&select=*'

        with urlopen(url_string) as webpage:
            content = webpage.read().decode()

        content = StringIO(content)
        content = pd.read_csv(content, sep=",")

    elif KIC!=0:

        nkic = str(KIC)

        url_string = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&where=kepid=" + nkic + '&select=*'

        with urlopen(url_string) as webpage:
            content = webpage.read().decode()

        content = StringIO(content)
        content = pd.read_csv(content, sep=",")

    if KepName!=0:

        nkepname = KepName.replace(' ', '%20')

        url_string = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&where=kepler_name%20like%20%27" + nkepname + '%27&select=*'

        with urlopen(url_string) as webpage:
            content = webpage.read().decode()

        content = StringIO(content)
        content = pd.read_csv(content, sep=",")

    return url_string, content



def mast_query(request):
    """ From https://mast.stsci.edu/api/v0/MastApiTutorial.html
    
    Perform a MAST query for Kepler light curves."""

    import sys
    import json
    import requests
    from urllib.parse import quote as urlencode
    
    # Base API url
    request_url='https://mast.stsci.edu/api/v0/invoke'    
    
    # Grab Python Version 
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "User-agent":"python-requests/"+version}

    # Encoding the request as a json string
    req_string = json.dumps(request)
    req_string = urlencode(req_string)
    
    # Perform the HTTP request
    resp = requests.post(request_url, data="request="+req_string, headers=headers)
    
    # Pull out the headers and response content
    head = resp.headers
    content = resp.content.decode('utf-8')

    return head, content



def kepler_from_mast(object_of_interest):

    """Perform a MAST query for Kepler light curves."""

    import sys
    import json
    import requests
    from urllib.parse import quote as urlencode
    from astropy.table import Table
    import os


    resolver_request = {'service':'Mast.Name.Lookup',
                        'params':{'input':object_of_interest,
                                'format':'json'}}

    headers, resolved_object_string = mast_query(resolver_request)
    resolved_object = json.loads(resolved_object_string)

    obj_ra = resolved_object['resolvedCoordinate'][0]['ra']
    obj_dec = resolved_object['resolvedCoordinate'][0]['decl']

    # MAST cone search
    mast_request = {'service':'Mast.Caom.Cone',
                'params':{'ra':obj_ra,
                          'dec':obj_dec,
                          'radius':0.2},
                'format':'json',
                'pagesize':2000,
                'page':1,
                'removenullcolumns':True,
                'removecache':True}

    headers, mast_data_str = mast_query(mast_request)
    mast_data = json.loads(mast_data_str)

    mast_data_table = Table()
    for col,atype in [(x['name'],x['type']) for x in mast_data['fields']]:
        if atype=="string":
            atype="str"
        if atype=="boolean":
            atype="bool"
        mast_data_table[col] = np.array([x.get(col,None) for x in mast_data['data']],dtype=atype)

    kepler_lightcurves = mast_data_table[(mast_data_table['obs_collection'] == 'Kepler') & (mast_data_table['dataproduct_type'] == 'timeseries')][0]
    obsid = kepler_lightcurves['obsid']

    product_request = {'service':'Mast.Caom.Products',
                    'params':{'obsid':obsid},
                    'format':'json',
                    'pagesize':100,
                    'page':1}   

    headers, obs_products_string = mast_query(product_request)
    obs_products = json.loads(obs_products_string)

    sci_prod_arr = [x for x in obs_products['data'] if x.get("productType", None) == 'SCIENCE']
    science_products = Table()

    for col, atype in [(x['name'], x['type']) for x in obs_products['fields']]:
        if atype=="string":
            atype="str"
        if atype=="boolean":
            atype="bool"
        if atype == "int":
            atype = "float" # array may contain nan values, and they do not exist in numpy integer arrays
        science_products[col] = np.array([x.get(col,None) for x in sci_prod_arr],dtype=atype)

    url_list = [("uri", url) for url in science_products['dataURI'][:2]]
    extension = ".tar.gz"

    download_url = 'https://mast.stsci.edu/api/v0.1/Download/bundle'
    resp = requests.post(download_url + extension, data=url_list)

    out_file = "mastDownload" + extension
    with open(out_file, 'wb') as FLE:
        FLE.write(resp.content)
        
    # check for file 
    if not os.path.isfile(out_file):
        print("ERROR: " + out_file + " failed to download.")
    else:
        print("COMPLETE: ", out_file)


def get_timeseries_files(gzfile):
    """path: a tar.gz file downloaded from MAST"""

    import tarfile
    import os

    tar = tarfile.open(gzfile, "r:gz")
    tar.extractall('MAST')
    tar.close()
    
    for root, dirs, files in os.walk("./MAST", topdown=False):
        for name in files:
            if 'MAST' in root:
                if os.path.join(root, name).endswith("tar"):
                    tar = tarfile.open(os.path.join(root, name), "r:")
                    tar.extractall('./MAST/lightcurves')
                    tar.close()

    fits_files = []

    for root, dirs, files in os.walk("./MAST/lightcurves/", topdown=False):
        for name in files:
            if name.endswith('.fits'):
                fits_files.append(os.path.join(root, name))
    
    fits_files = np.array(fits_files)
    return fits_files



def get_lc_files(KIC, KICs, lcpath):
    """Gets a list of light curves from a directory."""

    import os

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

def get_mid(time):
    """Returns approximately 1/2 of cadence time."""

    return (time[1]-time[0])/2.

def find_nearest(array, value):
    """Gets the nearest element of array to a value."""
    array = np.asarray(array)
    idx = np.nanargmin((np.abs(array - value)))
    return array[idx]


def find_nearest_index(array, value):
    """Gets the index of the nearest element of array to a value."""
    array = np.asarray(array)
    idx = np.nanargmin((np.abs(array - value)))
    return idx

def get_sigma_individual(SNR, N, Ntransits, tdepth):
    """Gets size of individual error bar for a certain light curve signal to noise ratio.

    Parameters
    ----------
    SNR: float
        Target light curve signal to noise ratio
    N: int
        Number of in-transit flux points for each transit
    Ntransits: int
        Number of transits in light light curve
    tdepth: float
        Transit depth (Rp/Rs)^2

    Returns
    -------
    sigma_individual: float
        Size of individual flux error bar
    """
    sigma_full = np.sqrt(Ntransits)*(tdepth/SNR)
    sigma_individual = sigma_full*np.sqrt(N)
    return sigma_individual


def get_N_intransit(tdur, cadence):
    """Estimates number of in-transit points for transits in a light curve.

    Parameters
    ----------
    tdur: float
        Full transit duration
    cadence: float
        Cadence/integration time for light curve

    Returns
    -------
    n_intransit: int
        Number of flux points in each transit
    """
    n_intransit = tdur/cadence
    return n_intransit

def mode(dist, window=5, polyorder=2, bin_type='int', bins=25):
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

    from scipy.signal import savgol_filter

    if bin_type == 'int':
        n, rbins = np.histogram(dist, bins=bins)
    elif bin_type == 'arr':
        n, rbins = np.histogram(dist, bins=bins)

    bin_centers = np.array([np.mean((rbins[i], rbins[i+1])) for i in range(len(rbins)-1)])
    smooth = savgol_filter(n, window, polyorder)
    mode = bin_centers[np.argmax(n)]
    return mode


def get_sigmas(dist):
    """Gets + and - sigmas from a distribution (gaussian or not). Ignores nan values.

    Parameters
    ----------
    dist: np.array
        Distribution from which sigmas are needed

    Returns
    -------
    sigma_minus: float
        - sigma
    sigma_plus: float
        + sigma
    """

    sigma_minus = np.nanpercentile(dist, 50)-np.nanpercentile(dist, 16)
    sigma_plus = np.nanpercentile(dist, 84)-np.nanpercentile(dist, 50)

    return sigma_minus, sigma_plus


def get_e_from_def(g, w):
    """Gets eccentricity from definition (eqn 4)

    Parameters
    ----------
    g: float
        g value
    w: float
        Omega (angle periapse/apoapse)

    Returns
    -------
    e: float
        Eccentricity calculated solely on g and w

    """
    num = np.sqrt(2)*np.sqrt(2*g**4-g**2*np.cos(2*w)-g**2-2*np.sin(w))
    den = 2*(g**2+np.sin(w)**2)
    e = num/den
    return e

def calc_a_from_rho(period, rho_star):
    """Calculate semimajor axis in stellar radii (a/Rs) from orbital period (days) and average stellar density (SI units)."""

    import scipy.constants as c
    a_rs = (((period*86400.0)**2)*((c.G*rho_star)/(3*c.pi)))**(1./3.)
    return a_rs


def get_cdf(dist, nbins=250):
    """Gets a CDF of a distribution."""

    counts, bin_edges = np.histogram(dist, bins=nbins, range=(np.min(dist), np.max(dist)))
    cdf = np.cumsum(counts)
    cdf = cdf/np.max(cdf)
    return bin_edges[1:], cdf


def get_cdf_val(cdfx, cdfy, val):
    cdfval = cdfy[find_nearest_index(cdfx, val)]
    return cdfval

def get_ppf_val(cdfx, cdfy, val):
    cdfval = cdfx[find_nearest_index(cdfy, val)]
    return cdfval


def calc_r(a_rs, e, w):
    """Calculate r (the planet-star distance) at any point during an eccentric orbit.
    Equation 20 in Murray & Correia Text

    Parameters
    ----------
    a_rs: float
        Semi-major axis (Stellar radius)
    e: float
        Eccentricity
    w: float
        Longitude of periastron (degrees)

    Returns
    -------
    r_rs: float
        Planet-star distance (Stellar radius)
    """

    wrad = w*(np.pi/180.)
    r_rs = (a_rs*(1-e**2))/(1+e*np.cos(wrad-(np.pi/2.)))
    return r_rs


def reverse_ld_coeffs(ld_law, q1, q2):
    """This function adapted from the juliet package at https://github.com/nespinoza/juliet/blob/master/juliet/utils.py"""
    
    if ld_law == 'quadratic':
        coeff1 = 2. * np.sqrt(q1) * q2
        coeff2 = np.sqrt(q1) * (1. - 2. * q2)
    elif ld_law == 'squareroot':
        coeff1 = np.sqrt(q1) * (1. - 2. * q2)
        coeff2 = 2. * np.sqrt(q1) * q2
    elif ld_law == 'logarithmic':
        coeff1 = 1. - np.sqrt(q1) * q2
        coeff2 = 1. - np.sqrt(q1)
    elif ld_law == 'linear':
        return q1, q2
    return coeff1, coeff2