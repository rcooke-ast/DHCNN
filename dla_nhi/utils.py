import numpy as np
import astropy.io.fits as fits
from astropy.table import Table


def simulate_random_dla_Lya(rest_window=30.0, proxqso=0.0):
    """
    rest_window = Number of REST Angstroms to the left and right of the central DLA profile to use
    rest_proxqso = number of REST Angstroms to the left of the QSO to use for generating a DLA (only used if non-zero)
    """
    filename = '../data/DR1_quasars_master_NODLA.csv'
    t = Table.read(filename, format='ascii.csv')
    # Trim the table
    t_trim = t[np.where( (t['zem_Adopt'].data>2.6) & (t['zem_Adopt'].data<3.4) )]
    # Select a random QSO
    nqso = t_trim['Name_Adopt'].size
    qidx = np.random.randint(0, nqso)
    qso = t_trim[qidx]
    zem = qso['zem_Adopt']
    wavstr = qso['WavStart']+rest_window
    if (proxqso > 0.0):
        wavstr = max(wavstr, (1+zem)*(1215.6701 - proxqso))
    zdla = np.random.uniform(wavstr, (1+zem)*1215.6701)/1215.6701 - 1
    # Load the data
    dat = fits.open(qso['Name_Adopt']+'.fits')
    wave = dat[1].data['WAVE']
    cont = dat[1].data['CONTINUUM']
    flux = dat[1].data['FLUX']*cont
    flue = dat[1].data['ERR']*cont
    # Generate an N(H I) of a DLA from f(N)
    # The range of interest is:
    # 19.3 < log N(H I) < 21.0
    # 2.6 < z < 3.4
    # Generate a DLA Lya profile
