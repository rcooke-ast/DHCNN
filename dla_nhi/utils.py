import numpy as np
import astropy.io.fits as fits
from astropy.table import Table


def simulate_random_dla(rest_window=30.0):
    """
    rest_window = Number of Angstroms to the left and right of the central DLA profile to use
    """
    filename = '../data/DR1_quasars_master_NODLA.csv'
    t = Table.read(filename, format='ascii.csv')
    nqso = t['Name_Adopt'].size
    qidx = np.random.randint(0, nqso)
    qso = t[qidx]
    zem = qso['zem_Adopt']
    wavstr = qso['WavStart']+rest_window
    zdla = np.random.uniform(wavstr, (1+zem)*1215.6701)/1215.6701 - 1
    # Load the data
    dat = fits.open(qso['Name_Adopt']+'.fits')
    wave = dat[1].data['WAVE']
    cont = dat[1].data['CONTINUUM']
    flux = dat[1].data['FLUX']*cont
    flue = dat[1].data['ERR']*cont
    # Generate an N(H I) of a DLA from f(N)

