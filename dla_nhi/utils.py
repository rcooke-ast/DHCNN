import numpy as np
import astropy.io.fits as fits
from astropy.table import Table


def get_binsize(wave, bintype="km/s", maxonly=False):
    binsize  = np.zeros((2, wave.size))
    binsizet = wave[1:] - wave[:-1]
    if bintype == "km/s":
        binsizet *= 2.99792458E5/wave[:-1]
    elif bintype == "A":
        pass
    elif bintype == "Hz":
        pass
    maxbin = np.max(binsizet)
    binsize[0, :-1], binsize[1, 1:] = binsizet, binsizet
    binsize[0, -1], binsize[1, 0] = maxbin, maxbin
    binsize = binsize.min(0)
    if maxonly:
        return np.max(binsize)
    else:
        return binsize


def get_subpixels(wave, nsubpix=10):
    binsize = get_binsize(wave)
    binlen = 1.0 / np.float64(nsubpix)
    interpwav = (1.0 + ((np.arange(nsubpix) - (0.5 * (nsubpix - 1.0)))[np.newaxis, :] * binlen * binsize[:, np.newaxis] / 2.99792458E5))
    subwave = (wave.reshape(wave.size, 1) * interpwav).flatten(0)
    return subwave


def generate_wave(wavemin=3200.0, wavemax=5000.0, velstep=2.5, nsubpix=10):
    npix = np.log10(wavemax/wavemin) / np.log10(1.0+velstep/299792.458)
    npix = np.int(npix)
    wave = wavemin*(1.0+velstep/299792.458)**np.arange(npix)
    # Now generate a subpixellated wavelength grid
    subwave = get_subpixels(wave, nsubpix=nsubpix)
    return wave, subwave


def simulate_random_dla_Lya(rest_window=30.0, proxqso=0.0):
    """
    rest_window = Number of REST Angstroms to the left and right of the central DLA profile to use
    rest_proxqso = number of REST Angstroms to the left of the QSO to use for generating a DLA (only used if non-zero)
    """
    filename = '../data/DR1_quasars_master_trimmed.csv'
    t_trim = Table.read(filename, format='ascii.csv')
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
