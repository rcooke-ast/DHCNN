import numpy as np
from scipy.special import wofz
from scipy import interpolate
from matplotlib import pyplot as plt
import astropy.io.fits as fits
import astropy.units as u
from astropy.table import Table
from astropy import cosmology
from pyigm.fN.fnmodel import FNModel
from pyigm.fN.mockforest import monte_HIcomp
from pyigm.fN.mockforest import mk_mock
from pyigm.continuum import quasar as pycq
from linetools.spectra.xspectrum1d import XSpectrum1D
from IPython import embed

# Set some constants
const = (2.99792458E5 * (2.0 * np.sqrt(2.0 * np.log(2.0))))


def load_atomic(return_HIwav=True):
    """
    Load the atomic transitions data
    """
#    dir = "/Users/rcooke/Software/ALIS/alis/data/"
    atmname = "atomic.xml"
    print("Loading atomic data")
    # If the user specifies the atomic data file, make sure that it exists
    try:
        dir = "/home/rcooke/Software/ALIS/alis/data/"
        table = parse_single_table(dir+atmname)
    except:
        dir = "/cosma/home/durham/rcooke/Software/ALIS/alis/data/"
        table = parse_single_table(dir+atmname)
    isotope = table.array['MassNumber'].astype("|S3").astype(np.object)+table.array['Element']
    atmdata = dict({})
    atmdata['Ion'] = np.array(isotope+b"_"+table.array['Ion']).astype(np.str)
    atmdata['Wavelength'] = np.array(table.array['RestWave'])
    atmdata['fvalue'] = np.array(table.array['fval'])
    atmdata['gamma'] = np.array(table.array['Gamma'])
    if return_HIwav:
        ww = np.where(atmdata["Ion"] == "1H_I")
        wavs = atmdata["Wavelength"][ww][3:]
        return wavs*u.AA
    else:
        return atmdata


def voigt(par, wavein, logn=True):
    epar = [1215.6701 * 1.0e-8, 0.4164, 6.265E8]  # Note, the wavelength is converted to cm
    # Column density
    if logn:
        cold = 10.0 ** par[0]
    else:
        cold = par[0]
    # Redshift
    zp1 = par[1] + 1.0
    wv = epar[0]
    # Doppler parameter
    bl = par[2] * wv / 2.99792458E5
    a = epar[2] * wv * wv / (3.76730313461770655E11 * bl)
    cns = wv * wv * epar[1] / (bl * 2.002134602291006E12)
    cne = cold * cns
    ww = (wavein * 1.0e-8) / zp1
    v = wv * ww * ((1.0 / ww) - (1.0 / wv)) / bl
    tau = cne * wofz(v + 1j * a).real
    return np.exp(-tau)


def get_NHI(NHImin=19.2, NHImax=21.0, zmin=2.6, zmax=3.4, rstate=None, seed=None, numgen=100):

    # Get a random state so that the noise and components can be reproduced
    if seed is not None:
        rstate = np.random.RandomState(seed)

    # Get the CDDF
    NHI = np.array([12.0, 15.0, 17.0, 18.0, 20.0, 21.0, 21.5, 22.0])
    sply = np.array([-9.72, -14.41, -17.94, -19.39, -21.28, -22.82, -23.95, -25.50])
    boost = np.log10(1.6*numgen)  # Artificially scale f(N) so that we get numgen DLAs all at once
    params = dict(sply=sply+boost)
    fN_model = FNModel('Hspline', pivots=NHI, param=params, zmnx=(2., 5.))
    cosmo = cosmology.core.FlatLambdaCDM(70., 0.3)

    # Generate a fake spectrum
    #print("Note :: I made some changes in pyigm.fn.mockforest.py to avoid convolution and noise")
    #print("Note :: I made some changes in pyigm.fn.mockforest.py to perform my own subpixellation")
    HI_comps = monte_HIcomp((zmin, zmax), fN_model, NHI_mnx=(NHImin, NHImax), bfix=None, cosmo=cosmo, rstate=rstate)
    NHIvals = HI_comps['lgNHI'].data[:numgen]
    saveit = False
    if saveit:
        np.save("../data/NHIvals", NHIvals)

    # Check the result
    checkit = False
    if checkit:
        plt.hist(NHIvals, bins=np.linspace(NHImin, NHImax, numgen//10000), log=True)
        plt.show()

    # Return the results
    return NHIvals


def simulate_random_dla_Lya(rest_window=30.0, proxqso=0.0):
    """
    rest_window = Number of REST Angstroms to the left and right of the central DLA profile to use
    rest_proxqso = number of REST Angstroms to the left of the QSO to use for generating a DLA (only used if non-zero)
    """
    filename = '../data/DR1_quasars_master_trimmed.csv'
    t_trim = Table.read(filename, format='ascii.csv')
    # Select a random QSO
    nqso = t_trim['Name_Adopt'].size
    embed()
    qidx = np.random.randint(0, nqso)
    qso = t_trim[qidx]
    zem = qso['zem_Adopt']
    wavstr = qso['WavStart']+rest_window
    if (proxqso > 0.0):
        wavstr = max(wavstr, (1+zem)*(1215.6701 - proxqso))
    zdla = np.random.uniform(wavstr, (1+zem)*1215.6701)/1215.6701 - 1
    # Generate an N(H I) of a DLA from f(N)
    # The range of interest is:
    # 19.3 < log N(H I) < 21.0
    # 2.6 < z < 3.4
    logNHI = get_NHI(NHImin=19.2, NHImax=21.0)[0]
    # Load the data
    dat = fits.open("../data/{0:s}.fits".format(qso['Name_Adopt']))
    dwv = (1+zdla)*rest_window
    cwv = (1+zdla)*1215.6701
    wave = dat[1].data['WAVE']
    ww = np.where((wave>cwv-dwv) & (wave<cwv+dwv))
    wave = wave[ww]
    cont = dat[1].data['CONTINUUM'][ww]
    flux = dat[1].data['FLUX'][ww] * cont
    flue = dat[1].data['ERR'][ww] * cont
    stat = dat[1].data['STATUS'][ww]
    # Check Continuum/Noise ratio
    if (cont[cont.size//2]/flue[cont.size//2]) < 10:
        print("Low continuum/noise ratio")
        #return None, None

    # Generate a DLA Lya profile
    model = voigt([logNHI, zdla, 15.0], wave)
    fluxnew = flux.copy()
    fluxnew *= model
    # Determine the extra noise needed to maintain the same flue
    gd = np.where(stat == 1)
    bd = np.where(stat != 1)
    if (bd[0].size != 0):
        print("Number of bad pixels = {0:d}".format(bd[0].size))
        #return None, None
    # embed()
    exnse = np.random.normal(np.zeros(flue[gd].size), flue[gd] * np.sqrt(1 - model[gd] ** 2))
    # Add this noise to the data
    fluxnew[gd] += exnse
    plotit = True
    if plotit:
        # Plot the result to see if it looks OK
        plt.plot(wave, flux, 'k-', drawstyle='steps')
        plt.plot(wave, fluxnew, 'r-', drawstyle='steps')
        plt.plot(wave, flue, 'b-', drawstyle='steps')
        plt.show()
    return wave, fluxnew


def convolve(y, x, vfwhm):
    vsigd = vfwhm / const
    ysize = y.shape[0]
    fsigd = 6.0 * vsigd
    dwav = np.gradient(x) / x
    df = int(np.min([np.int(np.ceil(fsigd / dwav).max()), ysize // 2 - 1]))
    yval = np.zeros(2 * df + 1)
    yval[df:2 * df + 1] = (x[df:2 * df + 1] / x[df] - 1.0) / vsigd
    yval[:df] = (x[:df] / x[df] - 1.0) / vsigd
    gaus = np.exp(-0.5 * yval * yval)
    size = ysize + gaus.size - 1
    fsize = 2 ** np.int(np.ceil(np.log2(size)))  # Use this size for a more efficient computation
    conv = np.fft.fft(y, fsize, axis=0)
    if y.ndim == 1:
        conv *= np.fft.fft(gaus / gaus.sum(), fsize)
    else:
        conv *= np.fft.fft(gaus / gaus.sum(), fsize).reshape((fsize, 1))
    ret = np.fft.ifft(conv, axis=0).real.copy()
    del conv
    if y.ndim == 1:
        return ret[df:df + ysize]
    else:
        return ret[df:df + ysize, :]


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


def rebin_subpix(flux, nsubpix=10):
    model = flux.reshape(flux.size//nsubpix, nsubpix).sum(axis=1) / np.float64(nsubpix)
    return model


def generate_wave(wavemin=3200.0, wavemax=5000.0, velstep=2.5, nsubpix=10):
    npix = np.log10(wavemax/wavemin) / np.log10(1.0+velstep/299792.458)
    npix = np.int(npix)
    wave = wavemin*(1.0+velstep/299792.458)**np.arange(npix)
    # Now generate a subpixellated wavelength grid
    subwave = get_subpixels(wave, nsubpix=nsubpix)
    return wave, subwave


def generate_fakespectra(zqso, wave=None, subwave=None, NHI_mnx=(12.,22.), usecont="WFC3", nsubpix=10, snr=30.0, seed=1234, vfwhm=7.0):
    add_noise = True
    if snr <= 0.0:
        add_noise = False
        snr = 1.0
    # Get a random state so that the noise and components can be reproduced
    rstate = np.random.RandomState(seed)
    # Define the wavelength coverage
    if wave is None or subwave is None:
        wave, subwave = generate_wave(wavemax=1250.0*(1.0+zqso), nsubpix=nsubpix)
        wave *= u.AA
        subwave *= u.AA
    # Get the CDDF
    NHI = np.array([12.0, 15.0, 17.0, 18.0, 20.0, 21.0, 21.5, 22.0])
    sply = np.array([-9.72, -14.41, -17.94, -19.39, -21.28, -22.82, -23.95, -25.50])
    params = dict(sply=sply)
    fN_model = FNModel('Hspline', pivots=NHI, param=params, zmnx=(2., 5.))
    # Generate a fake spectrum
    print("Note :: I made some changes in pyigm.fn.mockforest.py to avoid convolution and noise")
    print("Note :: I made some changes in pyigm.fn.mockforest.py to perform my own subpixellation")
    _, HI_comps, mock_subspec = mk_mock(wave, zqso, fN_model, NHI_mnx=NHI_mnx, fwhm=0.0, s2n=0.0, subwave=subwave)
    # Generate a quasar continuum
    if usecont == "WFC3":
        # There are 53 WFC3 QSO spectra
        print("Note :: I made some changes in pyigm.continuum.quasar.py to return the raw WFC3 spectra")
        conti, wfc3_idx = pycq.wfc3_continuum(zqso=zqso, get_orig=True, rstate=rstate)
        convcont = convolve(conti.flux, conti.wavelength, 5000.0)  # Need to smooth out the noisy WFC3 spectra
        cspl = interpolate.interp1d(conti.wavelength, convcont, kind='cubic', bounds_error=False, fill_value="extrapolate")
        cflux = cspl(subwave)
    elif usecont == "UVES_popler":
        # There are 104 QSOs from UVES_popler within the right redshift range...
        print("NOT IMPLEMENTED")
        assert(False)
    else:
        print("CONT TYPE NOT AVAILABLE")
        assert(False)
    # Create the final subpixellated model
    model_flux_sub = mock_subspec[0].flux * cflux
    return subwave, model_flux_sub


if __name__ == "__main__":
    wave, flux = simulate_random_dla_Lya()
    if wave is None:
        print("There must have been bad pixels, or too low S/N")
    
    # wave = np.linspace(1170, 1270, 10000)
    # model = voigt([19.2, 0.0, 15.0], wave)
    # plt.plot(wave, model, 'k')
    # model = voigt([21.0, 0.0, 15.0], wave)
    # plt.plot(wave, model, 'k')
    # plt.show()
