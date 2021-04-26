import numpy as np
from scipy.special import wofz
from matplotlib import pyplot as plt
import astropy.io.fits as fits
from astropy.table import Table
from astropy import cosmology
from pyigm.fN.fnmodel import FNModel
from pyigm.fN.mockforest import monte_HIcomp
from IPython import embed


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
