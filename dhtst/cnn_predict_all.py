import numpy as np
from linetools.spectra.xspectrum1d import XSpectrum1D
import astropy.units as units
import astropy.io.fits as fits
from astropy.table import Table
from tensorflow.python.keras.models import load_model
from cnn_train import get_available_gpus, load_dataset_trueqsos
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from IPython import embed
from matplotlib import pyplot as plt


# Now start the calculation...
print("WARNING - SPEC_LEN NEEDS  TO BE SET ACCORDING TO THE MODEL BEING LOADED!!!")
spec_len = 271
velstep = 2.5    # Pixel size in km/s

LyaD = 1215.3394
LyaH = 1215.6701
vfwhm = 7.0  # velocity FWHM in km/s


def redisperse(wave, flux):
    # Setup the spectrum based on a wavelength and flux
    spec = XSpectrum1D.from_tuple((wave, flux))

    # Generate a new wavelength array
    wavemin, wavemax = wave.min(), wave.max()
    newvpix = velstep
    npix = np.log10(wavemax / wavemin) / np.log10(1.0 + newvpix / 299792.458)
    npix = np.int(npix)
    newwave = wavemin * (1.0 + newvpix / 299792.458) ** np.arange(npix)

    # Resample the data
    newspec = spec.rebin(newwave * units.AA)
    return newspec.wavelength.value, newspec.flux.value


def load_all_quasars():
    t = Table.read('../data/DR1_quasars_master.csv', format='ascii.csv')
    t_trim = t[np.where((t['zem_Adopt'].data > 1.5) & (t['zem_Adopt'].data < 5.0))]
    nqso = t_trim['Name_Adopt'].size
    all_wave, all_flux, all_name = [], [], []
    # Go through all quasars, redisperse, and cut out regions in the Lya forest
    for qq in range(nqso):
        qso = t_trim[qq]
        disp = qso['Dispersion']
        zem = qso['zem_Adopt']
        # Load the data
        try:
            dat = fits.open("../data/{0:s}.fits".format(qso['Name_Adopt']))
        except FileNotFoundError:
            print("File not found: ", qso['Name_Adopt'])
            continue
        wave = dat[1].data['WAVE']
        flux = dat[1].data['FLUX']
        if wave.shape[0] != 1: embed()
        wave = wave[0,:]
        flux = flux[0, :]
        # Redisperse
        if disp != velstep:
            newwave, newflux = redisperse(wave, flux)
        else:
            newwave, newflux = wave.copy(), flux.copy()
        try:
            ww = np.where(newwave < 1215.6701*(1+zem))
        except:
            embed()
        if ww[0].size < spec_len:
            continue
        # If we've made it to hear, then we need to predict on this QSO. Add it to the list.
        all_wave.append(newwave.copy())
        all_flux.append(newflux.copy())
        all_name.append(t_trim[qq]['Name_Adopt'])
    return all_wave, all_flux, all_name


print("Loading model")
mnum = 0
batch_sz = 1
loadname = 'fit_data/model_{0:03d}.hdf5'.format(mnum)
# Construct network
ngpus = len(get_available_gpus())
if ngpus > 1:
    model = load_model(loadname, compile=False)
    # Make this work on multiple GPUs
    gpumodel = multi_gpu_model(model, gpus=ngpus)
else:
    gpumodel = load_model(loadname, compile=False)

cntr = 0
# Load the data
print("Loading data")
allWave, allFlux, allName = load_all_quasars()
nqso = len(allWave)
tst_input = ({})
offs = (spec_len-1)//2
catalogue = dict(name=[], prob=[], zabs=[])
for qso in range(nqso):
    print("Searching QSO =", allName[qso])
    # Load a quasar spectrum
    wave, flux = allWave[qso], allFlux[qso]
    # Construct the input and prediction arrays
    IDarr, SHarr = np.zeros(wave.size), np.zeros(wave.size)
    inarray = np.zeros((flux.size-spec_len+1, spec_len, 1))
    wa = np.arange(offs, flux.size-offs).reshape((inarray.shape[0],1))
    df = np.arange(-offs,spec_len//2+1).reshape((1,spec_len))
    inarray[:,:,0] = flux[wa+df]
    tst_input['input_1'] = inarray
    # Predict!!
    tst_output = gpumodel.predict(tst_input)
    IDarr[offs:flux.size-offs] = tst_output[0].flatten()
    SHarr[offs:flux.size-offs] = tst_output[1].flatten()
    # Parse the IDarr to find all suitable systems and store in an array.
    ww = np.where(IDarr > 0.1)[0]
    if ww.size == 0: continue
    msk = np.zeros(ww.size)
    while True:
        wmin = np.where(msk == 0)[0]
        ws = np.where((ww > ww[wmin]-30) & (ww < ww[wmin]+30))
        pix = ww[ws]
        prob = np.mean(IDarr[pix])
        dwav = (velstep/299792.458)*wave[ww[wmin]]
        wcen = wave[pix] + dwav*SHarr[pix]
        zabs = wcen/LyaD - 1
        catalogue['name'].append(allName[qso])
        catalogue['prob'].append(prob)
        catalogue['zabs'].append(zabs)
        print(allName[qso], prob, zabs)
        fig, axs = plt.subplots(3, 1, sharex=True)
        axs[0].plot(wave[pix]/(1+zabs), flux[pix], 'k-')
        axs[0].axvline(LyaD, color='r')
        axs[1].plot(wave[pix]/(1+zabs), IDarr[pix], 'k-')
        axs[1].axvline(LyaD, color='r')
        axs[2].plot(wave[pix]/(1+zabs), SHarr[pix], 'k-')
        axs[2].axvline(LyaD, color='r')
        plt.show()
        plt.clf()
        cntr += 1
        # Update the mask
        msk[ws] = 1
        if np.all(msk): break
    print("Candidates found so far = ", cntr)

catout = open("DI_catalogue.dat", 'w')
ncat = len(catalogue['zabs'])
for cc in range(ncat):
    catout.write("{0:s} {1:f} {2:f}\n".format(catalogue['name'][cc], catalogue['prob'][cc], catalogue['zabs'][cc]))
print("{0:d} candidates found".format(ncat))
