import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import load_model
from cnn_train import get_available_gpus, load_dataset_trueqsos, get_restwin
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
import utils
import time
from IPython import embed


# Now start the calculation...
velstep = 2.5    # Pixel size in km/s
zdla_min, zdla_max = 2.5, 2.93#3.4
NHI_min, NHI_max = 17.0, 18.2
DH_min, DH_max = -4.7, -4.5
turb_min, turb_max = 2.0, 7.0
temp_min, temp_max = 1.0E4, 2.5E4
shft_min, shft_max = -10, +10

LyaD = 1215.3394
LyaH = 1215.6701
vfwhm = 7.0  # velocity FWHM in km/s


def yield_data_trueqso(wave, flux, flue, stat, zem, batch_sz, spec_len):
    """
    Based on imprinting a DLA on observations of _real_ QSOs
    """
    flag_fake = 0.#15  # Generate pure H absorption sometimes (no D I line)
    restwin = get_restwin(spec_len)
    nqso = zem.shape[0]
    indict = ({})
    # Setup batch params
    X_batch = np.zeros((batch_sz, spec_len, 1))
    yld_NHI = np.random.uniform(NHI_min, NHI_max, batch_sz)
    yld_DH = np.random.uniform(DH_min, DH_max, batch_sz)
    yld_dopp = np.random.uniform(turb_min, turb_max, batch_sz)
    yld_temp = np.random.uniform(temp_min, temp_max, batch_sz)
    label_ID = np.zeros(batch_sz)
    label_sh = np.random.uniform(shft_min, shft_max, batch_sz)
    flag_none = np.random.uniform(0, 1, batch_sz)
    # Prepare the batch
    cntr_batch = 0
    while cntr_batch < batch_sz:
        # Select a random QSO
        qso = np.random.randint(0, nqso)
        zdmin = max(zdla_min, ((wave[0, qso] + restwin) / LyaD) - 1.0)  # Can't have a DLA below the data for this QSO
        zdmax = min(zdla_max, zem[qso])  # Can't have a DLA above the QSO redshift
        pxmin = np.argmin(np.abs(wave[:, qso] - LyaD * (1 + zdmin)))
        pxmax = np.argmax(np.abs(wave[:, qso] - LyaD * (1 + zdmax)))
        absp = np.random.randint(pxmin, pxmax)
        imin = absp - spec_len // 2
        imax = imin + spec_len
        fmin = pxmin#absp - spec_len // 2
        fmax = pxmax#imin + spec_len
        bd = np.where(stat[imin:imax, qso] == 0)
        if bd[0].size == 0 and stat[imin:imax, qso].size == spec_len:
            # This is a good system fill it in
            zpix = absp + int(np.floor(label_sh[cntr_batch]))
            wval = wave[zpix, qso] + (wave[zpix + 1, qso] - wave[zpix, qso]) * (
                    label_sh[cntr_batch] - np.floor(label_sh[cntr_batch]))
            zval = (wval / LyaD) - 1
            label_ID[cntr_batch] = stat[zpix, qso] - 1  # 0 for no absorption, 1 for absorption
            if label_ID[cntr_batch] == 1:
                HI_send, DH_send = yld_NHI[cntr_batch], yld_DH[cntr_batch]
                if flag_none[cntr_batch] < flag_fake:
                    DH_send = -10  # Sometimes don't put a D I lines there.
                    label_ID[cntr_batch] = 0
                elif flag_none[cntr_batch] < 2 * flag_fake:
                    # Sometimes don't put a H I lines there.
                    HI_send = yld_NHI[cntr_batch] - 10
                    DH_send = yld_DH[cntr_batch] + 10
                    label_ID[cntr_batch] = 0
                model = utils.DH_model([HI_send, DH_send, zval, yld_dopp[cntr_batch], yld_temp[cntr_batch]],
                                       wave[imin:imax, qso], vfwhm)
                # Determine the extra noise needed to maintain the same flue
                exnse = np.random.normal(np.zeros(imax-imin), flue[imin:imax, qso] * np.sqrt(1 - model ** 2))
                # Add this noise to the data
                spec = flux[:, qso].copy()
                spec[imin:imax] = flux[imin:imax, qso] * model + exnse
                # Don't optimize shift when there's no absorption - zero values are masked
                label_sh[cntr_batch] *= label_ID[cntr_batch]
                cntr_batch += 1
            # Increment the counter
    # Store output
    return (wave[:, qso], spec, zval)

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
print("WARNING - SPEC_LEN NEEDS  TO BE SET ACCORDING TO THE MODEL BEING LOADED!!!")
spec_len = 271
# Load the data
allWave, allFlux, allFlue, allStat, allzem = load_dataset_trueqsos()
# Put an artificial D/H system on the spectrum
wave, spec, zval = yield_data_trueqso(allWave, allFlux, allFlue, allStat, allzem, batch_sz, spec_len)
IDarr, SHarr = np.zeros(wave.size), np.zeros(wave.size)
tst_input = ({})
# for pp in range(spec_len//2, spec.size-spec_len//2):
#     if pp%100==0: print(pp, spec_len//2, spec.size-spec_len//2)
#     tst_input['input_1'] = spec[pp-spec_len//2:pp-spec_len//2 + spec_len].reshape((1, spec_len, 1))
#     tst_output = gpumodel.predict(tst_input)
#     IDarr[pp] = tst_output[0].flatten()[0]
#     SHarr[pp] = tst_output[1].flatten()[0]

a = time.time()
offs = (spec_len-1)//2
inarray = np.zeros((spec.size-spec_len+1, spec_len, 1))
wa = np.arange(offs, spec.size-offs).reshape((inarray.shape[0],1))
df = np.arange(-offs,spec_len//2+1).reshape((1,spec_len))
inarray[:,:,0] = spec[wa+df]
tst_input['input_1'] = inarray
tst_output = gpumodel.predict(tst_input)
IDarr[offs:spec.size-offs] = tst_output[0].flatten()
SHarr[offs:spec.size-offs] = tst_output[1].flatten()
print("time/spec =", time.time()-a)
print(zval)

np.savetxt("test_spec/results.dat", np.transpose((wave/(1+zval), spec, IDarr, SHarr)))
wavplt = wave/(1+zval)
fig, axs = plt.subplots(3,1, sharex=True)
axs[0].plot(wavplt, spec, 'k-')
axs[0].axvline(LyaD, color='r')
axs[1].plot(wavplt, IDarr, 'k-')
axs[1].axvline(LyaD, color='r')
axs[2].plot(wavplt, SHarr, 'k-')
axs[2].axvline(LyaD, color='r')
plt.show()
