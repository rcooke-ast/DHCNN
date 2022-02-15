import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import load_model
from cnn_train import get_available_gpus, load_dataset_trueqsos, get_restwin
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
import utils

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


def yield_data_trueqso(wave, flux, flue, stat, zem, batch_sz, spec_len, debug=False):
    """
    Based on imprinting a DLA on observations of _real_ QSOs
    """
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
    label_sh = np.linspace(shft_min, shft_max, batch_sz)
    # Prepare the batch
    cntr_batch = 0
    while True:
        # Select a random QSO
        qso = np.random.randint(0, nqso)
        zdmin = max(zdla_min, ((wave[0, qso]+restwin)/LyaD) - 1.0)  # Can't have a DLA below the data for this QSO
        zdmax = min(zdla_max, zem[qso])  # Can't have a DLA above the QSO redshift
        pxmin = np.argmin(np.abs(wave[:, qso] - LyaD * (1 + zdmin)))
        pxmax = np.argmax(np.abs(wave[:, qso] - LyaD * (1 + zdmax)))
        absp = np.random.randint(pxmin, pxmax)
        imin = absp - spec_len // 2# + int(np.round(label_sh[cntr_batch]))
        imax = imin + spec_len# + int(np.round(label_sh[cntr_batch]))
        bd = np.where(stat[imin:imax, qso] == 0)
        if bd[0].size == 0 and stat[imin:imax, qso].size == spec_len:
            for cntr_batch in range(0, batch_sz):
                zpix = absp + int(np.floor(label_sh[cntr_batch]))
                # This is a good system fill it in
                label_ID[cntr_batch] = stat[zpix, qso]-1  # 0 for no absorption, 1 for absorption
                label_sh[cntr_batch] *= label_ID[cntr_batch]  # Don't optimize shift when there's no absorption - zero values are masked
                if debug:
                    plt.subplot(batch_sz, 1, cntr_batch + 1)
                    plt.plot(wave[imin:imax, qso], flux[imin:imax, qso], 'k-', drawstyle='steps-mid')
                if stat[zpix, qso] == 2 or debug:
                    wval = wave[zpix, qso] + (wave[zpix+1, qso]-wave[zpix, qso])*(label_sh[cntr_batch]-np.floor(label_sh[cntr_batch]))
                    zval = (wval/LyaD) - 1
                    model = utils.DH_model([yld_NHI[cntr_batch], yld_DH[cntr_batch], zval, yld_dopp[cntr_batch], yld_temp[cntr_batch]],
                                           wave[imin:imax, qso], vfwhm)
                    # Determine the extra noise needed to maintain the same flue
                    exnse = np.random.normal(np.zeros(spec_len), flue[imin:imax, qso] * np.sqrt(1 - model ** 2))
                    # Add this noise to the data
                    X_batch[cntr_batch, :, 0] = flux[imin:imax, qso] * model + exnse
                    if debug:
                        plt.plot(wave[imin:imax, qso], X_batch[cntr_batch, :, 0], 'r-', drawstyle='steps-mid')
                        plt.axvline(LyaD*(1+zval))
                else:
                    X_batch[cntr_batch, :, 0] = flux[imin:imax, qso]
                if debug:
                    plt.title("{0:f} - {1:f}".format(label_ID[cntr_batch], label_sh[cntr_batch]))
        if debug:
            plt.show()
        indict['input_1'] = X_batch.copy()
        # Store output
        outdict = {'output_ID': label_ID,
                   'output_sh': label_sh}
        if not debug:
            return (indict, outdict)

mnum = 0
batch_sz = 30
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
spec_len = 179
while True:
    print(cntr)
    allWave, allFlux, allFlue, allStat, allzem = load_dataset_trueqsos()
    test_input, test_output = yield_data_trueqso(allWave, allFlux, allFlue, allStat, allzem, batch_sz, spec_len)
    input_arr = test_input['input_1']
    test_vals = gpumodel.predict(test_input)
    ID = np.zeros(input_arr.shape[1])
    sh = np.zeros(input_arr.shape[1])
    ID[spec_len//2-batch_sz//2:spec_len//2-batch_sz//2+batch_sz] = test_output['output_ID']
    sh[spec_len//2-batch_sz//2:spec_len//2-batch_sz//2+batch_sz] = test_output['output_sh']
    cntr += 1
    if np.any(ID!=0):
        break
print(test_output['output_ID'])
print(test_output['output_sh'])
IDt = np.zeros(input_arr.shape[1])
sht = np.zeros(input_arr.shape[1])
print(test_vals)
print(len(test_vals))
pred_ID = test_vals[0].flatten()
pred_sh = test_vals[1].flatten()
IDt[spec_len // 2 - batch_sz // 2:spec_len // 2 - batch_sz // 2 + batch_sz] = pred_ID
sht[spec_len // 2 - batch_sz // 2:spec_len // 2 - batch_sz // 2 + batch_sz] = pred_sh

for pp in range(batch_sz):
    plt.subplot(6,5,pp+1)
    if pred_ID[pp] > 0.85:
        plt.plot(input_arr[pp, :, 0])
        plt.axvline(spec_len // 2 + test_output['output_sh'][pp], color='r')
        plt.axvline(spec_len // 2 + pred_sh[pp], color='b')
plt.show()

# plt.subplot(311)
# plt.plot(input_arr[0, :, 0])
# plt.subplot(312)
# plt.plot(ID, 'b-')
# plt.plot(IDt, 'r-')
# plt.subplot(313)
# plt.plot(sh, 'b-')
# plt.plot(sht, 'r-')
# plt.show()
