import os
import pickle
import numpy as np
from astropy.table import Table
import astropy.io.fits as fits
from IPython import embed
from scipy.ndimage import uniform_filter1d
from scipy import interpolate
from matplotlib import pyplot as plt
import utils

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.python.keras.utils.vis_utils import plot_model
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.python.keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.optimizer_v1 import Adam

from contextlib import redirect_stdout

# Disable eager execution
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

# Now start the calculation...
velstep = 2.5    # Pixel size in km/s
zdla_min, zdla_max = 2.5, 2.93#3.4
NHI_min, NHI_max = 17.0, 18.2
DH_min, DH_max = -4.7, -4.5
turb_min, turb_max = 2.0, 7.0
temp_min, temp_max = 1.0E4, 2.5E4
# turb_min, turb_max = 1, 2
# temp_min, temp_max = 0.0E4, 0.005E4
shft_min, shft_max = -10, +10

LyaD = 1215.3394
LyaH = 1215.6701
vfwhm = 7.0  # velocity FWHM in km/s


# Define custom loss
def mse_mask():
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        return K.mean(mask * K.sqrt(K.abs(y_pred - y_true)), axis=-1)
        # return K.mean( (y_true/(y_true+epsilon)) * K.square(y_pred - y_true), axis=-1)
        #return K.mean(K.square(y_pred - y_true), axis=-1)
    # Return a function
    return loss


def get_restwin(spec_len):
    # Rest window in angstroms (the full window size is twice this)
    return 0.5 * spec_len * velstep * LyaD / 299792.458


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def save_obj(obj, dirname):
    with open(dirname + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(dirname):
    with open(dirname + '.pkl', 'rb') as f:
        return pickle.load(f)


def hyperparam_orig(mnum):
    """Generate a random set of hyper parameters

    mnum (int): Model index number
    """
    # Define all of the allowed parameter space
    allowed_hpars = dict(spec_len           = [179],
                         learning_rate      = [0.0001],
                         lr_decay           = [0.0],
                         l2_regpen          = [0.0],
                         dropout_prob       = [0.1],
                         num_epochs         = [20],
                         batch_size         = [512],
                         num_batch_train    = [128],
                         num_batch_validate = [128],
                         ID_loss            = [1.0],
                         sh_loss            = [1.0],
                         # Number of filters in each convolutional layer
                         conv_filter_1 = [512],
                         conv_filter_2 = [512],
                         conv_filter_3 = [512],
                         # Kernel size
                         conv_kernel_1 = [6],
                         conv_kernel_2 = [7],
                         conv_kernel_3 = [4],
                         # Stride of each kernal
                         conv_stride_1 = [1],
                         conv_stride_2 = [1],
                         conv_stride_3 = [1],
                         # Pooling kernel size
                         pool_kernel_1 = [2],
                         pool_kernel_2 = [2],
                         pool_kernel_3 = [2],
                         # Pooling stride
                         pool_stride_1 = [2],
                         pool_stride_2 = [2],
                         pool_stride_3 = [2],
                         # Fully connected layers
                         fc1_neurons   = [64],
                         fc2_ID_neurons = [32],
                         fc2_sh_neurons = [256]
                         )
    # Generate dictionary of values
    hyperpar = dict({})
    for key in allowed_hpars.keys():
        hyperpar[key] = np.random.choice(allowed_hpars[key])
    # Save these parameters and return the hyperpar
    save_obj(hyperpar, 'fit_data/model_{0:03d}'.format(mnum))
    return hyperpar


def custom_objects(hpar):
    # Loss functions
    loss = {'output_ID': 'binary_crossentropy',
            'output_sh': mse_mask()}
    # Loss weights
    loss_weights = {'output_ID': hpar['ID_loss'],
                    'output_sh': hpar['sh_loss']}
    # Optimizer
    decay = hpar['lr_decay']*hpar['learning_rate']/hpar['num_epochs']
    optadam = Adam(lr=hpar['learning_rate'], decay=decay)
    return loss, loss_weights, optadam


def hyperparam(mnum):
    """Generate a random set of hyper parameters

    mnum (int): Model index number
    """
    # Define all of the allowed parameter space
    allowed_hpars = dict(learning_rate      = [0.0005, 0.0007, 0.0010, 0.0030, 0.0050, 0.0070, 0.0100],
                         lr_decay           = [0.0, 1.0],
                         l2_regpen          = [0.0, 0.00001, 0.00010, 0.00100, 0.00500, 0.01000],
                         dropout_prob       = [0.0, 0.01, 0.02, 0.05],
                         num_epochs         = [30, 50, 100],
                         batch_size         = [100, 500, 1000, 2000, 5000],
                         num_batch_train    = [128, 256, 512, 1024],
                         num_batch_validate = [32, 64, 128],
                         # Number of filters in each convolutional layer
                         conv_filter_1 = [80, 96, 128, 192, 256],
                         conv_filter_2 = [80, 96, 128, 192, 256],
                         conv_filter_3 = [80, 96, 128, 192, 256],
                         # Kernel size
                         conv_kernel_1 = [20, 22, 24, 26, 28, 32, 40, 48, 54],
                         conv_kernel_2 = [10, 14, 16, 20, 24, 28, 32, 34],
                         conv_kernel_3 = [10, 14, 16, 20, 24, 28, 32, 34],
                         # Stride of each kernal
                         conv_stride_1 = [1, 2, 4, 6],
                         conv_stride_2 = [1, 2, 4, 6],
                         conv_stride_3 = [1, 2, 4, 6],
                         # Pooling kernel size
                         pool_kernel_1 = [2, 3, 4, 6],
                         pool_kernel_2 = [2, 3, 4, 6],
                         pool_kernel_3 = [2, 3, 4, 6],
                         # Pooling stride
                         pool_stride_1 = [1, 2, 3],
                         pool_stride_2 = [1, 2, 3],
                         pool_stride_3 = [1, 2, 3],
                         # Fully connected layers
                         fc1_neurons   = [256, 512, 1024, 2048],
                         fc2_ID_neurons = [32, 64, 128, 256],
                         fc2_sh_neurons = [32, 64, 128, 256]
                         )
    # Generate dictionary of values
    hyperpar = dict({})
    for key in allowed_hpars.keys():
        hyperpar[key] = np.random.choice(allowed_hpars[key])
    # Save these parameters and return the hyperpar
    save_obj(hyperpar, 'fit_data/model_{0:03d}'.format(mnum))
    return hyperpar


def generate_dataset_trueqsos(spec_len):
    rest_window = get_restwin(spec_len)
    filename = '../data/DR1_quasars_master_trimmed.csv'
    t_trim = Table.read(filename, format='ascii.csv')
    # Select a random QSO
    nqso = t_trim['Name_Adopt'].size
    # Go through each QSO one at a time

    # Just get the size of the data first
    maxsz = 0
    for qq in range(nqso):
        qso = t_trim[qq]
        zem = qso['zem_Adopt']
        # Load the data
        dat = fits.open("../data/{0:s}.fits".format(qso['Name_Adopt']))
        wvmax = (1+zem)*(LyaH+rest_window)  # Data will not be used to the right of the QSO Lya emission line + rest+window (the rest_window is to include the DLA profile)
        wvmin = (1+zdla_min)*(LyaH-rest_window)  # Now data are needed below this DLA cutoff redshift... minus the rest window
        wave = dat[1].data['WAVE']
        ww = np.where((wave > wvmin) & (wave < wvmax))
        sz = wave[ww].size
        if sz > maxsz: maxsz = sz
        stat = dat[1].data['STATUS'][ww]
        bd = np.where(stat != 1)
        if bd[0].size != 0:
            print("Number of bad pixels in QSO {0:d} = {1:d}".format(qq, bd[0].size))
        gd = np.where(stat == 1)
        if gd[0].size < spec_len:
            print("WARNING :: Not enough good pixels in QSO {0:d}".format(qq))
    # Generate the data arrays and insert the data
    allWave = np.zeros((maxsz, nqso))
    allFlux = np.zeros((maxsz, nqso))
    allFlue = np.zeros((maxsz, nqso))
    allStat = np.zeros((maxsz, nqso))
    allzem  = np.zeros(nqso)
    goodID = 0
    for qq in range(nqso):
        qso = t_trim[qq]
        zem = qso['zem_Adopt']
        # Load the data
        dat = fits.open("../data/{0:s}.fits".format(qso['Name_Adopt']))
        wvmax = (1+zem)*(LyaH+rest_window)  # Data will not be used to the right of the QSO Lya emission line + rest+window (the rest_window is to include the DLA profile)
        wvmin = (1+zdla_min)*(LyaH-rest_window)  # Now data are needed below this DLA cutoff redshift... minus the rest window
        wave = dat[1].data['WAVE']
        ww = np.where((wave > wvmin) & (wave < wvmax))
        sz = wave[ww].size
        cont = dat[1].data['CONTINUUM'][ww]
        this_flx = dat[1].data['FLUX'][ww]
        this_fle = dat[1].data['ERR'][ww]
        if np.all(this_fle<0): continue  # Make sure this isn't bad
        allWave[:sz, goodID] = wave[ww].copy()
        allFlux[:sz, goodID] = this_flx
        allFlue[:sz, goodID] = this_fle
        allStat[:sz, goodID] = (dat[1].data['STATUS'][ww]==1).astype(np.float)
        allzem[goodID] = zem
        # Find the regions that are consistent with the continuum
        nsigma = 2
        window = 5
        wc = (np.abs((this_flx-1)/this_fle) < nsigma).astype(np.float)
        msk = (uniform_filter1d(wc, size=window) == 1).astype(np.float)
        allStat[:sz, goodID] *= (msk+1)  # So, 0=bad, 1=good, 2=clean
        goodID += 1
    # Only keep the good QSOs
    print("Good = ", goodID, "/", allWave.shape[1])
    allWave = allWave[:, :goodID]
    allFlux = allFlux[:, :goodID]
    allFlue = allFlue[:, :goodID]
    allStat = allStat[:, :goodID]
    allzem = allzem[:goodID]
    # Save the data
    np.save("../data/train_data/true_qsos_DH/wave.npy", allWave)
    np.save("../data/train_data/true_qsos_DH/flux.npy", allFlux)
    np.save("../data/train_data/true_qsos_DH/flue.npy", allFlue)
    np.save("../data/train_data/true_qsos_DH/stat.npy", allStat)
    np.save("../data/train_data/true_qsos_DH/zem.npy", allzem)
    print("Data generated successfully")
    return


def load_dataset_trueqsos():
    allWave = np.load("../data/train_data/true_qsos_DH/wave.npy")
    allFlux = np.load("../data/train_data/true_qsos_DH/flux.npy")
    allFlue = np.load("../data/train_data/true_qsos_DH/flue.npy")
    allStat = np.load("../data/train_data/true_qsos_DH/stat.npy")
    allzem = np.load("../data/train_data/true_qsos_DH/zem.npy")
    # ntrain = int(ftrain*allzem.shape[0])
    # # Select the training data
    # trainW = allWave[:, :ntrain]
    # trainF = allFlux[:, :ntrain]
    # trainE = allFlue[:, :ntrain]
    # trainS = allStat[:, :ntrain]
    # trainZ = allzem[:ntrain]
    # # Select the test data
    # testW = allWave[:, ntrain:]
    # testF = allFlux[:, ntrain:]
    # testE = allFlue[:, ntrain:]
    # testS = allStat[:, ntrain:]
    # testZ = allzem[:ntrain]
    return allWave, allFlux, allFlue, allStat, allzem


def yield_data_trueqso(wave, flux, flue, stat, zem, batch_sz, spec_len, debug=False):
    """
    Based on imprinting a DLA on observations of _real_ QSOs
    """
    flag_fake = 0.15  # Generate pure H absorption sometimes (no D I line)
    nqso = zem.shape[0]
    restwin = get_restwin(spec_len)
    while True:
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
            zdmin = max(zdla_min, ((wave[0, qso]+restwin)/LyaD) - 1.0)  # Can't have a DLA below the data for this QSO
            zdmax = min(zdla_max, zem[qso])  # Can't have a DLA above the QSO redshift
            pxmin = np.argmin(np.abs(wave[:, qso] - LyaD * (1 + zdmin)))
            pxmax = np.argmax(np.abs(wave[:, qso] - LyaD * (1 + zdmax)))
            absp = np.random.randint(pxmin, pxmax)
            imin = absp - spec_len // 2# + int(np.round(label_sh[cntr_batch]))
            imax = imin + spec_len# + int(np.round(label_sh[cntr_batch]))
            bd = np.where(stat[imin:imax, qso] == 0)
            if bd[0].size == 0 and stat[imin:imax, qso].size == spec_len:
                # This is a good system fill it in
                zpix = absp + int(np.floor(label_sh[cntr_batch]))
                wval = wave[zpix, qso] + (wave[zpix + 1, qso] - wave[zpix, qso]) * (
                            label_sh[cntr_batch] - np.floor(label_sh[cntr_batch]))
                zval = (wval / LyaD) - 1
                label_ID[cntr_batch] = stat[zpix, qso]-1  # 0 for no absorption, 1 for absorption
                label_sh[cntr_batch] *= label_ID[cntr_batch]  # Don't optimize shift when there's no absorption - zero values are masked
                if debug:
                    plt.subplot(batch_sz, 1, cntr_batch + 1)
                    plt.plot(wave[imin:imax, qso], flux[imin:imax, qso], 'k-', drawstyle='steps-mid')
                if stat[zpix, qso] == 2 or debug:
                    HI_send, DH_send = yld_NHI[cntr_batch], yld_DH[cntr_batch]
                    if flag_none[cntr_batch] < flag_fake:
                        DH_send = -10  # Sometimes don't put a D I lines there.
                        label_ID[cntr_batch] = 0
                        label_sh[cntr_batch] = 0
                    elif flag_none[cntr_batch] < 2*flag_fake:
                        # Sometimes don't put a H I lines there.
                        HI_send = yld_NHI[cntr_batch] - 10
                        DH_send = yld_DH[cntr_batch] + 10
                        label_ID[cntr_batch] = 0
                        label_sh[cntr_batch] = 0
                    model = utils.DH_model([HI_send, DH_send, zval, yld_dopp[cntr_batch], yld_temp[cntr_batch]],
                                           wave[imin:imax, qso], vfwhm)
                    # Determine the extra noise needed to maintain the same flue
                    exnse = np.random.normal(np.zeros(spec_len), flue[imin:imax, qso] * np.sqrt(1 - model ** 2))
                    # Add this noise to the data
                    X_batch[cntr_batch, :, 0] = flux[imin:imax, qso] * model + exnse
                    if debug:
                        plt.plot(wave[imin:imax, qso], X_batch[cntr_batch, :, 0],'r-', drawstyle='steps-mid')
                        plt.axvline(LyaD*(1+zval))
                else:
                    if flag_none[cntr_batch] < 2*flag_fake:
                        HI_send, DH_send = yld_NHI[cntr_batch], yld_DH[cntr_batch]
                        if flag_none[cntr_batch] < flag_fake:
                            DH_send = -10  # Sometimes don't put a D I lines there.
                        elif flag_none[cntr_batch] < 2 * flag_fake:
                            # Sometimes don't put a H I lines there.
                            HI_send = yld_NHI[cntr_batch] - 10
                            DH_send = yld_DH[cntr_batch] + 10
                        # Generate a feature that looks like just H or just D
                        model = utils.DH_model([HI_send, DH_send, zval, yld_dopp[cntr_batch], yld_temp[cntr_batch]], wave[imin:imax, qso], vfwhm)
                        # Determine the extra noise needed to maintain the same flue
                        exnse = np.random.normal(np.zeros(spec_len), flue[imin:imax, qso] * np.sqrt(1 - model ** 2))
                        # Add this noise to the data
                        X_batch[cntr_batch, :, 0] = flux[imin:imax, qso] * model + exnse
                    else:
                        X_batch[cntr_batch, :, 0] = flux[imin:imax, qso]
                if debug:
                    plt.title("{0:f} - {1:f}".format(label_ID[cntr_batch], label_sh[cntr_batch]))
                # Increment the counter
                cntr_batch += 1
        if debug:
            plt.show()
        indict['input_1'] = X_batch.copy()
        # Store output
        outdict = {'output_ID': label_ID,
                   'output_sh': label_sh}
        if not debug:
            # return (indict, outdict)
            yield (indict, outdict)


def build_model_simple(hyperpar):
    # Extract parameters
    fc1_neurons = hyperpar['fc1_neurons']
    fc2_ID_neurons = hyperpar['fc2_ID_neurons']
    fc2_sh_neurons = hyperpar['fc2_sh_neurons']
    conv1_kernel = hyperpar['conv_kernel_1']
    conv2_kernel = hyperpar['conv_kernel_2']
    conv3_kernel = hyperpar['conv_kernel_3']
    conv1_filter = hyperpar['conv_filter_1']
    conv2_filter = hyperpar['conv_filter_2']
    conv3_filter = hyperpar['conv_filter_3']
    conv1_stride = hyperpar['conv_stride_1']
    conv2_stride = hyperpar['conv_stride_2']
    conv3_stride = hyperpar['conv_stride_3']
    pool1_kernel = hyperpar['pool_kernel_1']
    pool2_kernel = hyperpar['pool_kernel_2']
    pool3_kernel = hyperpar['pool_kernel_3']
    pool1_stride = hyperpar['pool_stride_1']
    pool2_stride = hyperpar['pool_stride_2']
    pool3_stride = hyperpar['pool_stride_3']
    dropout_prob = hyperpar['dropout_prob']

    # Build model
    # Shape is (batches, steps, channels)
    # For example, a 3-color 1D image of side 100 pixels, dealt in batches of 32 would have a shape=(32,100,3)
    input_1 = Input(shape=(hyperpar['spec_len'], 1), name='input_1')
    conv1 = Conv1D(filters=conv1_filter, kernel_size=(conv1_kernel,), strides=(conv1_stride,), activation='relu')(input_1)
    pool1 = MaxPooling1D(pool_size=(pool1_kernel,), strides=(pool1_stride,))(conv1)
    conv2 = Conv1D(filters=conv2_filter, kernel_size=(conv2_kernel,), strides=(conv2_stride,), activation='relu')(pool1)
    pool2 = MaxPooling1D(pool_size=(pool2_kernel,), strides=(pool2_stride,))(conv2)
    conv3 = Conv1D(filters=conv3_filter, kernel_size=(conv3_kernel,), strides=(conv3_stride,), activation='relu')(pool2)
    pool3 = MaxPooling1D(pool_size=(pool3_kernel,), strides=(pool3_stride,))(conv3)
    drop1 = Dropout(dropout_prob)(pool3)
    flatlay = Flatten()(drop1)

    # Interpretation model
    regpen = hyperpar['l2_regpen']
    fullcon1 = Dense(fc1_neurons, activation='relu', kernel_regularizer=regularizers.l2(regpen))(flatlay)
    # Second fully connected layer
    fullcon2_ID = Dense(fc2_ID_neurons, activation='relu', kernel_regularizer=regularizers.l2(regpen))(fullcon1)
    drop2_ID = Dropout(dropout_prob)(fullcon2_ID)
    fullcon2_sh = Dense(fc2_sh_neurons, activation='relu', kernel_regularizer=regularizers.l2(regpen))(fullcon1)
    drop2_sh = Dropout(dropout_prob)(fullcon2_sh)
    # Output nodes
    output_ID = Dense(1, activation='sigmoid', name='output_ID')(drop2_ID)
    output_sh = Dense(1, activation='linear', name='output_sh')(drop2_sh)
    model = Model(inputs=[input_1], outputs=[output_ID, output_sh])
    return model


# fit and evaluate a model
def evaluate_model(allWave, allFlux, allFlue, allStat, allzem,
                   hyperpar, mnum, epochs=10, verbose=1):
    debug = False
    if debug:
        indict, outdict = yield_data_trueqso(allWave, allFlux, allFlue, allStat, allzem, hyperpar['batch_size'], hyperpar['spec_len'])
        X_batch = indict['input_1']
        output_ID, output_sh = outdict['output_ID'], outdict['output_sh']
        wavetmp = np.arange(X_batch.shape[1])
        for ff in range(output_ID.size):
            plt.subplot(output_ID.size,1,ff+1)
            plt.plot(wavetmp, X_batch[ff,:,0], 'k-', drawstyle='steps-mid')
            plt.axvline(hyperpar['spec_len']//2, color='b')
            plt.axvline(hyperpar['spec_len']//2+output_sh[ff], color='r')
            plt.title("{0:f} - {1:f}".format(output_ID[ff], output_sh[ff]))
        plt.show()
        assert(False)
    filepath = os.path.dirname(os.path.abspath(__file__))
    model_name = '/fit_data/model_{0:03d}'.format(mnum)
    ngpus = len(get_available_gpus())
    print("Number of GPUS = {0:d}".format(ngpus))
    # Construct network
    if ngpus > 1:
        model = build_model_simple(hyperpar)
        # Make this work on multiple GPUs
        gpumodel = multi_gpu_model(model, gpus=ngpus)
    else:
        gpumodel = build_model_simple(hyperpar)

    # Summarize layers
    print("Saving summary")
    summary = True
    if summary:
        with open(filepath + model_name + '.summary', 'w') as f:
            with redirect_stdout(f):
                gpumodel.summary()
    print("Summary complete")
    # Plot graph
    plotit = False
    if plotit:
        pngname = filepath + model_name + '.png'
        plot_model(gpumodel, to_file=pngname)
    # Compile
    print("Compiling")
    loss, loss_weights, optadam = custom_objects(hyperpar)
    gpumodel.compile(loss=loss, loss_weights=loss_weights, optimizer=optadam, metrics=['mean_squared_error'])
    print("Compiled")
    # Initialise callbacks
    ckp_name = filepath + model_name + '.hdf5'
    sav_name = filepath + model_name + '_save.hdf5'
    csv_name = filepath + model_name + '.log'
    print("Preparing checkpoints")
    checkpointer = ModelCheckpoint(filepath=ckp_name, verbose=1, save_best_only=True)
    csv_logger = CSVLogger(csv_name, append=True)
    # Fit network
    print("Begin Fit network")
    gpumodel.fit_generator(
        yield_data_trueqso(allWave, allFlux, allFlue, allStat, allzem, hyperpar['batch_size'], hyperpar['spec_len']),
        steps_per_epoch=hyperpar['num_batch_train'],  # Total number of batches (i.e. num data/batch size)
        epochs=epochs, verbose=verbose,
        callbacks=[checkpointer, csv_logger],
        validation_data=yield_data_trueqso(allWave, allFlux, allFlue, allStat, allzem, hyperpar['batch_size'], hyperpar['spec_len']),
        validation_steps=hyperpar['num_batch_validate'])
    print("Saving network")
    gpumodel.save(sav_name)

    # Evaluate model
#    _, accuracy
    print("Evaluating accuracy")
    accuracy = gpumodel.evaluate_generator(yield_data_trueqso(allWave, allFlux, allFlue, allStat, allzem, hyperpar['batch_size'], hyperpar['spec_len']),
                                           steps=allzem.shape[0],
                                           verbose=0)
    return accuracy, gpumodel.metrics_names


# summarize scores
def summarize_results(scores):
    keys = scores.keys()
    for ii in keys:
        m, s = np.mean(scores[ii]), np.std(scores[ii])
        print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# Detect features in a dataset
def localise_features(mnum, repeats=3):
    # Generate hyperparameters
    hyperpar = hyperparam_orig(0)
    #hyperpar = hyperparam(mnum)
    # load data
    allWave, allFlux, allFlue, allStat, allzem = load_dataset_trueqsos()
    # repeat experiment
    allscores = dict({})
    for r in range(repeats):
        scores, names = evaluate_model(allWave, allFlux, allFlue, allStat, allzem,
                                       hyperpar, mnum, epochs=hyperpar['num_epochs'], verbose=1)
        if r == 0:
            for name in names:
                allscores[name] = []
        for ii, name in enumerate(names):
            allscores[name].append(scores[ii] * 100.0)
            if '_acc' in name:
                print('%s >#%d: %.3f' % (name, r + 1, allscores[name][-1]))
            else:
                print('%s >#%d: %.3f' % (name, r + 1, scores[ii]))
    # Summarize results
    summarize_results(allscores)

if __name__ == "__main__":
    # Run the code...
    gendata = False
    pltrange = False
    if gendata:
        # Generate data
        spec_len = 200  # This just needs to be approximate, and ideally larger than the final optimised value
        generate_dataset_trueqsos(spec_len)
    elif pltrange:
        wavein = np.linspace(1214.5,1216.5,100)
        nNHI = 5
        nwid = 5
        NHvals = np.linspace(NHI_min, NHI_max, nNHI)
        wdvals = np.linspace(temp_min, temp_max, nwid)
        cnt=1
        for nn in range(nNHI):
            for ww in range(nwid):
                par = [NHvals[nn], -4.6, 0.0, 0.0, wdvals[ww]]
                model = utils.DH_model(par, wavein, 7.0)
                plt.subplot(nNHI, nwid, cnt)
                plt.plot(wavein, model, 'k-')
                cnt += 1
        plt.show()
    else:
        # Once the data exist, run the experiment
        m_init = 0
        mnum = m_init
        localise_features(mnum, repeats=1)
        # while True:
        #     try:
        #         localise_features(mnum, repeats=1)
        #     except ValueError:
        #         continue
        #     mnum += 1
        #     if mnum >= m_init+1000:
        #         break
