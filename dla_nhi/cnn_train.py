import os
import pickle
import numpy as np
from astropy.table import Table
import astropy.io.fits as fits
from utils import voigt

import tensorflow as tf
from tensorflow.python.client import device_lib
import keras.backend as K
from keras.utils import plot_model, multi_gpu_model
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout, Flatten
from keras import regularizers
from contextlib import redirect_stdout

velstep = 2.5    # Pixel size in km/s
spec_len = 4096  # Number of pixels to use
zdla_min, zdla_max = 2.5, 3.4
NHI_min, NHI_max = 19.2, 21.0
restwin = 0.5*spec_len*velstep*1215.6701/299792.458  # Rest window in angstroms (the full window size is twice this)


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
    allowed_hpars = dict(learning_rate      = [0.001],
                         lr_decay           = [0.0],
                         l2_regpen          = [0.0],
                         dropout_prob       = [0.0],
                         num_epochs         = [100],
                         batch_size         = [200],
                         num_batch_train    = [128],
                         num_batch_validate = [32],
                         # Number of filters in each convolutional layer
                         conv_filter_1 = [128],
                         conv_filter_2 = [128],
                         conv_filter_3 = [128],
                         # Kernel size
                         conv_kernel_1 = [32],
                         conv_kernel_2 = [32],
                         conv_kernel_3 = [32],
                         # Stride of each kernal
                         conv_stride_1 = [1],
                         conv_stride_2 = [1],
                         conv_stride_3 = [1],
                         # Pooling kernel size
                         pool_kernel_1 = [2],
                         pool_kernel_2 = [2],
                         pool_kernel_3 = [2],
                         # Pooling stride
                         pool_stride_1 = [1],
                         pool_stride_2 = [1],
                         pool_stride_3 = [1],
                         # Fully connected layers
                         fc1_neurons   = [4096],
                         fc2_N_neurons = [256]
                         )
    # Generate dictionary of values
    hyperpar = dict({})
    for key in allowed_hpars.keys():
        hyperpar[key] = np.random.choice(allowed_hpars[key])
    # Save these parameters and return the hyperpar
    save_obj(hyperpar, 'fit_data/model_{0:03d}'.format(mnum))
    return hyperpar


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
                         fc2_N_neurons = [32, 64, 128, 256],
                         )
    # Generate dictionary of values
    hyperpar = dict({})
    for key in allowed_hpars.keys():
        hyperpar[key] = np.random.choice(allowed_hpars[key])
    # Save these parameters and return the hyperpar
    save_obj(hyperpar, 'fit_data/model_{0:03d}'.format(mnum))
    return hyperpar


def generate_dataset(rest_window=30.0):
    """
    rest_window = Number of REST Angstroms to the left and right of the central DLA profile to use - this is assumed to be an int, for file naming purposes
    rest_proxqso = number of REST Angstroms to the left of the QSO to use for generating a DLA (only used if non-zero)
    """
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
        wvmax = (1+zem)*(1215.6701+rest_window)  # Data will not be used to the right of the QSO Lya emission line + rest+window (the rest_window is to include the DLA profile)
        wvmin = (1+zdla_min)*(1215.6701-rest_window)  # Now data are needed below this DLA cutoff redshift... minus the rest window
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
    for qq in range(nqso):
        qso = t_trim[qq]
        zem = qso['zem_Adopt']
        allzem[qq] = zem
        # Load the data
        dat = fits.open("../data/{0:s}.fits".format(qso['Name_Adopt']))
        wvmax = (1+zem)*(1215.6701+rest_window)  # Data will not be used to the right of the QSO Lya emission line + rest+window (the rest_window is to include the DLA profile)
        wvmin = (1+zdla_min)*(1215.6701-rest_window)  # Now data are needed below this DLA cutoff redshift... minus the rest window
        wave = dat[1].data['WAVE']
        ww = np.where((wave > wvmin) & (wave < wvmax))
        sz = wave[ww].size
        cont = dat[1].data['CONTINUUM'][ww]
        allWave[:sz, qq] = wave[ww].copy()
        allFlux[:sz, qq] = dat[1].data['FLUX'][ww] * cont
        allFlue[:sz, qq] = dat[1].data['ERR'][ww] * cont
        allStat[:sz, qq] = dat[1].data['STATUS'][ww]
    # Save the data
    np.save("../data/train_data/wave_{0:.2f}.npy".format(rest_window), allWave)
    np.save("../data/train_data/flux_{0:.2f}.npy".format(rest_window), allFlux)
    np.save("../data/train_data/flue_{0:.2f}.npy".format(rest_window), allFlue)
    np.save("../data/train_data/stat_{0:.2f}.npy".format(rest_window), allStat)
    np.save("../data/train_data/zem_{0:.2f}.npy".format(rest_window), allzem)
    print("Data generated successfully")
    return


def load_dataset(rest_window=30.0, ftrain=0.9):
    allWave = np.load("../data/train_data/wave_{0:.2f}.npy".format(rest_window))
    allFlux = np.load("../data/train_data/flux_{0:.2f}.npy".format(rest_window))
    allFlue = np.load("../data/train_data/flue_{0:.2f}.npy".format(rest_window))
    allStat = np.load("../data/train_data/stat_{0:.2f}.npy".format(rest_window))
    allzem = np.load("../data/train_data/zem_{0:.2f}.npy".format(rest_window))
    ntrain = int(ftrain*allzem.shape[0])
    # Select the training data
    trainW = allWave[:, :ntrain]
    trainF = allFlux[:, :ntrain]
    trainE = allFlue[:, :ntrain]
    trainS = allStat[:, :ntrain]
    trainZ = allzem[:ntrain]
    # Select the test data
    testW = allWave[:, ntrain:]
    testF = allFlux[:, ntrain:]
    testE = allFlue[:, ntrain:]
    testS = allStat[:, ntrain:]
    testZ = allzem[:ntrain]
    return trainW, trainF, trainE, trainS, trainZ, testW, testF, testE, testS, testZ


def yield_data(wave, flux, flue, stat, zem, batch_sz):
    cntr_batch = 0
    while True:
        indict = ({})
        toobad = 0
        qso = cntr_batch
        while True:
            zdmin = max(zdla_min, ((wave[0, qso]+restwin)/1215.6701) - 1.0)  # Can't have a DLA below the data for this QSO
            zdmax = min(zdla_max, zem[qso])  # Can't have a DLA above the QSO redshift
            dla = np.random.uniform(zdmin, zdmax)
            amin = np.argmin(np.abs(wave[:, qso] - 1215.6701 * (1 + dla)))
            imin = amin - spec_len // 2
            imax = amin - spec_len // 2 + spec_len
            bd = np.where(stat[imin:imax, qso] != 1)
            if bd[0].size == 0:
                break
            elif toobad > 10:
                # Can't find anything in this QSO, move onto the next one, reset the toobad counter
                qso += 1
                cntr_batch += 1
                if cntr_batch >= zem.shape[0]:
                    qso = 0
                    cntr_batch = 0
                toobad = 0
            else:
                toobad += 1
        # We've found a good system, now extract the data
        X_batch = np.zeros((batch_sz, spec_len))
        yld_NHI = np.random.uniform(NHI_min, NHI_max, batch_sz)
        for mm in range(batch_sz):
            model = voigt([yld_NHI[mm], dla, 15.0], wave[imin:imax, qso])
            # Determine the extra noise needed to maintain the same flue
            exnse = np.random.normal(np.zeros(spec_len), flue[imin:imax, qso] * np.sqrt(1 - model**2))
            # Add this noise to the data
            X_batch[mm, :] = flux[imin:imax, qso]*model + exnse
        indict['input_1'] = X_batch.copy()
        # Store output
        outdict = {'output_NHI': yld_NHI}
        yield (indict, outdict)

        cntr_batch += 1
        if cntr_batch >= zem.shape[0]:
            cntr_batch = 0


def build_model_simple(hyperpar):
    # Extract parameters
    fc1_neurons = hyperpar['fc1_neurons']
    fc2_N_neurons = hyperpar['fc2_N_neurons']
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

    # Build model
    # Shape is (batches, steps, channels)
    # For example, a 3-color 1D image of side 100 pixels, dealt in batches of 32 would have a shape=(32,100,3)
    input_1 = Input(shape=(spec_len, 1), name='input_1')
    conv1 = Conv1D(filters=conv1_filter, kernel_size=(conv1_kernel,), strides=(conv1_stride,), activation='relu')(input_1)
    pool1 = MaxPooling1D(pool_size=(pool1_kernel,), strides=(pool1_stride,))(conv1)
    conv2 = Conv1D(filters=conv2_filter, kernel_size=(conv2_kernel,), strides=(conv2_stride,), activation='relu')(pool1)
    pool2 = MaxPooling1D(pool_size=(pool2_kernel,), strides=(pool2_stride,))(conv2)
    conv3 = Conv1D(filters=conv3_filter, kernel_size=(conv3_kernel,), strides=(conv3_stride,), activation='relu')(pool2)
    pool3 = MaxPooling1D(pool_size=(pool3_kernel,), strides=(pool3_stride,))(conv3)
    flatlay = Flatten()(pool3)

    # Interpretation model
    regpen = hyperpar['l2_regpen']
    fullcon1 = Dense(fc1_neurons, activation='relu', kernel_regularizer=regularizers.l2(regpen))(flatlay)
    drop1 = Dropout(hyperpar['dropout_prob'])(fullcon1)
    # Second fully connected layer
    fullcon2_N = Dense(fc2_N_neurons, activation='relu', kernel_regularizer=regularizers.l2(regpen))(drop1)
    drop2_N = Dropout(hyperpar['dropout_prob'])(fullcon2_N)
    output_N = Dense(1, activation='linear', name='output_NHI')(drop2_N)
    model = Model(inputs=[input_1], outputs=[output_N])
    return model


# fit and evaluate a model
def evaluate_model(trainW, trainF, trainE, trainS, trainZ, testW, testF, testE, testS, testZ,
                   hyperpar, mnum, epochs=10, verbose=1):
    #yield_data(trainX, trainN, trainb)
    #assert(False)
    filepath = os.path.dirname(os.path.abspath(__file__))
    model_name = '/fit_data/model_{0:03d}'.format(mnum)
    ngpus = len(get_available_gpus())
    # Construct network
    if ngpus > 1:
        model = build_model_simple(hyperpar)
        # Make this work on multiple GPUs
        gpumodel = multi_gpu_model(model, gpus=ngpus)
    else:
        gpumodel = build_model_simple(hyperpar)
    # else:
    #     inputs = []
    #     inputs.append(Input(shape=(spec_len, 1), name='input_1'))
    #     conv1 = Conv1D(filters=128, kernel_size=16, activation='relu')(inputs[0])
    #     #    pool1 = MaxPooling1D(pool_size=(pool1_kernel,), strides=(pool1_stride,))(conv1)
    #     pool1 = MaxPooling1D(pool_size=2)(conv1)
    #     conv2 = Conv1D(filters=128, kernel_size=16, activation='relu')(pool1)
    #     #    pool2 = MaxPooling1D(pool_size=(pool2_kernel,), strides=(pool2_stride,))(conv2)
    #     pool2 = MaxPooling1D(pool_size=2)(conv2)
    #     conv3 = Conv1D(filters=128, kernel_size=16, activation='relu')(pool2)
    #     pool3 = MaxPooling1D(pool_size=2)(conv3)
    #     flat = Flatten()(pool3)
    #
    #     # Interpretation model
    #     fullcon1 = Dense(4096, activation='relu')(flat)
    #     output_N = Dense(1, activation='linear', name='output_N')(fullcon1)
    #     output_z = Dense(1, activation='linear', name='output_z')(fullcon1)
    #     output_b = Dense(1, activation='linear', name='output_b')(fullcon1)
    #     model = Model(inputs=inputs, outputs=[output_N, output_z, output_b])
    #     gpumodel = multi_gpu_model(model, gpus=ngpus)

    # Summarize layers
    summary = False
    if summary:
        with open(filepath + model_name + '.summary', 'w') as f:
            with redirect_stdout(f):
                model.summary()
    # Plot graph
    plotit = False
    if plotit:
        pngname = filepath + model_name + '.png'
        plot_model(model, to_file=pngname)
    # Compile
    loss = {'output_NHI': 'mse'}
    decay = hyperpar['lr_decay']*hyperpar['learning_rate']/hyperpar['num_epochs']
    optadam = Adam(lr=hyperpar['learning_rate'], decay=decay)
    gpumodel.compile(loss=loss, optimizer=optadam, metrics=['mean_squared_error'])
    # Initialise callbacks
    ckp_name = filepath + model_name + '.hdf5'
    sav_name = filepath + model_name + '_save.hdf5'
    csv_name = filepath + model_name + '.log'
    checkpointer = ModelCheckpoint(filepath=ckp_name, verbose=1, save_best_only=True)
    csv_logger = CSVLogger(csv_name, append=True)
    # Fit network
    gpumodel.fit_generator(
        yield_data(trainW, trainF, trainE, trainS, trainZ, hyperpar['batch_size']),
        steps_per_epoch=hyperpar['num_batch_train'],  # Total number of batches (i.e. num data/batch size)
        epochs=epochs, verbose=verbose,
        callbacks=[checkpointer, csv_logger],
        validation_data=yield_data(testW, testF, testE, testS, testZ, hyperpar['batch_size']),
        validation_steps=hyperpar['num_batch_validate'])

    gpumodel.save(sav_name)

    # Evaluate model
#    _, accuracy
    accuracy = gpumodel.evaluate_generator(yield_data(testW, testF, testE, testS, testZ, hyperpar['batch_size']),
                                           steps=testZ.shape[0],
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
    trainW, trainF, trainE, trainS, trainZ, testW, testF, testE, testS, testZ = load_dataset(rest_window=restwin)
    # repeat experiment
    allscores = dict({})
    for r in range(repeats):
        scores, names = evaluate_model(trainW, trainF, trainE, trainS, trainZ, testW, testF, testE, testS, testZ,
                                       hyperpar, mnum, epochs=hyperpar['num_epochs'], verbose=2)
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


# Run the code...
gendata = False
if gendata :
    # Generate data
    generate_dataset(rest_window=restwin)
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
