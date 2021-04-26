import os
import pickle
import numpy as np
from astropy.table import Table
import astropy.io.fits as fits
# from pyigm.fN.fnmodel import FNModel
# from pyigm.fN.mockforest import monte_HIcomp
from scipy.special import wofz

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
spec_len = 256
spec_ext = 64


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
                         num_epochs         = [10],
                         batch_size         = [1000],
                         num_batch_train    = [200],
                         num_batch_validate = [20],
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
                         pool_stride_1 = [1, 2, 3],
                         pool_stride_2 = [1, 2, 3],
                         pool_stride_3 = [1, 2, 3],
                         # Fully connected layers
                         fc1_neurons   = [4096],
                         fc2_N_neurons = [32, 64, 128, 256],
                         fc2_z_neurons = [32, 64, 128, 256],
                         fc2_b_neurons = [32, 64, 128, 256],
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
                         fc2_z_neurons = [32, 64, 128, 256],
                         fc2_b_neurons = [32, 64, 128, 256],
                         )
    # Generate dictionary of values
    hyperpar = dict({})
    for key in allowed_hpars.keys():
        hyperpar[key] = np.random.choice(allowed_hpars[key])
    # Save these parameters and return the hyperpar
    save_obj(hyperpar, 'fit_data/model_{0:03d}'.format(mnum))
    return hyperpar


def voigt(wave, params):
    p0, p1, p2 = params
    lam, fvl, gam = 1215.6701, 0.4164, 6.265E8
    cold = 10.0**p0
    zp1=p1+1.0
    wv=lam*1.0e-8
    bl=p2*wv/2.99792458E5
    a=gam*wv*wv/(3.76730313461770655E11*bl)
    cns=wv*wv*fvl/(bl*2.002134602291006E12)
    cne=cold*cns
    ww=(wave*1.0e-8)/zp1
    v=wv*ww*((1.0/ww)-(1.0/wv))/bl
    tau = cne*wofz(v + 1j * a).real
    return np.exp(-1.0*tau)


def generate_dataset(rest_window=30):
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
        wvmin = (1+2.5)*(1215.6701-rest_window)  # Now data are needed below this DLA cutoff redshift... minus the rest window
        wave = dat[1].data['WAVE']
        ww = np.where((wave > wvmin) & (wave < wvmax))
        sz = wave[ww].size
        if sz > maxsz: maxsz = sz
        stat = dat[1].data['STATUS'][ww]
        bd = np.where(stat != 1)
        if bd[0].size != 0:
            print("Number of bad pixels in QSO {0:d} = {1:d}".format(qq, bd[0].size))

    # Generate the data arrays and insert the data
    allWave = np.zeros((maxsz, nqso))
    allFlux = np.zeros((maxsz, nqso))
    allFlue = np.zeros((maxsz, nqso))
    allzem  = np.zeros(nqso)
    for qq in range(nqso):
        qso = t_trim[qq]
        zem = qso['zem_Adopt']
        allzem[qq] = zem
        # Load the data
        dat = fits.open("../data/{0:s}.fits".format(qso['Name_Adopt']))
        wvmax = (1+zem)*(1215.6701+rest_window)  # Data will not be used to the right of the QSO Lya emission line + rest+window (the rest_window is to include the DLA profile)
        wvmin = (1+2.5)*(1215.6701-rest_window)  # Now data are needed below this DLA cutoff redshift... minus the rest window
        wave = dat[1].data['WAVE']
        ww = np.where((wave > wvmin) & (wave < wvmax))
        sz = wave[ww].size
        cont = dat[1].data['CONTINUUM'][ww]
        allWave[:sz,qq] = wave[ww].copy()
        allFlux[:sz,qq] = dat[1].data['FLUX'][ww] * cont
        allFlue[:sz,qq] = dat[1].data['ERR'][ww] * cont
    # Save the data
    np.save("../data/train_data/wave_{0:d}.npy".format(rest_window), allWave)
    np.save("../data/train_data/flux_{0:d}.npy".format(rest_window), allFlux)
    np.save("../data/train_data/flue_{0:d}.npy".format(rest_window), allFlue)
    np.save("../data/train_data/zem_{0:d}.npy".format(rest_window), allzem)
    return


def load_dataset(zem=3.0, ftrain=0.9):
    zstr = "zem{0:.2f}".format(zem)
    fdata = np.load("train_data/svoigt_prof_zem3_nsubpix10_numspec1000.npy").T
    Nlabel = np.load("train_data/svoigt_Nval_zem3_nsubpix10_numspec1000.npy")
    blabel = np.load("train_data/svoigt_bval_zem3_nsubpix10_numspec1000.npy")
    zlabel = ((spec_len+spec_ext)//2)*np.ones(Nlabel.size)
    ntrain = int(ftrain*fdata.shape[0])
    # Select the training data
    trainX = fdata[:ntrain, :]
    trainN = Nlabel[:ntrain]
    trainb = blabel[:ntrain]
    # Select the test data
    testX = fdata[ntrain:, :]
    testN = Nlabel[ntrain:]
    testb = blabel[ntrain:]
    return trainX, testX


def yield_data(data, Nlabels, batch_sz):
    cntr_batch = 0
    cenpix = (spec_len+spec_ext)//2
    ll = np.arange(batch_sz).repeat(spec_len)
    while True:
        indict = ({})
        pertrb = np.random.randint(0, spec_ext, batch_sz)
        pp = pertrb.reshape((batch_sz, 1)).repeat(spec_len, axis=1) + np.arange(spec_len)
        X_batch = data[ll+cntr_batch, pp.flatten()].reshape((batch_sz, spec_len, 1))
        indict['input_1'] = X_batch.copy()
        z_batch = spec_len//2 - cenpix + pertrb.copy()
        # Extract the relevant bits of information
        yld_NHI = Nlabels[cntr_batch:cntr_batch+batch_sz]
        # Store output
        outdict = {'output_N': yld_NHI}

#        pdb.set_trace()
        yield (indict, outdict)

        cntr_batch += batch_sz
        if cntr_batch >= data.shape[0]-batch_sz:
            cntr_batch = 0


def build_model_simple(hyperpar):
    # Extract parameters
    fc1_neurons = hyperpar['fc1_neurons']
    fc2_N_neurons = hyperpar['fc2_N_neurons']
    fc2_b_neurons = hyperpar['fc2_b_neurons']
    fc2_z_neurons = hyperpar['fc2_z_neurons']
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
    fullcon2_z = Dense(fc2_z_neurons, activation='relu', kernel_regularizer=regularizers.l2(regpen))(drop1)
    drop2_z = Dropout(hyperpar['dropout_prob'])(fullcon2_z)
    fullcon2_b = Dense(fc2_b_neurons, activation='relu', kernel_regularizer=regularizers.l2(regpen))(drop1)
    drop2_b = Dropout(hyperpar['dropout_prob'])(fullcon2_b)
    output_N = Dense(1, activation='linear', name='output_N')(drop2_N)
    output_z = Dense(1, activation='linear', name='output_z')(drop2_z)
    output_b = Dense(1, activation='linear', name='output_b')(drop2_b)
    model = Model(inputs=[input_1], outputs=[output_N, output_z, output_b])
    return model


# fit and evaluate a model
def evaluate_model(trainX, testX,
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
        yield_data(trainX, trainNHI, hyperpar['batch_size']),
        steps_per_epoch=hyperpar['num_batch_train'],  # Total number of batches (i.e. num data/batch size)
        epochs=epochs, verbose=verbose,
        callbacks=[checkpointer, csv_logger],
        validation_data=yield_data(testX, testNHI, hyperpar['batch_size']),
        validation_steps=hyperpar['num_batch_validate'])

    gpumodel.save(sav_name)

    # Evaluate model
#    _, accuracy
    accuracy = gpumodel.evaluate_generator(yield_data(testX, testNHI, hyperpar['batch_size']),
                                           steps=testX.shape[0],
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
    hyperpar = hyperparam(mnum)
    # load data
    trainX, testX = load_dataset()
    # repeat experiment
    allscores = dict({})
    for r in range(repeats):
        scores, names = evaluate_model(trainX, testX,
                                       hyperpar, mnum, epochs=hyperpar['num_epochs'])
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
if True:
    # Generate data
    generate_dataset()
else:
    # Once the data exist, run the experiment
    m_init = 0
    mnum = m_init
    while True:
        try:
            localise_features(mnum, repeats=1, restart=False)
        except ValueError:
            continue
        mnum += 1
        if mnum >= m_init+1000:
            break
