#############################################################################
# -- Edited based on Ryan Cooke's code. --
# Tuning hyper-parameters using Bayesian Optimisation for CNN_Lya_single.py.
#############################################################################

import os
import time
import numpy as np
import pandas as pd
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

import matplotlib as mpl
mpl.use('Agg') # to use matplotlib without visualisation envs
from GPyOpt.methods import BayesianOptimization
from cnn_train import load_dataset_trueqsos, yield_data_trueqso, mse_mask, custom_objects, save_obj


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


def train_models(list_par):
    hyperpar = {'spec_len': int(list_par[0][0]), #Categorical([37,75,111,147,185,221,259,295]), # 27 - 295 pixels ~ 1.00 - 11.00 A
                ## Hyper-parameters for machine architecture ##
                'learning_rate': 0.0001,
                'lr_decay': 0.0,
                'l2_regpen': list_par[0][1], #default=0.01
                'dropout_prob': list_par[0][2],
                'batch_size': int(list_par[0][3]),
                'num_epochs': 5,
                'num_batch_train': 64,
                'num_batch_validate': 64,
                # Number of filters in each convolutional layer
                'conv_filter_1': int(list_par[0][4]),
                'conv_filter_2': int(list_par[0][5]),
                'conv_filter_3': int(list_par[0][6]),
                # Kernel size
                'conv_kernel_1': int(list_par[0][7]),
                'conv_kernel_2': int(list_par[0][8]),
                'conv_kernel_3': int(list_par[0][9]),
                # Stride size
                'conv_stride_1': 1,
                'conv_stride_2': 1,
                'conv_stride_3': 1,
                # Pooling kernel size
                'pool_kernel_1': 2,
                'pool_kernel_2': 2,
                'pool_kernel_3': 2,
                # Pooling stride
                'pool_stride_1': 2,
                'pool_stride_2': 2,
                'pool_stride_3': 2,
                # Fully connected layers
                'fc1_neurons': int(list_par[0][10]),
                'fc2_ID_neurons': int(list_par[0][11]),
                'fc2_sh_neurons': int(list_par[0][12]),
                ## Hyper-parameters for training condition ##
                # loss_weights
                'ID_loss': 1.0,
                'sh_loss': 1.0
                }
    mdsavename =  'lr%.4f_dp%.2f_cv1f%d_cv2f%d_cv3f%d_cv1k%d_cv2k%d_cv3k%d_fc1%d_fc2ID%d_fc2SH%d_lwID%.4f_lwSH%.4f_speclen%d_batchsize%d' \
                  %(hyperpar['learning_rate'], hyperpar['dropout_prob'], hyperpar['conv_filter_1'], hyperpar['conv_filter_2'], hyperpar['conv_filter_3'], \
                    hyperpar['conv_kernel_1'], hyperpar['conv_kernel_2'], hyperpar['conv_kernel_3'], hyperpar['fc1_neurons'], \
                    hyperpar['fc2_ID_neurons'], hyperpar['fc2_sh_neurons'],
                    hyperpar['ID_loss'], hyperpar['sh_loss'], hyperpar['spec_len'], hyperpar['batch_size'])
    # save the hyperparameters used in the most optimised model, i.e. last model in optimisation
    save_obj(hyperpar, params_save)
    # build training model
    gpumodel = build_model_simple(hyperpar)
    loss, loss_weights, optadam = custom_objects(hyperpar)
    gpumodel.compile(loss=loss, loss_weights=loss_weights, optimizer=optadam)#, metrics=['accuracy']) # add customised loss weight assignment
    # Initialise callbacks
    model_name = mdsavepath + mdsavename
    ckp_name = model_name + '_min_val_loss.hdf5'
    csv_name = model_name + '.log'
    # Best model is defined as the model with minimal validation loss
    checkpointer = ModelCheckpoint(filepath=ckp_name, verbose=1, save_best_only=True, monitor='val_loss', mode='min')
    csv_logger = CSVLogger(csv_name, append=True)
    # Fit network
    h = gpumodel.fit(
        yield_data_trueqso(allWave, allFlux, allFlue, allStat, allzem, hyperpar['batch_size'], hyperpar['spec_len']),
                    steps_per_epoch=hyperpar['num_batch_train'],
                    epochs=hyperpar['num_epochs'], verbose=0,
                    callbacks=[checkpointer, csv_logger],
                    validation_data= yield_data_trueqso(allWave, allFlux, allFlue, allStat, allzem, hyperpar['batch_size'], hyperpar['spec_len']),
                    validation_steps=hyperpar['num_batch_validate'])
    validation_loss = np.min(h.history['val_loss'])
    list_validation_loss.append(validation_loss)
    list_saved_model_name.append(model_name)
    return validation_loss

if __name__ == "__main__":
    '''
    Optimising the hyper-parameters used in single fitting.
    '''
    # optimisation setting
    max_iter = 40 # number of search tests

    # file paths
    parsavepath = 'archi_optimisation/cnn_fits/' # path to CNN model
    if not os.path.exists(parsavepath):
        os.mkdir(parsavepath)

    num_epochs = 5
    mnum = 0
    random_state = 0 # random_state=None, to have randomly draw samples for each training (for repeated tests) # random_state=0, to fix the shuffle.

    ##### MAIN #####
    start = time.time()
    # load data
    allWave, allFlux, allFlue, allStat, allzem = load_dataset_trueqsos()

    # Store hyperparameters
    parsavename = 'mnum%d' \
                   %(mnum)
    params_save = parsavepath + parsavename # for hyperparams saving (Storing the most optimised model, i.e. the last model in optimisation)
    mdsavepath =  parsavepath + parsavename + '/' # save all optimisation model with minimal loss in training epochs
    if not os.path.exists(mdsavepath):
        os.mkdir(mdsavepath)
    ## OPTIMISATION TRAINING ##
    list_validation_loss, list_saved_model_name = [], []
    # Define all of the allowed parameter space
    list_par = [{'name': 'spec_len', 'type': 'discrete', 'domain': (75,89,103,117,131,145,159,173,187,201,215,229,243,257,271,285,299,313)}, # After the primary test, use the interval of ~0.52A from a size of 7.0A (187) to 11.1A (299)
                #{'name': 'learning_rate', 'type': 'discrete', 'domain': (0.0001,0.0005)},
                {'name': 'l2_regpen', 'type': 'discrete', 'domain': (0.,0.01)},
                {'name': 'dropout_prob', 'type': 'discrete', 'domain': (0.0,0.1,0.2,0.3,0.4,0.5)},
                {'name': 'batch_size', 'type': 'discrete', 'domain': (32,64,128,512,1024)},
                {'name': 'conv_filter_1', 'type': 'discrete', 'domain': (64,128,256,512)},
                {'name': 'conv_filter_2', 'type': 'discrete', 'domain': (64,128,256,512)},
                {'name': 'conv_filter_3', 'type': 'discrete', 'domain': (64,128,256,512)},
                {'name': 'conv_kernel_1', 'type': 'discrete', 'domain': (4,6,8,10,12,14)},
                {'name': 'conv_kernel_2', 'type': 'discrete', 'domain': (3,5,7,9,11)},
                {'name': 'conv_kernel_3', 'type': 'discrete', 'domain': (2,4,6,8)},
                {'name': 'fc1_neurons', 'type': 'discrete', 'domain': (32,64,128,256)},
                {'name': 'fc2_ID_neurons', 'type': 'discrete', 'domain': (32,64,128,256)},
                {'name': 'fc2_sh_neurons', 'type': 'discrete', 'domain': (32,64,128,256)}]
    #hyperpar = hyperparam(list_par)
    optSearch = BayesianOptimization(f=train_models, domain=list_par, model_type='GP')
    rf = parsavepath + parsavename + '.report'
    ef = parsavepath + parsavename + '.evaluation'
    mf = parsavepath + parsavename + '.models'
    optSearch.run_optimization(max_iter=max_iter, report_file=rf, evaluations_file=ef, models_file=mf, verbosity=True)
    optSearch.plot_acquisition(filename=parsavepath+parsavename+'_acquisition.png')
    optSearch.plot_convergence(filename=parsavepath+parsavename+'_convergence.png')
    header_params = []
    for param in list_par:
        header_params.append(param['name'])
    df_results = pd.DataFrame(data=optSearch.X, columns=header_params)
    df_results['validation_loss'] = optSearch.Y
    df_results['validation loss check'] = list_validation_loss
    df_results['model_name'] = list_saved_model_name
    df_results.to_pickle(parsavepath + 'df_results.pkl')
    df_results.to_csv(parsavepath + 'df_results.csv')
    print('Run Time: %.f mins' %( (time.time()-start)/60. ))
