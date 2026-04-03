''' Standard Imports '''
import os
import sys
import math
import random
from datetime import datetime
import time

''' Special Imports '''
import numpy as np
np.set_printoptions( precision = 5 , threshold = 50 , edgeitems = 5 , linewidth = 150 , floatmode = 'fixed' )

import tensorflow as tf

# Clear custom imports in this layer only.
tf.keras.utils.get_custom_objects().clear()

#tf.autograph.set_verbosity(0, alsologtostdout=False)

''' Local Imports'''

import proj_utils as utils          # Print, Plot, & Helper functions 
from trainer import ModelTrainer    # Model trainer.

# Data Imports
from data import MackeyGlassGenerator , CopyMemoryGenerator , psMNISTGenerator

## Model imports for standard RNN arch. ##
from models import mkygls_rnn_callback , mkygls_lstm_callback , mkygls_gru_callback
from models import psMNIST_rnn_callback , psMNIST_lstm_callback , psMNIST_gru_callback
from models import cpymem_rnn_callback , cpymem_lstm_callback , cpymem_gru_callback

## Model imports for MY BASE RNN arch. ##
from models import mkygls_hopf_base_callback , psMNIST_hopf_base_callback , cpymem_hopf_base_callback
from models import mkygls_hopf_radial_callback , psMNIST_hopf_radial_callback , cpymem_hopf_radial_callback
from models import mkygls_hopf_theta_callback , psMNIST_hopf_theta_callback , cpymem_hopf_theta_callback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR )

def mkygls_experiments( window_size = 32 , callback_list = [ mkygls_hopf_theta_callback ] ):
    
    ''' ----- INPUT FEATURE PARAMETERS FOR MODEL ----- '''
    tf.keras.backend.clear_session()
    
    # Model params
    input_size = window_size
    output_size = input_size
    
    # Train params #
#    '''
    batch_size = 16
    epoch_size = 16
    num_epochs = 40
    '''
    batch_size = 50
    epoch_size = 25
    num_epochs = 50
#    '''
    
    step_size = output_size
    test_split = 0.25       # as a percent(%) of training parameters
    valid_split = 0.0
    verbose = 1
    
    # Dtype for data generation #
    data_dtype = np.float32
    wndo_str = str( window_size )
    
    ''' TRAIN MODEL '''
    trainer = ModelTrainer(
        data_callback = MackeyGlassGenerator,
        batch_size = batch_size,
        epoch_size = epoch_size,
        num_epochs = num_epochs,
        input_size = input_size,
        output_size = output_size,
        step_size = step_size,
        test_split = test_split,
        makedataset = False,
        dtype = data_dtype,
        name = 'mkygls' + '\\window_size_' + wndo_str
    )
    
    '''
    callback_list = [
        mkygls_hopf_theta_callback,
#        mkygls_hopf_radial_callback,
#        mkygls_hopf_base_callback,
#        mkygls_rnn_callback,
#        mkygls_lstm_callback,
#        mkygls_gru_callback
    ]
#    '''
    
    start_time = time.time()
    
#    try:
    chkpnt_str = 'C:\\Users\\isuan\\Desktop\\Research\\training_runs'
    chkpnt_str += '\\mkygls\\window_size_32\\hopf_theta\\gen_4\\training\\checkpoints'
    
    # Starts training for callback_list models #
    trainer.execute(
        model_callbacks = callback_list,
        fit_valid_split = valid_split,
        do_prediction = True,               # True False
        do_evaluation = True,
        verbose = verbose,
        testing = None,
        load_chkpnt = None
    )
    
    rt = time.time() - start_time
    rt_hrs = rt // 3600
    rt_min = ( rt - 3600 * rt_hrs ) // 60
    rt_sec = ( rt - 3600 * rt_hrs - 60 * rt_min )
    rt_rem = math.fmod( rt_sec , 1.0e2 ) - math.floor( rt_sec )
    rt_sec = math.floor( rt_sec )
    print('Runtime: {:.1f} (hrs)  {:.1f} (min)  {:.1f} (sec)  {:.2f} (0.01 sec)\n'.format( rt_hrs , rt_min , rt_sec , rt_rem ))
    
#    '''
    # Save all runs for callback_list models
    trainer.save(
        max_plot_len = 2048,
        show_plot = False,           # True False
        save_plot = True,
        show_weights = False,
        save_weights = True,
        verbose = verbose
    )
    
#   except:
#       print( '\nTEST FAILED: for Mackey-Glass\n'*3 + '\n'*3 )
    
    tf.keras.backend.clear_session()


def psmnist_experiments( callback_list = [ psMNIST_hopf_theta_callback ] ):
    
    ''' ----- INPUT FEATURE PARAMETERS FOR MODEL ----- '''
    tf.keras.backend.clear_session()
    
    # Model params
    input_size = 784
    output_size = 1
    
    # Train params #
    batch_size = 4
    epoch_size = None
    num_epochs = 4
    
    step_size = 1
    test_split = 0.25       # as a percent(%) of training parameters
    valid_split = 0.0
    verbose = 1
    
    # Dtype for data generation #
    data_dtype = np.float32
    
    ''' TRAIN MODEL '''
    trainer = ModelTrainer(
        data_callback = psMNISTGenerator,
        batch_size = batch_size,
        epoch_size = epoch_size,
        num_epochs = num_epochs,
        input_size = input_size,
        output_size = output_size,
        test_split = test_split,
        step_size = step_size,
        makedataset = False,
        dtype = data_dtype,
        name = 'psmnst'
    )
    
#    '''
    callback_list = [
        psMNIST_hopf_theta_callback,
#        psMNIST_hopf_radial_callback,
#        psMNIST_hopf_base_callback,
#        psMNIST_rnn_callback,
#        psMNIST_lstm_callback,
#        psMNIST_gru_callback
    ]
#    '''
    
    start_time = time.time()
    
#    try:
    
    # Starts training for callback_list models.
    res = trainer.execute(
        model_callbacks = callback_list,
        fit_valid_split = valid_split,
        do_prediction = True,              # True False
        do_evaluation = True,
        verbose = verbose,
        testing = None
    )
#    except:
#        print( 'TEST FAILED: for psMNIST' )
        
    rt = time.time() - start_time
    rt_hrs = rt // 3600
    rt_min = ( rt - 3600 * rt_hrs ) // 60
    rt_sec = ( rt - 3600 * rt_hrs - 60 * rt_min )
    rt_rem = math.fmod( rt_sec , 1.0e2 ) - math.floor( rt_sec )
    rt_sec = math.floor( rt_sec )
    print('Runtime: {:.1f} (hrs)  {:.1f} (min)  {:.1f} (sec)  {:.2f} (0.01 sec)\n'.format( rt_hrs , rt_min , rt_sec , rt_rem ))
    
    # Save all runs for callback_list models #
    trainer.save(
        max_plot_len = 2048,
        show_plot = False,       # True False
        save_plot = True,
        show_weights = False,
        save_weights = False,
        verbose = verbose
    )

    tf.keras.backend.clear_session()


def cpymem_experiments( seq_len = 10 ):
    
    ''' ----- INPUT FEATURE PARAMETERS FOR MODEL ----- '''
    tf.keras.backend.clear_session()
    
    # Model params
    input_size = 10
    output_size = input_size - 2
    
    # Train params #
    batch_size = 25
    epoch_size = 25
    num_epochs = 20
    
    step_size = 1
    test_split = 0.25       # as a percent(%) of training parameters
    valid_split = 0.2
    _verbose = 1
    
    # Dtype for data generation #
    data_dtype = np.float32
    
    seqstr = str( seq_len*2 + 2 )
    
    ''' TRAIN MODEL '''
    trainer = ModelTrainer(
        data_callback = CopyMemoryGenerator,
        in_bits = input_size,
        out_bits = input_size-2,
        max_seq = seq_len,
        batch_size = batch_size,
        epoch_size = epoch_size,
        num_epochs = num_epochs,
        input_size = input_size,
        output_size = output_size,
        test_split = test_split,
        step_size = step_size,
        makedataset = False,
        dtype = data_dtype,
        name = 'cpymem\\'+'seq_'+seqstr
    )
    
    callback_list = [
        cpymem_hopf_theta_callback,
#        cpymem_hopf_radial_callback,
#        cpymem_hopf_base_callback,
#        cpymem_rnn_callback,
#        cpymem_lstm_callback,
#        cpymem_gru_callback
    ]
    
    start_time = time.time()
    
    # Starts training for callback_list models.
    res = trainer.execute(
        model_callbacks = callback_list,
        fit_valid_split = valid_split,
        do_prediction = True,           # True False
        do_evaluation = True,
        verbose = _verbose,
        testing = 1
    )
    
    rt = time.time() - start_time
    rt_hrs = rt // 3600
    rt_min = ( rt - 3600 * rt_hrs ) // 60
    rt_sec = ( rt - 3600 * rt_hrs - 60 * rt_min )
    rt_rem = math.fmod( rt_sec , 1.0e2 ) - math.floor( rt_sec )
    rt_sec = math.floor( rt_sec )
    print('Sequence Length: ' + seqstr )
    print('Runtime: {:.1f} (hrs)  {:.1f} (min)  {:.1f} (sec)  {:.2f} (0.01 sec)'.format( rt_hrs , rt_min , rt_sec , rt_rem ))
    print( '\n' )
    
#    ''' Save all runs for callback_list models #
    if res: trainer.save(
        max_plot_len = 2048,
        show_plot = False,   # True False
        save_plot = True,
        show_weights = False,
        save_weights = True,
        verbose = _verbose
    )
#   '''


''' TESTING SECTION ''' 
if __name__ == "__main__":
    
    '''
    device_names = tf.config.list_physical_devices(device_type=None)
    print('Tensorflow Version:',tf.__version__)
    for dname in device_names:
        print('    Available Devices:',dname)
    exit(0)
#    '''
    
#    tf.debugging.set_log_device_placement(True)
    print('\n'*10 + '- '*50 + '\n'*10)
    print('\n'*10 + '- * '*25 + '\n'*10)
    print('\n'*10 + '-- * '*20 + '\n'*10)
    print('\n'*10 + '** - '*20 + '\n'*10)
    
    # Run experiments #
    now = datetime.now()
    print('Start Date-Time:\n' , now , '\n' )
    
    
    mkygls_experiments( window_size = 32 )
#    psmnist_experiments()
#    cpymem_experiments( seq_len = 24 )
    exit(0)
    
    ''' # COMMENT-OUT FOR COPY MEMORY EXPERIMENTS #
    ## Sequences = ( value * 2 + 2 )
    start_time = time.time()
    cnt = 10 ## Number of test to run for each 'value'
    seqlst = [ 999 ]*9 + [ 499 ]*cnt +  [ 249 ]*cnt + [ 124 ]*cnt + [ 49 ]*cnt + [ 24 ]*cnt
    for seq in seqlst: cpymem_experiments( seq_len = seq )
    rt = time.time() - start_time
    rt_hrs = rt // 3600
    rt_min = ( rt - 3600 * rt_hrs ) // 60
    rt_sec = ( rt - 3600 * rt_hrs - 60 * rt_min )
    rt_rem = math.fmod( rt_sec , 1.0e2 ) - math.floor( rt_sec )
    rt_sec = math.floor( rt_sec )
    print('Copy Memory Total Runtime: {:.1f} (hrs)  {:.1f} (min)  {:.1f} (sec)  {:.2f} (0.01 sec)\n'.format( rt_hrs , rt_min , rt_sec , rt_rem ))
#   '''
    
    
#    ''' # COMMENT-OUT FOR MACKEY-GLASS EXPERIMENTS #
#    callback_list = [
#        mkygls_hopf_theta_callback,
#        mkygls_hopf_radial_callback,
#        mkygls_hopf_base_callback,
#        mkygls_rnn_callback,
#        mkygls_lstm_callback,
#        mkygls_gru_callback ]
    
    start_time = time.time()
    
    cnt = 3
    winlst = [ 32 ]*cnt + [ 64 ]*cnt + [ 128 ]*cnt
    for win in winlst: mkygls_experiments( window_size = win , callback_list = [ mkygls_hopf_theta_callback ] )
    
    rt = time.time() - start_time
    rt_hrs = rt // 3600
    rt_min = ( rt - 3600 * rt_hrs ) // 60
    rt_sec = ( rt - 3600 * rt_hrs - 60 * rt_min )
    rt_rem = math.fmod( rt_sec , 1.0e2 ) - math.floor( rt_sec )
    rt_sec = math.floor( rt_sec )
    print('Mackey-Glass Total Runtime: {:.1f} (hrs)  {:.1f} (min)  {:.1f} (sec)  {:.2f} (0.01 sec)\n'.format( rt_hrs , rt_min , rt_sec , rt_rem ))
#   '''
    
    
#    ''' # COMMENT-OUT FOR COPY-MEMORY EXPERIMENTS #
#    callback_list = [
#        psMNIST_hopf_theta_callback,
#        psMNIST_hopf_radial_callback,
#        psMNIST_hopf_base_callback,
#        psMNIST_rnn_callback,
#        psMNIST_lstm_callback,
#        psMNIST_gru_callback ]
    
    start_time = time.time()
    iters = 5
    for iter in range( iters ): psmnist_experiments( callback_list = [ psMNIST_lstm_callback ] )
    for iter in range( iters ): psmnist_experiments( callback_list = [ psMNIST_gru_callback ] )
    for iter in range( iters ): psmnist_experiments( callback_list = [ psMNIST_rnn_callback ] )
    rt = time.time() - start_time
    rt_hrs = rt // 3600
    rt_min = ( rt - 3600 * rt_hrs ) // 60
    rt_sec = ( rt - 3600 * rt_hrs - 60 * rt_min )
    rt_rem = math.fmod( rt_sec , 1.0e2 ) - math.floor( rt_sec )
    rt_sec = math.floor( rt_sec )
    print('psMNIST Total Runtime: {:.1f} (hrs)  {:.1f} (min)  {:.1f} (sec)  {:.2f} (0.01 sec)\n'.format( rt_hrs , rt_min , rt_sec , rt_rem ))
#   '''
    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    