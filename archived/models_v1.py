# Standard Imports
import os
import sys
import math
from datetime import datetime
from inspect import currentframe, getframeinfo

# Numpy & Scipy Imports
import numpy as np

# Imaging and Plotting Imports
import matplotlib.pyplot as plt

# Tensorflow Imports
import tensorflow as tf

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import keras

# Local Imports

from proj_utils import save_meta, load_meta

# Locally created metrics
from metrics import NormRootMeanSquaredError , SquaredDifference

# Locally created optimizers
from optimizers import MyOptimizer

# Locally created (or) imported Models, Cells, & Layers
from test_model import TestRNNLayer
from hopfbifur_layer import HopfBifurRNNLayer
from hopfbifur_wk_layer import HopfBifurWkRNNLayer
from hopfbifur_cpx_layer import HopfBifurCpxRNNLayer
from lmu_layers import LMU


''' Generates a dictionary object (I call a structure) that is a wrapper 
    for a model. It contains useful stuff used, along with the model itself, 
    during the course of training and testing. Information is added to and 
    removed from the structure periodically during the course of training
    and testing. In the end this structure is saved as a json file in
    connection with the model.  '''
def create_struct(model_name, train_dir):
    
    # The itteration of training (Updated after we check if a meta.json exists)
    gen = 0
    
    model_struct = {}
    model_struct['name'] = model_name
    
    # Model directory with for just name
    model_dir = os.path.join( train_dir , model_name )
    model_struct['dir'] = model_dir
    
    # Load meta will return an empty dictionary if file doesn't exist
    model_meta = load_meta(model_dir)
    
    # Collect meta information for training
    if len(model_meta) > 0:
        gen = model_meta['gen'] + 1
        
    model_struct['gdir'] = os.path.join( model_dir , 'gen_'+str(gen) )
    
    model_struct['chkpnt_dirs'] = {}
    
    # Update meta information
    model_meta['gen'] = gen
    model_struct['meta'] = model_meta
    
    return model_struct


''' Finalizes the dictionary object discussed in the create_struct() function above. '''
def finalize_struct(
                    model_struct,
                    model,
                    loss = None,
                    loss_monitor='val_loss',
                    loss_mode='auto',
                    optimizer = None,
                    metrics=[],
                    chkpnts=[],
                    verbose=0
                 ):
    
    gen_dir = model_struct['gdir']
    chkpnt_str = gen_dir + '\\training\\checkpoints\\cp-{epoch:04d}.ckpt'
    chkpnt_dir = os.path.dirname( os.path.join( gen_dir , chkpnt_str ) )
    
    model_struct['chkpnt_dirs']['loss'] = chkpnt_dir
    
    if optimizer is None:
        optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 )
    
    if loss is None:
        loss = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.AUTO,
            name='mse'
        )
        
    # Always add the standard loss callback.
    loss_chkpnt = tf.keras.callbacks.ModelCheckpoint(
        filepath = chkpnt_dir,
        monitor = loss_monitor,
        mode = loss_mode,
        verbose = verbose,
        save_weights_only = True,
        save_best_only = False
    )
    
    # Setup traing and evaluation callbacks
    callbacks=[]
    callbacks.append( loss_chkpnt )
    for chkpnt in chkpnts:
        callbacks.append( chkpnt )
    model_struct['callbacks'] = callbacks
    
    # Add a standard time-series forcasting metric.
    if len( metrics ) == 0:
        metrics = [
            tf.keras.metrics.MeanAbsolutePercentageError( name = 'mape' ), # 'accuracy'
        ]
    
    model.compile(
        loss = loss,
        optimizer = optimizer,
        metrics = metrics
    )
    model.summary(
        line_length = None,
        positions = None,
        print_fn = print,
        expand_nested = True,
        show_trainable = True,
        layer_range = None
    )
    model_struct['model'] = model
    
    # Save updated meta.json information
    model_meta = model_struct['meta']
    save_meta( model_meta , model_struct['dir'] )
    del model_struct['meta']
    
    return model_struct



def mkygls_hopf_cpx_rnn_callback( input_shape , output_shape , batch_size , train_dir ):
    
    model_struct = create_struct( 'hopf_cpx_rnn' , train_dir )
    
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    with tf.device('/cpu:0'):
        hopf = HopfBifurCpxRNNLayer(
            state_size = input_shape[-1],
            output_size = output_shape[-1],
            dtype = tf.complex64,
            name = 'cpx_hopf'
            )( input )
    '''
    dense = tf.keras.layers.Dense( input_shape[-1] // 2 , activation = None)( hopf )
    dense = tf.keras.layers.Dense( 1 , activation = None )( dense )
    model = tf.keras.Model( inputs = input , outputs = dense , name = 'hopf_cpx_rnn_model' )
    '''
    model = tf.keras.Model( inputs = input , outputs = hopf , name = 'hopf_cpx_rnn_model' )
#   '''
    
    # Add a list of weight names to save for comparison
    #   Searches the weights variables before and after training 
    #   which contain the strings in the list.
    model_struct[ 'wgtlst' ] = [ 'kernel' , 'recurrent_kernel' ]
    
    # Construct any other checkpoints or addons
#    metrics = [ 'accuracy' ]
    
    # Build checkpoint callbacks, compile, and update final model structure.
    # finalize always add a loss_checkpoint.
    # If optimazer is None, then RMSProp() is set.
    # If loss is None, then MeanSquaredError() is set.
    model_struct = finalize_struct(
        model_struct,
        model,
        loss_monitor = 'loss',
        loss_mode = 'min',
#        optimizer = tf.keras.optimizers.SGD( learning_rate = 0.001 )
    )
    
    return model_struct

def mkygls_simple_lmu_callback( input_shape , output_shape , batch_size , train_dir ):
    ''' SOURCE:
        Legendre Memory Unit (LMU) Model.
        Reference:
        A. Voelker, I. Kajić, and C. Eliasmith,  “Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks,” p. 10, 2019.
        See code: <https://github.com/nengo/keras-lmu>
        '''
    
    model_struct = create_struct('lmu',train_dir)
    
    memd = 4
    thta = 4
    unts = 32
    ordr = input_shape[0]
    
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    lmu1 = LMU(
        memory_d = memd,
        order = ordr,
        theta = thta,
        hidden_cell = tf.keras.layers.SimpleRNNCell(
            units = unts,
            activation = None
        ),
        return_sequences = True,
        name = 'lmu1'
    )( input )
    lmu2 = LMU(
        memory_d = memd,
        order = ordr,
        theta = thta,
        hidden_cell = tf.keras.layers.SimpleRNNCell(
            units = unts,
            activation = None
        ),
        return_sequences = True,
        name = 'lmu2'
    )( lmu1 )
    lmu3 = LMU(
        memory_d = memd,
        order = ordr,
        theta = thta,
        hidden_cell = tf.keras.layers.SimpleRNNCell(
            units = unts,
            activation = None
        ),
        return_sequences = True,
        name = 'lmu3'
    )( lmu2 )
    lmu4 = LMU(
        memory_d = memd,
        order = ordr,
        theta = thta,
        hidden_cell = tf.keras.layers.SimpleRNNCell(
            units = output_shape[0],
            activation = None
        ),
        name = 'lmu4'
    )( lmu3 )
    model = tf.keras.Model( input , lmu4 , name = 'lmu' )
    
    # Add a list of weight names to save for comparison
    #   Searches the weights variables before and after training 
    #   which contain the strings in the list.
    model_struct['wgtlst'] = ['hidden_cell']
    
    metrics = [ 'accuracy' ]
    
    # Construct any other checkpoints or metrics
    
    metrics = [
#        tf.keras.metrics.MeanSquaredError( name = 'mse' ),
        tf.keras.metrics.RootMeanSquaredError(name = 'rmse' ),
#        tf.keras.metrics.MeanAbsolutePercentageError( name = 'mape' )
    ]
    
    _loss = tf.keras.losses.MeanSquaredError( name = 'mse' )
    
#   '''
    # Build checkpoint callbacks, compile, and update final model structure.
    # finalize always add a loss_checkpoint.
    # If optimizer is None, then RMSProp() is set.
    # If loss is None, then MeanSquaredError() is set.
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'loss',
        loss_mode = 'min',
        metrics = metrics
    )
    
    return model_struct
    
def mkygls_simple_lstm_callback( input_shape , output_shape , batch_size , train_dir ):
    
#    print( input_shape )
#    exit(0)
    
    ''' SOURCE:
    Built as a structure sudo-similar to the one described in:
    A. Voelker, I. Kajić, and C. Eliasmith,  
    “Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks,”
    p. 10, 2019. See code: <https://github.com/nengo/keras-lmu>
    '''
    model_struct = create_struct( 'lstm' , train_dir )
    
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    lstm1 = tf.keras.layers.LSTM(
        units = 25,
        return_sequences = True,
        stateful = True,
        activation = None,
        name = 'lstm1'
    )( input )
    lstm2 = tf.keras.layers.LSTM(
        units = 25,
        return_sequences = True,
        stateful = True,
        activation = None,
        name = 'lstm2'
    )( lstm1 )
    lstm3 = tf.keras.layers.LSTM(
        units = 25,
        return_sequences = True,
        stateful = True,
        activation = None,
        name = 'lstm3'
    )( lstm2 )
    lstm4 = tf.keras.layers.LSTM(
        units = 25,
        return_sequences = False,
        stateful = True,
        activation = None,
        name = 'lstm4'
    )( lstm3 )
    model = tf.keras.Model( input , lstm4 , name = 'mkygls_lstm_model' )
    
    # Add a list of weight names to save for comparison
    #   Searches the weights variables before and after training 
    #   which contain the strings in the list.
    model_struct['wgtlst'] = ['kernel']
    
    _metrics = [
#        tf.keras.metrics.MeanSquaredError( name = 'mse' ),
        tf.keras.metrics.RootMeanSquaredError(name = 'rmse' ),
#        tf.keras.metrics.MeanAbsolutePercentageError( name = 'mape' )
    ]
    
    _loss = tf.keras.losses.MeanSquaredError( name = 'mse' )
    
    # Build checkpoint callbacks, compile, and update final model structure.
    # finalize always add a loss_checkpoint.
    # If optimazer is None, then RMSProp() is set.
    # If loss is None, then MeanSquaredError() is set.
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'val_loss',
        loss_mode = 'val_min',
        metrics = metrics
    )
    
#    exit(0)
    
    return model_struct

''' ##OTHER MODELS ##
def mkygls_hopf_rnn_wk_callback( input_shape , output_shape , batch_size , train_dir ):
    
    model_struct = create_struct( 'hopf_rnn_wk_keras' , train_dir )
    
#    run_eagerly = TODO: set to reduce debugging.
    
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    hopf = HopfBifurWkRNNLayer(
        size = input_shape[-1],
        dtype = tf.float32,
        name = 'hopf_wk'
    )( input )
    dense = tf.keras.layers.Dense( input_shape[-1] , activation = None )( hopf )
    model = tf.keras.Model( inputs = input , outputs = dense , name = 'hopf_rnn_keras_model' )
#    model = tf.keras.Model(inputs=input, outputs=hopf, name='hopf_rnn_wk_keras_model')
    
    # Add a list of weight names to save for comparison
    #   Searches the weights variables before and after training 
    #   which contain the strings in the list.
    model_struct['wgtlst'] = ['kernel','recurrent_kernel']
    
    # Construct any other checkpoints or addons
    # ...
    
    # Build checkpoint callbacks, compile, and update final model structure.
    # finalize always add a loss_checkpoint.
    # If optimazer is None, then RMSProp() is set.
    # If loss is None, then MeanSquaredError() is set.
    model_struct = finalize_struct(
        model_struct,
        model,
        loss_monitor = 'loss',
        loss_mode = 'min'
    )
    
    return model_struct

def mkygls_hopf_rnn_callback( input_shape , output_shape , batch_size , train_dir ):
    
    model_struct = create_struct('hopf_rnn',train_dir)
    
    input = tf.keras.Input(shape=input_shape,batch_size=batch_size)
    hopf = HopfBifurRNNLayer(
        size=input_shape[-1],
        dtype=tf.float32,
        name='hopf'
    )( input )
    dense = tf.keras.layers.Dense(input_shape[-1]//2, activation=None)(hopf)
    dense = tf.keras.layers.Dense(1, activation=None)(dense)
    model = tf.keras.Model(inputs=input, outputs=dense, name='hopf_rnn_model')
    
    # Add a list of weight names to save for comparison
    #   Searches the weights variables before and after training 
    #   which contain the strings in the list.
    model_struct['wgtlst'] = ['kernel','recurrent_kernel']
    
    # Construct any other checkpoints or addons
    # ...
    
    
    # Build checkpoint callbacks, compile, and update final model structure.
    # finalize always add a loss_checkpoint.
    # If optimazer is None, then RMSProp() is set.
    # If loss is None, then MeanSquaredError() is set.
    model_struct = finalize_struct(
                                    model_struct,
                                    model,
#                                    loss = loss
                                    loss_monitor='loss',
                                    loss_mode='min'
#                                    optimizer=optimizer,
#                                    metrics=metrics
                                   )
    
    return model_struct

def mkygls_simple_rnn_callback( input_shape , output_shape , batch_size , train_dir ):
    
    model_struct = create_struct( 'mkygls_simple_rnn' , train_dir )
    
    input = tf.keras.Input( shape = input_shape , dtype = tf.float64 )
    rnn = tf.keras.layers.SimpleRNN(
                                       units = input_shape[-1],
                                       input_shape = input_shape,
                                       activation = "relu",
                                       name = 'layer_1'
                                      )(input)
    dens = tf.keras.layers.Dense( 16 , activation = None )( rnn )
    dens = tf.keras.layers.Dense( 1 , activation = None )( dens )
    model = tf.keras.Model( input , dens , name = 'mkygls_simple_rnn' )
    
    # Add a list of weight names to save for comparison
    #   Searches the weights variables before and after training 
    #   which contain the strings in the list.
    
    model_struct['wgtlst'] = ['kernel','bias']
    
    # Construct any other checkpoints or addons
    metrics = [
                tf.keras.metrics.RootMeanSquaredError(name='rmse', dtype=tf.float64)
               ]
    
    # Build checkpoint callbacks, compile, and update final model structure.
    # finalize always add a loss_checkpoint.
    # If optimazer is None, then RMSProp() is set.
    # If loss is None, then MeanSquaredError() is set.
    model_struct = finalize_struct(model_struct, model)
    
    return model_struct
#'''


def psMNIST_hopf_cpx_rnn_callback( input_shape , output_shape , batch_size , train_dir ):
    
    model_struct = create_struct( 'hopf_cpx_rnn' , train_dir )
    
#    input = tf.keras.Input(shape=input_shape,batch_size=batch_size)
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    with tf.device('/cpu:0'):
        hopf = HopfBifurCpxRNNLayer(
            state_size = 64, #input_shape[-1],
#            output_size = input_shape[-1],
            dtype = tf.complex64,
            name = 'cpx_hopf'
        )( input )
#    '''
    dense = tf.keras.layers.Dense( 10 , activation = None , use_bias = False )( hopf )
    model = tf.keras.Model( inputs = input , outputs = dense , name = 'hopf_cpx_rnn_model' )
    '''
    model = tf.keras.Model( inputs = input , outputs = hopf , name = 'hopf_cpx_rnn_model' )
#   '''
    
    # Add a list of weight names to save for comparison
    #   Searches the weights variables before and after training 
    #   which contain the strings in the list.
    model_struct[ 'wgtlst' ] = ['']
    
    # Construct any other checkpoints or addons
    
    _loss = tf.keras.losses.SparseCategoricalCrossentropy( from_logits = True )  # Used in sqMNIST
    _optimizer = 'adam'
    
    # Construct any other checkpoints or metrics
    
    _metrics = [ 'accuracy' ]
    
    # Build checkpoint callbacks, compile, and update final model structure.
    # finalize always add a loss_checkpoint.
    # If optimizer is None, then RMSProp() is set.
    # If loss is None, then MeanSquaredError() is set.
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'val_loss',
        loss_mode = 'val_min',
        optimizer = _optimizer,
        metrics = _metrics
    )
    
    return model_struct
    
def psMNIST_simple_lmu_callback( input_shape , output_shape , batch_size , train_dir ):
    ''' SOURCE:
        Legendre Memory Unit (LMU) Model.
        Reference:
        A. Voelker, I. Kajić, and C. Eliasmith,  “Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks,” p. 10, 2019.
        See code: <https://github.com/nengo/keras-lmu>
        '''
    
    model_struct = create_struct('lmu',train_dir)
    
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    lmu = LMU(
        memory_d = 1,
        order = 256,
        theta = input_shape[0],
        hidden_cell = tf.keras.layers.SimpleRNNCell(
            units = 212,
            activation = 'tanh'
        ),
        input_to_hidden = True,
        kernel_initializer = "ones",
    )( input )
    dense = tf.keras.layers.Dense( 10 , activation = None )( lmu )
    model = tf.keras.Model( input , dense , name = 'lmu' )
    
    # Add a list of weight names to save for comparison
    #   Searches the weights variables before and after training 
    #   which contain the strings in the list.
    model_struct['wgtlst'] = ['hidden_cell']
    
    _loss = tf.keras.losses.SparseCategoricalCrossentropy( from_logits = True )  # Used in sqMNIST
    _optimizer = 'adam'
    
    # Construct any other checkpoints or metrics
    
    _metrics = [
        'accuracy'
    ]
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 1.e-2,
        patience = 2,
        verbose = 0,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    
    _chkpnts = [ earlystop_chkpnt ]
    
    # Build checkpoint callbacks, compile, and update final model structure.
    # finalize always add a loss_checkpoint.
    # If optimizer is None, then RMSProp() is set.
    # If loss is None, then MeanSquaredError() is set.
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'val_loss',
        loss_mode = 'val_min',
        optimizer = _optimizer,
        metrics = _metrics,
        chkpnts = _chkpnts
    )
    
    return model_struct

def psMNIST_simple_lstm_callback( input_shape , output_shape , batch_size , train_dir ):
    ''' SOURCE
        Basic LSTM model provided from tensorflow. 
        Built as a structure sudo-similar to the one described in:
        A. Voelker, I. Kajić, and C. Eliasmith,  “Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks,” p. 10, 2019.
        See code: <https://github.com/nengo/keras-lmu>
        '''
    
    model_struct = create_struct( 'lstm' , train_dir )
    
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    lstm = tf.keras.layers.LSTM(
        units = 157,
        stateful = True,
        activation = 'tanh',
        name = 'lstm'
    )( input )
#    dense = tf.keras.layers.Dense( 10 , activation = None )( lstm )
    dense = tf.keras.layers.Softmax( )( lstm )
    model = tf.keras.Model( input , dense , name = 'lstm' )
    
    # Add a list of weight names to save for comparison
    #   Searches the weights variables before and after training 
    #   which contain the strings in the list.
    model_struct[ 'wgtlst' ] = [ 'kernel' ]
    
    _loss = tf.keras.losses.SparseCategoricalCrossentropy( ) # from_logits = True )
    _optimizer = 'adam'
    
    _metrics = [ 'accuracy' ]
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 1.e-2,
        patience = 5,
        verbose = 0,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    
    _chkpnts = [ earlystop_chkpnt ]
    
    # Build checkpoint callbacks, compile, and update final model structure.
    # finalize always add a loss_checkpoint.
    # If optimazer is None, then RMSProp() is set.
    # If loss is None, then MeanSquaredError() is set.
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'val_loss',
        loss_mode = 'val_min',
        optimizer = _optimizer,
        metrics = _metrics
    )
    
#    exit(0)
    
    return model_struct
    
    
    
def cpymem_hopf_cpx_rnn_callback( input_shape , output_shape , batch_size , train_dir ):
    
    print( 'cpymem hopf cpx rnn callback')
    print( 'input_shape' , input_shape )
    print( 'output_shape' , output_shape )
    print( 'batch_size' , batch_size )
#    exit(0)
    
    model_struct = create_struct( 'hopf_cpx_rnn' , train_dir )
    
#    input = tf.keras.Input(shape=input_shape,batch_size=batch_size)
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    with tf.device('/cpu:0'):
        hopf = HopfBifurCpxRNNLayer(
            state_size = input_shape[-1],
            output_size = input_shape[-1],
            dtype = tf.complex64,
            name = 'cpx_hopf'
        )( input )
#    '''
    dense = tf.keras.layers.Dense( output_shape[-1] , activation = None )( hopf )
    model = tf.keras.Model( inputs = input , outputs = dense , name = 'hopf_cpx_rnn_model' )
    '''
    model = tf.keras.Model( inputs = input , outputs = hopf , name = 'hopf_cpx_rnn_model' )
#   '''
    
    # Add a list of weight names to save for comparison
    #   Searches the weights variables before and after training 
    #   which contain the strings in the list.
#    model_struct[ 'wgtlst' ] = [ 'kernel' , 'recurrent_kernel' ]
    
    # Construct any other checkpoints or addons
    
    metrics = [
        tf.keras.metrics.RootMeanSquaredError( name = 'rmse' , dtype = tf.float64 ),
        'accuracy'
    ]
    
    # Build checkpoint callbacks, compile, and update final model structure.
    # finalize always add a loss_checkpoint.
    # If optimazer is None, then RMSProp() is set.
    # If loss is None, then MeanSquaredError() is set.
    model_struct = finalize_struct(
        model_struct,
        model,
        optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 ),
        metrics = metrics,
    )
    
    return model_struct

def cpymem_simple_lmu_callback( input_shape , output_shape , batch_size , train_dir ):
    '''
    Legendre Memory Unit (LMU) Model.
    Reference:
    A. Voelker, I. Kajić, and C. Eliasmith,  “Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks,” p. 10, 2019.
    See code: <https://github.com/nengo/keras-lmu>
    '''
    model_struct = create_struct( 'lmu' , train_dir )
    
#    print( 'input_shape' , input_shape )
#    print( 'output_shape' , output_shape )
#    print( 'batch_size' , batch_size )
#    print( 'train_dir' , train_dir )
#    exit(0)
    
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    lmu = LMU(
        memory_d = 100,
        order = 1,
        theta = input_shape[0],
        hidden_cell = tf.keras.layers.SimpleRNNCell( 5 ),
        hidden_to_memory = False,
        memory_to_memory = False,
        input_to_hidden = True,
        kernel_initializer = 'ones',
    )( input )
    dense = tf.keras.layers.Dense( 10 , activation = None , )( lstm )
    model = tf.keras.Model( input , dense , name = 'lmu' )
    
    # Add a list of weight names to save for comparison
    #   Searches the weights variables before and after training 
    #   which contain the strings in the list.
    model_struct['wgtlst'] = ['hidden_cell']
    
    _loss = tf.keras.losses.SparseCategoricalCrossentropy( from_logits = True )  # Used in sqMNIST
    optimizer = 'adam'
    
    # Construct any other checkpoints or metrics
    
    _metrics = [
        
    ]
#   '''
    # Build checkpoint callbacks, compile, and update final model structure.
    # finalize always add a loss_checkpoint.
    # If optimizer is None, then RMSProp() is set.
    # If loss is None, then MeanSquaredError() is set.
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = loss,
        optimizer = optimizer,
        metrics = _metrics
    )
    
    return model_struct

# TODO: NRU ... 

# TODO: GRU ...