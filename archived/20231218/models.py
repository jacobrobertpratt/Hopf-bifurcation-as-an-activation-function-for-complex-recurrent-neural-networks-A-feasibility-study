# Standard Imports
import os
import sys
import math
from datetime import datetime
from inspect import currentframe, getframeinfo

# Numpy & Scipy Imports
import numpy as np
import matplotlib.pyplot as plt

## Tensorflow Imports ##
import tensorflow as tf
#tf.debugging.set_log_device_placement( True )
tf.config.set_soft_device_placement( True )

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import keras

## Local Imports ##
from proj_utils import save_meta, load_meta

## Locally created metrics ##
from metrics import NormRootMeanSquaredError , SquaredDifference

## Locally created (or) imported Models, Cells, & Layers ##
from lmu.lmu_layers import LMU

## My Layers ##
from hopf_layers import HopfRNNLayerBase , HopfRNNLayerRadial , HopfRNNLayerTheta

''' Generates a dictionary object that is a wrapper  for a model.
    It contains useful stuff used, along with the model itself, 
    during the course of training and testing. Information is added to and 
    removed from the structure periodically during the course of training
    and testing. In the end this structure is saved as a json file in
    connection with the model. '''
def create_struct(model_name, train_dir):
    
    # The itteration of training (Updated after we check if a meta.json exists)
    gen = 0
    
    model_struct = {}
    model_struct['name'] = model_name
    
    # Model directory with for just name
    model_dir = os.path.join( train_dir , model_name )
    model_struct['dir'] = model_dir
    
    # Load meta will return an empty dictionary if file doesn't exist
    model_meta = load_meta( model_dir )
    
    # Collect meta information for training
    if len( model_meta ) > 0:
        gen = model_meta['gen'] + 1
        
    model_struct['gdir'] = os.path.join( model_dir , 'gen_'+str(gen) )
    
    model_struct['chkpnt_dirs'] = {}
    
    # Update meta information
    model_meta['gen'] = gen
    if 'csvset' not in model_meta: model_meta['csvset'] = 0
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
                    shuffle=False,
                    verbose=0
                 ):
    
#    gen_dir = model_struct['gdir']
#    chkpnt_str = gen_dir + '\\training\\checkpoints\\'
    chkpnt_dir = model_struct['gdir'] + '\\training\\checkpoints\\cp-{epoch:04d}.ckpt'
    
#    print( 'model_struct[\'gdir\']' , model_struct['gdir'] )
#    print( 'chkpnt_dir' , chkpnt_dir )
#    exit(0)
    
    if shuffle is True: model_struct['shuffle'] = True
    else: model_struct['shuffle'] = False
    
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
        verbose = verbose,
        mode = loss_mode,
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
    if metrics is not None:
        if len( metrics ) == 0:
            metrics = [ tf.keras.metrics.MeanAbsolutePercentageError( name = 'mape' ) ]
            
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




def mkygls_hopf_theta_callback( input_shape , output_shape , batch_size , train_dir ):
    
    model_struct = create_struct( 'hopf_theta' , train_dir )
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    with tf.device( '/cpu:0' ):
        hopf1 = HopfRNNLayerTheta(
            units = 8,
            activation = 'hopf',
#            activation = 'tanh',
#            activation = 'relu',
#            activation = None,
            return_sequences = True,
            stateful = True,
            dtype = tf.complex64,
            name = 'hopf_L1'
        )( input )
    '''
    hopf2 = tf.keras.layers.SimpleRNN(
        units = output_shape[-1],
        activation = 'relu',
        return_sequences = True,
        stateful = True,
        name = 'hopf_L2'
    )( hopf1 )
    model = tf.keras.Model( inputs = input , outputs = hopf2 , name = 'mkygls_hopf_theta' )
    '''
    dens1 = tf.keras.layers.Dense( output_shape[-1] , activation = None )( hopf1 )
    model = tf.keras.Model( input , dens1 , name = 'mkygls_hopf_theta' )
#   '''
    
#    model_struct[ 'wgtlst' ] = [ 'H' , 'G' , 'M' , 'N' ]
    
    # Construct any other checkpoints or addons
    _metrics = [
        tf.keras.metrics.RootMeanSquaredError(name = 'rmse' ),
#        tf.keras.metrics.MeanAbsolutePercentageError( name = 'mape' )
    ]
    
    _optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 )  # Need device == cpu only.
#    _optimizer = tf.keras.optimizers.SGD( learning_rate = 0.001 )
    
    _loss = tf.keras.losses.MeanSquaredError( name = 'mse' )
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'loss',
        min_delta = 1.e-3,
        patience = 15,
        verbose = 1,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    _chkpnts = [ earlystop_chkpnt ]
    
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'loss',
        loss_mode = 'min',
        metrics = _metrics,
        optimizer = _optimizer,
        chkpnts = _chkpnts
    )
    
    return model_struct

def mkygls_hopf_radial_callback( input_shape , output_shape , batch_size , train_dir ):
    
    model_struct = create_struct( 'hopf_radial' , train_dir )
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    hopf1 = HopfRNNLayerRadial(
        state_size = 32,
        return_sequences = True,
        stateful = True,
        name = 'hopf_L1'
    )( input )
#    '''    # UNCOMMENT TO MAKE SINGLE LAYER WITH DENSE->SOFTMAX FINAL LAYER #
    hopf2 = HopfRNNLayerRadial(
        state_size = output_shape[-1],
        return_sequences = True,
        stateful = True,
        name = 'hopf_L2'
    )( hopf1 )
    model = tf.keras.Model( inputs = input , outputs = hopf2 , name = 'mkygls_hopf_radial' )
    '''
    dense = tf.keras.layers.Dense( output_shape[-1] , activation = 'relu' )( hopf1 )
    model = tf.keras.Model( inputs = input , outputs = dense , name = 'mkygls_hopf_base' )
#   '''
    
    model_struct[ 'wgtlst' ] = [ 'A' , 'B' ]
    
    # Construct any other checkpoints or addons
    _metrics = [ tf.keras.metrics.RootMeanSquaredError( name = 'rmse' ) ]
    
    _optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 )
#    _optimizer = tf.keras.optimizers.SGD( learning_rate = 0.001 )
    
    _loss = tf.keras.losses.MeanSquaredError( name = 'mse' )
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'loss',
        min_delta = 1.e-3,
        patience = 10,
        verbose = 1,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    
    _chkpnts = [ earlystop_chkpnt ]
    
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'loss',
        loss_mode = 'min',
        metrics = _metrics,
        optimizer = _optimizer,
        chkpnts = _chkpnts
    )
    
    return model_struct

def mkygls_hopf_base_callback( input_shape , output_shape , batch_size , train_dir ):
    
    model_struct = create_struct( 'hopf_base' , train_dir )
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    hopf1 = HopfRNNLayerBase(
        state_size = 32,
        activation = tf.keras.activations.tanh,
#        activation = None,
        return_sequences = True,
        stateful = True,
        name = 'hopf_L1'
    )( input )
#    '''    # UNCOMMENT TO MAKE SINGLE LAYER WITH DENSE->SOFTMAX FINAL LAYER #
    hopf2 = HopfRNNLayerBase(
        state_size = output_shape[-1],
        activation = tf.keras.activations.tanh,
#        activation = None,
        return_sequences = True,
        stateful = True,
        name = 'hopf_L2'
    )( hopf1 )
    model = tf.keras.Model( inputs = input , outputs = hopf2 , name = 'mkygls_hopf_base' )
    '''
    dense = tf.keras.layers.Dense( output_shape[-1] , activation = 'relu' )( hopf1 )
    model = tf.keras.Model( inputs = input , outputs = dense , name = 'mkygls_hopf_base' )
#   '''
    
    model_struct[ 'wgtlst' ] = [ 'A' , 'B' ]
    
    # Construct any other checkpoints or addons
    _metrics = [ tf.keras.metrics.RootMeanSquaredError( name = 'rmse' ) ]
    
    _optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 )
#    _optimizer = tf.keras.optimizers.SGD( learning_rate = 0.001 )
    
    _loss = tf.keras.losses.MeanSquaredError( name = 'mse' )
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'loss',
        min_delta = 1.e-3,
        patience = 10,
        verbose = 1,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    
    _chkpnts = [ earlystop_chkpnt ]
    
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'loss',
        loss_mode = 'min',
        metrics = _metrics,
        optimizer = _optimizer,
        chkpnts = _chkpnts
    )
    
    return model_struct
    
def mkygls_rnn_callback( input_shape , output_shape , batch_size , train_dir ):
    
    model_struct = create_struct( 'rnn' , train_dir )
    
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    rnn1 = tf.keras.layers.SimpleRNN(
        units = 32,
        return_sequences = True,
        stateful = True,
        activation = 'tanh',
        name = 'rnn_L1'
    )( input )
#    '''    # UNCOMMENT TO MAKE SINGLE LAYER WITH DENSE->SOFTMAX FINAL LAYER #
    rnn2 = tf.keras.layers.SimpleRNN(
        units = output_shape[-1],
        return_sequences = True,
        stateful = True,
        activation = 'relu',
        name = 'rnn_L2'
    )( rnn1 )
    model = tf.keras.Model( input , rnn2 , name = 'mkygls_rnn' )
    '''
    dense = tf.keras.layers.Dense( output_shape[-1] , activation = 'relu' )( rnn1 )
    model = tf.keras.Model( input , dense , name = 'mkygls_rnn' )
#   '''
    
    _optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 )
#    _optimizer = tf.keras.optimizers.SGD( learning_rate = 0.001 )
    
    _metrics = [
        tf.keras.metrics.RootMeanSquaredError(name = 'rmse' ),
        tf.keras.metrics.MeanAbsolutePercentageError( name = 'mape' )
    ]
    
    _loss = tf.keras.losses.MeanSquaredError( name = 'mse' )
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'loss',
        min_delta = 1.e-3,
        patience = 10,
        verbose = 1,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    
    _chkpnts = [ earlystop_chkpnt ]
    
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'loss',
        loss_mode = 'min',
        metrics = _metrics,
        optimizer = _optimizer,
        chkpnts = _chkpnts
    )
    
    return model_struct
    
def mkygls_lstm_callback( input_shape , output_shape , batch_size , train_dir ):
    ''' SOURCE:
        Built as a structure sudo-similar to the one described in:
        A. Voelker, I. Kajić, and C. Eliasmith,  
        “Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks,”
        p. 10, 2019. See code: <https://github.com/nengo/keras-lmu>
        '''
    
    model_struct = create_struct( 'lstm' , train_dir )
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    lstm1 = tf.keras.layers.LSTM(
        units = 32,
        activation = 'tanh',
        return_sequences = True,
        stateful = True,
        name = 'lstm_L1'
    )( input )
#    '''    # UNCOMMENT TO MAKE SINGLE LAYER WITH DENSE->SOFTMAX FINAL LAYER #
    lstm2 = tf.keras.layers.LSTM(
        units = output_shape[-1],
        activation = 'relu',
        return_sequences = True,
        stateful = True,
        name = 'lstm_L2'
    )( lstm1 )
    model = tf.keras.Model( input , lstm2 , name = 'mkygls_lstm' )
    '''
    dense = tf.keras.layers.Dense( output_shape[-1] , activation = 'relu' )( lstm1 )
    model = tf.keras.Model( input , dense , name = 'mkygls_lstm' )
#   '''
    
    _optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 )
    
    _metrics = [
        tf.keras.metrics.RootMeanSquaredError(name = 'rmse' ),
        tf.keras.metrics.MeanAbsolutePercentageError( name = 'mape' )
    ]
    
    _loss = tf.keras.losses.MeanSquaredError( name = 'mse' )
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'loss',
        min_delta = 1.e-3,
        patience = 10,
        verbose = 1,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    
    _chkpnts = [ earlystop_chkpnt ]
    
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'loss',
        loss_mode = 'min',
        metrics = _metrics,
        optimizer = _optimizer,
        chkpnts = _chkpnts
    )
    
    return model_struct

def mkygls_gru_callback( input_shape , output_shape , batch_size , train_dir ):
    ''' SOURCE:
        Built as a structure sudo-similar to the one described in:
        A. Voelker, I. Kajić, and C. Eliasmith,  
        “Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks,”
        p. 10, 2019. See code: <https://github.com/nengo/keras-lmu>
        '''
    
    model_struct = create_struct( 'gru' , train_dir )
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    gru1 = tf.keras.layers.GRU(
        units = 32,
        activation = 'tanh',
        return_sequences = True,
        stateful = True,
        name = 'gru_L1'
    )( input )
#    '''    # UNCOMMENT TO MAKE SINGLE LAYER WITH DENSE->SOFTMAX FINAL LAYER #
    gru2 = tf.keras.layers.GRU(
        units = output_shape[-1],
        activation = 'relu',
        return_sequences = True,
        stateful = True,
        name = 'gru_L2'
    )( gru1 )
    model = tf.keras.Model( input , gru2 , name = 'mkygls_gru' )
    '''
    dense = tf.keras.layers.Dense( output_shape[-1] , activation = 'relu' )( gru1 )
    model = tf.keras.Model( input , dense , name = 'mkygls_gru' )
#   '''
    
    _optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 )
    
    _metrics = [
        tf.keras.metrics.RootMeanSquaredError(name = 'rmse' ),
        tf.keras.metrics.MeanAbsolutePercentageError( name = 'mape' )
    ]
    
    _loss = tf.keras.losses.MeanSquaredError( name = 'mse' )
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'loss',
        min_delta = 1.e-3,
        patience = 10,
        verbose = 1,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    
    _chkpnts = [ earlystop_chkpnt ]
    
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'loss',
        loss_mode = 'min',
        metrics = _metrics,
        optimizer = _optimizer,
        chkpnts = _chkpnts
    )
    
    return model_struct







def psMNIST_hopf_theta_callback( input_shape , output_shape , batch_size , train_dir ):
    
    model_struct = create_struct( 'hopf_theta' , train_dir )
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    hopf1 = HopfRNNLayerTheta(
        units = 20,
#        activation = None,
        activation = 'hopf',
#        activation = tf.keras.activations.tanh,
        stateful = True,
        dtype = tf.complex64,
        name = 'hopf_L1'
    )( input )
    dens1 = tf.keras.layers.Dense( 10 , activation = 'softmax' )( hopf1 )
    model = tf.keras.Model( inputs = input , outputs = dens1 , name = 'psmnist_hopf_theta' )
    
#    model_struct[ 'wgtlst' ] = [ 'A' , 'B' , 'C' ]
    
    _loss = tf.keras.losses.SparseCategoricalCrossentropy()
    _optimizer = tf.keras.optimizers.SGD( learning_rate = 0.01 )
    
    _metrics = [ 'accuracy' ]
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 1.e-3,
        patience = 15,
        verbose = 0,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    _chkpnts = [ earlystop_chkpnt ]
    
    # Build checkpoint callbacks, compile, and update final model structure.
    #   finalize always add a loss_checkpoint.
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'val_loss',
        loss_mode = 'val_min',
        optimizer = _optimizer,
        metrics = _metrics,
        chkpnts = _chkpnts,
        shuffle = True
    )
    
    return model_struct

def psMNIST_hopf_radial_callback( input_shape , output_shape , batch_size , train_dir ):
    
    model_struct = create_struct( 'hopf_radial' , train_dir )
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    hopf1 = HopfRNNLayerRadial(
        state_size = 10,
        stateful = True,
        name = 'hopf_L1'
    )( input )
    dense = tf.keras.layers.Dense( 10 , activation = 'softmax' )( hopf1 )
    model = tf.keras.Model( inputs = input , outputs = dense , name = 'psmnist_hopf_radial' )
    
    model_struct[ 'wgtlst' ] = [ 'A' , 'B' ]
    
    _loss = tf.keras.losses.SparseCategoricalCrossentropy()
    _optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 )
    
    _metrics = [ 'accuracy' ]
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 1.e-3,
        patience = 15,
        verbose = 0,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    
    _chkpnts = [ earlystop_chkpnt ]
    
    # Build checkpoint callbacks, compile, and update final model structure.
    #   finalize always add a loss_checkpoint.
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'val_loss',
        loss_mode = 'val_min',
        optimizer = _optimizer,
        metrics = _metrics,
        shuffle = True
    )
    
    return model_struct

def psMNIST_hopf_base_callback( input_shape , output_shape , batch_size , train_dir ):
    
    model_struct = create_struct( 'hopf_base' , train_dir )
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    hopf1 = HopfRNNLayerBase(
        state_size = 10,
#        activation = tf.keras.activations.relu,
        activation = None,
        return_sequences = False,
        stateful = True,
        name = 'hopf_L1'
    )( input )
    dense = tf.keras.layers.Dense( 10 , activation = 'softmax' )( hopf1 )
    model = tf.keras.Model( inputs = input , outputs = dense , name = 'psmnist_hopf_base' )
    
    model_struct[ 'wgtlst' ] = [ 'A' , 'B' ]
    
    _loss = tf.keras.losses.SparseCategoricalCrossentropy()
    _optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 )
    
    _metrics = [ 'accuracy' ]
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 1.e-3,
        patience = 15,
        verbose = 0,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    
    _chkpnts = [ earlystop_chkpnt ]
    
    # Build checkpoint callbacks, compile, and update final model structure.
    #   finalize always add a loss_checkpoint.
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'val_loss',
        loss_mode = 'val_min',
        optimizer = _optimizer,
        metrics = _metrics,
        shuffle = True
    )
    
    return model_struct

def psMNIST_rnn_callback( input_shape , output_shape , batch_size , train_dir ):
    
    model_struct = create_struct( 'rnn' , train_dir )
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    rnn1 = tf.keras.layers.SimpleRNN(
        units = 100,
        activation = 'tanh',
        kernel_initializer = 'glorot_normal',
        recurrent_initializer = 'glorot_normal',
        stateful = True,
        name = 'rnn_L1'
    )( input )
    dens1 = tf.keras.layers.Dense( 10 , activation = 'softmax' )( rnn1 )
    model = tf.keras.Model( input , dens1 , name = 'psmnist_rnn' )
    
    model_struct[ 'wgtlst' ] = [ 'kernel' , 'recurrent_kernel' , 'bias' ]
    
#    _loss = tf.keras.losses.CategoricalCrossentropy()
    _loss = tf.keras.losses.SparseCategoricalCrossentropy()
    _optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 )
    
    _metrics = [ 'accuracy' ]
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 1.e-3,
        patience = 15,
        verbose = 1,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    _chkpnts = [ earlystop_chkpnt ]
    
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'val_loss',
        loss_mode = 'val_min',
        optimizer = _optimizer,
        metrics = _metrics,
        chkpnts = _chkpnts,
        shuffle = True
    )
    
#    exit(0)
    
    return model_struct

def psMNIST_lstm_callback( input_shape , output_shape , batch_size , train_dir ):
    
    model_struct = create_struct( 'lstm' , train_dir )
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    lstm1 = tf.keras.layers.LSTM(
        units = 150,
        activation = 'tanh',
        kernel_initializer = 'glorot_normal',
        recurrent_initializer = 'glorot_normal',
        stateful = True,
        name = 'lstm_L1'
    )( input )
    dens1 = tf.keras.layers.Dense( 10 , activation = 'softmax' )( lstm1 )
    model = tf.keras.Model( input , dens1 , name = 'psmnist_lstm' )
    
    model_struct[ 'wgtlst' ] = [ 'kernel' , 'recurrent_kernel' , 'bias' ]
    
#    _loss = tf.keras.losses.CategoricalCrossentropy()
    _loss = tf.keras.losses.SparseCategoricalCrossentropy()
    _optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 )
    
    _metrics = [ 'accuracy' ]
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 1.e-3,
        patience = 15,
        verbose = 1,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    _chkpnts = [ earlystop_chkpnt ]
    
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'val_loss',
        loss_mode = 'val_min',
        optimizer = _optimizer,
        metrics = _metrics,
        chkpnts = _chkpnts,
        shuffle = True
    )
    
    return model_struct

def psMNIST_gru_callback( input_shape , output_shape , batch_size , train_dir ):
    
    model_struct = create_struct( 'gru' , train_dir )
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    gru1 = tf.keras.layers.GRU(
        units = 175,
        activation = 'tanh',
        kernel_initializer = 'glorot_normal',
        recurrent_initializer = 'glorot_normal',
        stateful = True,
        name = 'gru_L1'
    )( input )
    dens1 = tf.keras.layers.Dense( 10 , activation = 'softmax' )( gru1 )
    model = tf.keras.Model( input , dens1 , name = 'psmnist_gru' )
    
    model_struct[ 'wgtlst' ] = [ 'kernel' , 'recurrent_kernel' , 'bias' ]
    
#    _loss = tf.keras.losses.CategoricalCrossentropy()
    _loss = tf.keras.losses.SparseCategoricalCrossentropy()
    _optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 )
    
    _metrics = [ 'accuracy' ]
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 1.e-3,
        patience = 15,
        verbose = 1,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    _chkpnts = [ earlystop_chkpnt ]
    
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'val_loss',
        loss_mode = 'val_min',
        optimizer = _optimizer,
        metrics = _metrics,
        chkpnts = _chkpnts,
        shuffle = True
    )
    
    return model_struct









def cpymem_hopf_theta_callback( input_shape , output_shape , batch_size , train_dir ):
    
    model_struct = create_struct( 'hopf_theta' , train_dir )
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
#    '''
    hopf1 = HopfRNNLayerTheta(
        units = 32,
#        activation = None,
        activation = 'hopf',
        return_sequences = True,
        stateful = True,
        dtype = tf.complex64,
        name = 'hopf_L1'
    )( input )
#   '''
    '''
    hopf1 = tf.keras.layers.SimpleRNN(
        units = 32,
        kernel_initializer = 'glorot_normal',
        recurrent_initializer = 'orthogonal',
        return_sequences = True,
        stateful = True,
        activation = 'tanh',
        name = 'hopf_L1'
    )( input )
#    '''
    hopf2 = HopfRNNLayerTheta(
        units = output_shape[-1],
        activation = 'hopf',
        return_sequences = True,
        stateful = True,
        dtype = tf.complex64,
        name = 'hopf_L2'
    )( hopf1 )
    model = tf.keras.Model( input , hopf2 , name = 'cpymem_hopf_theta' )
    
#    model_struct[ 'wgtlst' ] = [ 'A' , 'B' , 'C' ]
    
    _loss = tf.keras.losses.BinaryCrossentropy()
    _optimizer = tf.keras.optimizers.SGD( learning_rate = 0.001 )
    
    # Construct any other checkpoints or metrics
    _metrics = None
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 1.e-3,
        patience = 10,
        verbose = 0,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    _chkpnts = [ earlystop_chkpnt ]
    
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'val_loss',
        loss_mode = 'val_min',
        optimizer = _optimizer,
        metrics = _metrics,
        chkpnts = _chkpnts,
        shuffle = True
    )
    
    return model_struct

def cpymem_hopf_radial_callback( input_shape , output_shape , batch_size , train_dir ):
    
    model_struct = create_struct( 'hopf_radial' , train_dir )
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    hopf1 = HopfRNNLayerRadial(
        state_size = 50,
        return_sequences = True,
        stateful = True,
        name = 'hopf_L1'
    )( input )
    hopf2 = HopfRNNLayerRadial(
        state_size = output_shape[-1],
        return_sequences = True,
        stateful = True,
        activation = tf.keras.activations.tanh,
        name = 'hopf_L2'
    )( hopf1 )
    model = tf.keras.Model( input , hopf2 , name = 'cpymem_hopf_radial' )

    model_struct[ 'wgtlst' ] = [ 'A' , 'B' ]

    _loss = tf.keras.losses.BinaryCrossentropy()
    _optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 )
    
    # Construct any other checkpoints or metrics
    _metrics = None # [ tf.keras.metrics.BinaryCrossentropy() ]
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'loss',
        min_delta = 1.e-3,
        patience = 15,
        verbose = 0,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    
    _chkpnts = [ earlystop_chkpnt ]
    
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'loss',
        loss_mode = 'min',
        optimizer = _optimizer,
        metrics = _metrics,
        chkpnts = _chkpnts,
        shuffle = True
    )
    
    return model_struct

def cpymem_hopf_base_callback( input_shape , output_shape , batch_size , train_dir ):
    
    model_struct = create_struct( 'hopf_base' , train_dir )
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    hopf1 = HopfRNNLayerBase(
        state_size = 50,
        activation = tf.keras.activations.tanh,
        return_sequences = True,
        stateful = True,
        name = 'hopf_L1'
    )( input )
    '''
    hopf2 = HopfRNNLayerBase(
        state_size = output_shape[-1],
        return_sequences = True,
        stateful = True,
        activation = tf.keras.activations.tanh,
        name = 'hopf_L2'
    )( hopf1 )
    '''
    rnn2 = tf.keras.layers.SimpleRNN(
        units = output_shape[-1],
        return_sequences = True,
        stateful = True,
        activation = 'sigmoid',
        name = 'rnn_L2'
    )( hopf1 )
    model = tf.keras.Model( input , rnn2 , name = 'cpymem_hopf_base' )

    model_struct[ 'wgtlst' ] = [ 'A' , 'B' ]

    _loss = tf.keras.losses.BinaryCrossentropy()
    _optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 )
    
    # Construct any other checkpoints or metrics
    _metrics = None # [ tf.keras.metrics.BinaryCrossentropy() ]
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'loss',
        min_delta = 1.e-3,
        patience = 15,
        verbose = 0,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    
    _chkpnts = [ earlystop_chkpnt ]
    
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'loss',
        loss_mode = 'min',
        optimizer = _optimizer,
        metrics = _metrics,
        chkpnts = _chkpnts,
        shuffle = True
    )
    
    return model_struct

def cpymem_rnn_callback( input_shape , output_shape , batch_size , train_dir ):
    
    model_struct = create_struct( 'rnn' , train_dir )
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    rnn1 = tf.keras.layers.SimpleRNN(
        units = 25,
        kernel_initializer = 'glorot_normal',
        recurrent_initializer = 'orthogonal',
        return_sequences = True,
        stateful = True,
        activation = 'tanh',
        name = 'rnn_L1'
    )( input )
    rnn2 = tf.keras.layers.SimpleRNN(
        units = output_shape[-1],
        kernel_initializer = 'glorot_normal',
        recurrent_initializer = 'orthogonal',
        return_sequences = True,
        stateful = True,
        activation = 'sigmoid',
        name = 'rnn_L2'
    )( rnn1 )
    model = tf.keras.Model( input , rnn2 , name = 'cpymem_rnn' )
    
    model_struct[ 'wgtlst' ] = [ 'kernel' , 'recurrent_kernel' , 'bias' ]
    
    _loss = tf.keras.losses.BinaryCrossentropy()
    _optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 )
    
    # Construct any other checkpoints or metrics
    _metrics = None
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 1.e-3,
        patience = 10,
        verbose = 0,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    _chkpnts = [ earlystop_chkpnt ]
    
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'val_loss',
        loss_mode = 'val_min',
        optimizer = _optimizer,
        metrics = _metrics,
        chkpnts = _chkpnts,
        shuffle = True
    )
    
    return model_struct

def cpymem_lstm_callback( input_shape , output_shape , batch_size , train_dir ):
    
#    print( output_shape )
#    exit(0)
    
    model_struct = create_struct( 'lstm' , train_dir )
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    lstm1 = tf.keras.layers.LSTM(
        units = 25,
        activation = 'tanh',
        recurrent_activation = 'sigmoid',
        kernel_initializer = 'glorot_normal',
        recurrent_initializer = 'orthogonal',
        return_sequences = True,
        stateful = True,
        name = 'lstm_L1'
    )( input )
    lstm2 = tf.keras.layers.LSTM(
        activation = 'sigmoid',
        recurrent_activation = 'sigmoid',
        units = output_shape[-1],
        kernel_initializer = 'glorot_normal',
        recurrent_initializer = 'orthogonal',
        return_sequences = True,
        stateful = True,
        name = 'lstm_L2'
    )( lstm1 )
    model = tf.keras.Model( input , lstm2 , name = 'cpymem_lstm' )
    
    model_struct[ 'wgtlst' ] = [ 'kernel' , 'recurrent_kernel' , 'bias' ]
    
    _loss = tf.keras.losses.BinaryCrossentropy()
    _optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 )
    
    # Construct any other checkpoints or metrics
    _metrics = None
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 1.e-3,
        patience = 15,
        verbose = 0,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    _chkpnts = [ earlystop_chkpnt ]
    
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'val_loss',
        loss_mode = 'val_min',
        optimizer = _optimizer,
        metrics = _metrics,
        chkpnts = _chkpnts,
        shuffle = True
    )
    
    return model_struct

def cpymem_gru_callback( input_shape , output_shape , batch_size , train_dir ):
    
    model_struct = create_struct( 'gru' , train_dir )
    input = tf.keras.Input( shape = input_shape , batch_size = batch_size )
    gru1 = tf.keras.layers.GRU(
        units = 25,
        activation = 'tanh',
        recurrent_activation = 'sigmoid',
        kernel_initializer = 'glorot_normal',
        recurrent_initializer = 'orthogonal',
        return_sequences = True,
        stateful = True,
        name = 'gru_L1'
    )( input )
    gru2 = tf.keras.layers.GRU(
        units = output_shape[-1],
        activation = 'sigmoid',
        recurrent_activation = 'sigmoid',
        kernel_initializer = 'glorot_normal',
        recurrent_initializer = 'orthogonal',
        return_sequences = True,
        stateful = True,
        name = 'gru_L2'
    )( gru1 )
    model = tf.keras.Model( input , gru2 , name = 'cpymem_gru' )
    
    model_struct[ 'wgtlst' ] = [ 'kernel' , 'recurrent_kernel' , 'bias' ]
    
    _loss = tf.keras.losses.BinaryCrossentropy()
    _optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 )
    
    # Construct any other checkpoints or metrics
    _metrics = None
    
    earlystop_chkpnt = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 1.e-3,
        patience = 10,
        verbose = 0,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    _chkpnts = [ earlystop_chkpnt ]
    
    model_struct = finalize_struct(
        model_struct,
        model,
        loss = _loss,
        loss_monitor = 'val_loss',
        loss_mode = 'val_min',
        optimizer = _optimizer,
        metrics = _metrics,
        chkpnts = _chkpnts,
    )
    
#    exit(0)
    
    return model_struct


