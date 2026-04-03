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
from metrics import NormRootMeanSquaredError

# Locally created optimizers
from optimizers import MyOptimizer

# Locally created (or) imported Models, Cells, & Layers
from test_model import TestRNNLayer
from prattrnn_layer_not_RNN_class_cpx import PrattRNNLayer
#from prattrnn_layer_not_RNN_class import PrattRNNLayer
#from prattrnn_layer_use_RNN_class import PrattRNNLayer
from lmu_layers import LMU


''' Generates a dictionary object (I call a structure) that is a wrapper 
    for a model. It contains useful stuff used, along with the model itself, 
    during the course of training and testing. Information is added to and 
    removed from the structure periodically during the course of training
    and testing. In the end this structure is saved as a json file in
    connection with the model.  '''
def create_struct(model_name,train_dir):
    
    # The itteration of training (Updated after we check if a meta.json exists)
    gen = 0
    
    model_struct = {}
    model_struct['name'] = model_name
    
    # Model directory with for just name
    model_dir = os.path.join(train_dir,model_name)
    model_struct['dir'] = model_dir
    
    # Load meta will return an empty dictionary if file doesn't exist
    model_meta = load_meta(model_dir)
    
    # Collect meta information for training
    if len(model_meta) > 0:
        gen = model_meta['gen'] + 1
        
    model_struct['gdir'] = os.path.join(model_dir,'gen_'+str(gen))
    
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
                    optimizer = tf.keras.optimizers.RMSprop(),
                    metrics=[],
                    chkpnts=[],
                    verbose=0
                 ):
    
    gen_dir = model_struct['gdir']
    chkpnt_str = gen_dir + '\\training\\checkpoints\\cp-{epoch:04d}.ckpt'
    chkpnt_dir = os.path.dirname(os.path.join(gen_dir,chkpnt_str))
    
    model_struct['chkpnt_dirs']['loss'] = chkpnt_dir
    
    if loss is None:
        loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO,name='mse')
    
    # Always add the standard loss optimizer.
    loss_chkpnt = tf.keras.callbacks.ModelCheckpoint(
                                                      filepath=chkpnt_dir,
                                                      monitor=loss_monitor,
                                                      mode=loss_mode,
                                                      verbose=verbose,
                                                      save_weights_only=True,
                                                      save_best_only=False
                                                     )
    
    # Setup traing and evaluation callbacks
    callbacks=[]
    callbacks.append(loss_chkpnt)
    for chkpnt in chkpnts:
        callbacks.append(chkpnt)
    model_struct['callbacks'] = callbacks
    
    # Add a standard time-series forcasting metric.
    if len(metrics) == 0: metrics = [
        tf.keras.metrics.MeanAbsolutePercentageError(name='mape')
    ]
    
    model.compile( 
                    loss=loss,
                    optimizer=optimizer,
                    metrics=metrics
                   )
    model.summary( 
                    line_length=None,
                    positions=None,
                    print_fn=print,
                    expand_nested=True,
                    show_trainable=True,
                    layer_range=None
                   )
    model_struct['model'] = model
    
    # Save updated meta.json information
    model_meta = model_struct['meta']
    save_meta(model_meta,model_struct['dir'])
    del model_struct['meta']
    
    return model_struct











def simple_cpx_prattrnn_callback(input_shape, output_shape, batch_size, train_dir):
    
    model_struct = create_struct('simple_pratt_rnn',train_dir)
    
#    run_eagerly = TODO: set to reduce debugging.
    
    input = tf.keras.Input(shape=input_shape,batch_size=batch_size)
    pratt = PrattRNNLayer(
                           size=input_shape[-1],
                           dtype=tf.float32,
                           name='cpx_rnn_layer'
                          )(input)
    dense = tf.keras.layers.Dense(input_shape[-1]//2, activation=None)(pratt)
    dense = tf.keras.layers.Dense(1, activation=None)(dense)
    model = tf.keras.Model(inputs=input, outputs=dense, name='pratt_rnn')
    
    # Add a list of weight names to save for comparison
    #   Searches the weights variables before and after training 
    #   which contain the strings in the list.
    
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













def simple_prattrnn_callback(input_shape, output_shape, batch_size, train_dir):
    
    model_struct = create_struct('simple_pratt_rnn',train_dir)
    
#    run_eagerly = TODO: set to reduce debugging.
    
#    with tf.device('/cpu:0'):
    input = tf.keras.Input(shape=input_shape,batch_size=batch_size)
    pratt = PrattRNNLayer(
                           size=input_shape[-1],
                           name='rnn_layer'
                          )(input)
    dense = tf.keras.layers.Dense(input_shape[-1]//2, activation=None)(pratt)
    dense = tf.keras.layers.Dense(1, activation=None)(dense)
    model = tf.keras.Model(inputs=input, outputs=dense,name='pratt_rnn')
    
    # Add a list of weight names to save for comparison
    #   Searches the weights variables before and after training 
    #   which contain the strings in the list.
    
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
#                                    loss_monitor='val_loss',
#                                    loss_mode='auto',
#                                    optimizer=optimizer,
#                                    metrics=metrics
                                   )
    
    return model_struct













''' ----------------------- TESTING HERE ----------------------- '''

''' ----------------------- TESTING HERE ----------------------- '''



''' My Model that I'm experimenting with '''
def simple_test_callback(input_shape, output_shape, batch_size, train_dir):
    
    model_struct = create_struct('simple_test',train_dir)
    
    input = tf.keras.Input(shape=input_shape,batch_size=batch_size)
    test = TestRNNLayer(
    )(input)
    dense = tf.keras.layers.Dense(16, activation="relu")(test)
    dense = tf.keras.layers.Dense(1, activation="relu")(dense)
    model = tf.keras.Model(inputs=input, outputs=dense)
    
#    model.summary()
#    exit(0)
    
    # Add a list of weight names to save for comparison
    #   Searches the weights variables before and after training 
    #   which contain the strings in the list.
    
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
#                                    loss_monitor='val_loss',
#                                    loss_mode='auto',
#                                    optimizer=optimizer,
#                                    metrics=metrics
                                   )
    
#    exit(0)
    
    return model_struct




''' ----------------------- TESTING HERE ----------------------- '''

''' ----------------------- TESTING HERE ----------------------- '''

''' ----------------------- TESTING HERE ----------------------- '''







'''
    Legendre Memory Unit (LMU) Model.
    Reference:
    A. Voelker, I. Kajić, and C. Eliasmith,  “Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks,” p. 10, 2019.
    See code: <https://github.com/nengo/keras-lmu>
'''
def simple_lmu_callback(input_shape, output_shape, batch_size, train_dir):
    
    model_struct = create_struct('simple_lmu',train_dir)

    input = tf.keras.Input(shape=input_shape, dtype=tf.float64)
    lmu = LMU(
                memory_d=1,
                order=256,
                theta=784,
                hidden_cell=tf.keras.layers.SimpleRNNCell(units=212),
                hidden_to_memory=False,
                memory_to_memory=False,
                input_to_hidden=True,
                kernel_initializer="ones"
               )(input)
    dense = tf.keras.layers.Dense(output_shape[-1])(lmu)
    model = tf.keras.Model(input,dense,name='simple_lmu')
    
    # Add a list of weight names to save for comparison
    #   Searches the weights variables before and after training 
    #   which contain the strings in the list.
#    model_struct['wgtlst'] = ['kernel','bias']
    
#    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # Used in sqMNIST
    optimizer='adam'
    
    # Construct any other checkpoints or metrics
    metrics = [
#                tf.keras.metrics.RootMeanSquaredError(name='rmse', dtype=tf.float64),
                NormRootMeanSquaredError(name='nrmse', dtype=tf.float64),
                'accuracy'
               ]
    
    # Build checkpoint callbacks, compile, and update final model structure.
    # finalize always add a loss_checkpoint.
    # If optimizer is None, then RMSProp() is set.
    # If loss is None, then MeanSquaredError() is set.
    model_struct = finalize_struct(
                                    model_struct,
                                    model,
#                                    loss = loss
                                    loss_monitor='val_loss',
                                    loss_mode='min',
#                                    optimizer=optimizer,
                                    metrics=metrics
                                   )
    
    return model_struct












''' Basic LSTM model provided from tensorflow. 
    Built as a structure sudo-similar to the one described in:
    A. Voelker, I. Kajić, and C. Eliasmith,  “Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks,” p. 10, 2019.
    See code: <https://github.com/nengo/keras-lmu>
    '''
def simple_lstm_callback(input_shape, output_shape, batch_size, train_dir):
    
    model_struct = create_struct('simple_lstm',train_dir)
    
    input = tf.keras.Input(
                             shape=input_shape,
                             dtype=tf.float64
                            )
    lstm = tf.keras.layers.LSTM(
                                 units=25,
                                 return_sequences=True,
                                 activation='relu',
                                 name='lstm_1',
                                 dtype=tf.float64
                                )(input)
    lstm = tf.keras.layers.LSTM(
                                 units=25,
                                 return_sequences=True,
                                 activation='relu',
                                 name='lstm_2',
                                 dtype=tf.float64
                                )(lstm)
    lstm = tf.keras.layers.LSTM(
                                 units=25,
                                 return_sequences=True,
                                 activation='relu',
                                 name='lstm_3',
                                 dtype=tf.float64
                                )(lstm)
    lstm = tf.keras.layers.LSTM(
                                 units=25,
                                 activation='relu',
                                 name='lstm_4',
                                 dtype=tf.float64
                                )(lstm)
    dens = tf.keras.layers.Dense(
                                  output_shape[-1],
                                  dtype=tf.float64
                                 )(lstm)
    model = tf.keras.Model(input, dens)
    model.summary()
    
    # Add a list of weight names to save for comparison
    #   Searches the weights variables before and after training 
    #   which contain the strings in the list.
#    model_struct['wgtlst'] = ['kernel','bias']
    
    metrics = [
                tf.keras.metrics.RootMeanSquaredError(name='rmse', dtype=tf.float64)
               ]
    
    # Build checkpoint callbacks, compile, and update final model structure.
    # finalize always add a loss_checkpoint.
    model_struct = finalize_struct( model_struct, model, metrics=metrics)
    
    return model_struct











''' Basic RNN model provided from tensorflow. '''
def simple_rnn_callback(input_shape, output_shape, batch_size, train_dir):
    
    model_struct = create_struct('simple_rnn',train_dir)
    
    input = tf.keras.Input(shape=input_shape,dtype=tf.float64)
    rnn = tf.keras.layers.SimpleRNN(
                                       units=input_shape[-1],
                                       input_shape=input_shape,
                                       activation="relu",
                                       name='layer_1'
                                      )(input)
    dens = tf.keras.layers.Dense(16, activation="relu")(rnn)
    dens = tf.keras.layers.Dense(1,activation="relu")(dens)
    model = tf.keras.Model(input,dens,name='simple_rnn')
    
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






# TODO: NRU ... 

# TODO: GRU ...