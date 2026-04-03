''' Standard Imports '''
import os
import sys
import math
import random
from datetime import datetime


''' Special Imports '''
import numpy as np

import tensorflow as tf

# Clear custom imports in this layer only.
tf.keras.utils.get_custom_objects().clear()

''' Local Imports'''

import proj_utils as utils          # Print, Plot, & Helper functions 
from trainer import ModelTrainer    # Model trainer.

# Data Imports
from data import MackeyGlassGenerator, seqMNISTGenerator, psMNISTGenerator
from data import CopyMemoryGenerator

# Model imports
from models import simple_rnn_callback, simple_lstm_callback
from models import simple_lmu_callback, simple_test_callback
from models import simple_prattrnn_callback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#tf.compat.v1.disable_eager_execution()

'''
TODO:
- Setup testing benchmarks with times ... etc.
'''


# Experiment setup #
def experiments():
    
    ''' ----- INPUT FEATURE PARAMETERS FOR MODEL ----- '''
    
    # Model params
    input_size = 128
    output_size = 1
    
    # Train params
    batch_size = 128
    epoch_size = 64
    num_epochs = 128
    step_size = 1
    test_split = 0.25       # as a percent(%) of training parameters
    
    # Dtype for data generation.
    data_dtype = np.float32
    
    ''' TRAIN MODEL '''
    trainer = ModelTrainer(
                            data_callback=MackeyGlassGenerator,
                            batch_size=batch_size,
                            epoch_size=epoch_size,
                            num_epochs=num_epochs,
                            input_size=input_size,
                            output_size=output_size,
                            step_size=step_size,
                            test_split=test_split,
                            makedataset=False,
                            dtype=data_dtype,
                            name='mackeyglass'
                           )
    
    # List of models to run on same dataset.
    # - models are put together in the model.py file.
    # - callback is just a function name and is instantiated in the 
    #   trainer.py ModelTrainer().execute() function. This is so that
    #   the model can be built with the correct input_shape and output_size
    #   shape of the dataset.
    callback_list = [
#                     simple_lstm_callback
#                     simple_rnn_callback
#                     simple_lmu_callback
                     simple_prattrnn_callback
                     ]
    
    gen = 2
    moddir = 'C:\\Users\\isuan\\Desktop\\Research\\training_runs\\mackeyglass\\'
    moddir += 'simple_pratt_rnn\\gen_'+str(gen)+'\\model\\simple_pratt_rnn_model.keras'
    
    # Starts training for callback_list models.
    res = trainer.execute(  
                            model_callbacks=callback_list,
                            model_load=False,
                            model_directory=moddir,
                            fit_valid_split=0.25,
                            verbose=1
                          )

    # Save all runs for callback_list models
    if res: trainer.save(
                         max_plot_len=2048,
                         show_plot=True,
                         save_plot=True,
                         save_weights=False,
                         verbose=1
                        )
    
    # TODO: Model comparisons & further evalutaion.

''' TESTING SECTION ''' 
if __name__ == "__main__":
    
    '''
    device_names = tf.config.list_physical_devices(device_type=None)
    print('Tensorflow Version:',tf.__version__)
    for dname in device_names:
        print('    Available Devices:',dname)
    '''
    
    # Run experiments
    experiments()