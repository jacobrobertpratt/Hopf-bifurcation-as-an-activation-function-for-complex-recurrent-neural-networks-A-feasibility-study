''' Standard Imports '''
import os
import sys
import math
import random
from datetime import datetime


''' Special Imports '''
import numpy as np
import scipy
from scipy import signal
from scipy.stats import unitary_group

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import rgb2hex

import cv2
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'



''' Produces the MackeyGlass dataset for input parameters '''
def get_mackey_glass(tao=30,delta_x=10,steps=600):
    y = [0.2]
    x = [0]
    delta = 1/delta_x
    for t in range(steps):
        y_ = 0.0
        if t < tao:
            y_ = y[t] + delta * ((0.2 * y[t])/(1 + pow(y[t],10)) - 0.1 * y[t])
        else:
            y_ = y[t] + ((0.2 * y[t-tao])/(1 + pow(y[t-tao],10)) - 0.1 * y[t])
        y.append(y_)
    return y
   

class TestRNNCell( tf.keras.layers.Layer ):
    
    def __init__( self , **kwargs ):
        
        super( TestRNN , self ).__init__( **kwargs )
        
    def build( self , input_shape ):
        
        super( TestRNN , self ).build( input_shape )
        
        self.built = True
        
    def call( self , inputs , states , training = False ):
        
        return inputs , states 
        
        
class TestRNNLayer(tf.keras.layers.Layer):
    
    def __init__(self,
                state_size,
                output_size,
                **kwargs):
        
        self.state_size = state_size
        self.output_size = output_size
        
        super(TestRNNLayer, self).__init__(**kwargs)

    def build(self , input_shape ):
        
        batch_size = input_shape[0]
        feature_size = input_shape[-1]
        
        self.state = self.add_weight(
            name='global_state',
            shape=( batch_size , feature_size , self.state_size ),
            dtype=self.dtype,
            initializer=tf.keras.initializers.Orthogonal,
        )
        
        self.output_kernel = self.add_weight(
            name='output_kernel',
            shape=( batch_size , self.state_size , self.output_size ),
            dtype=self.dtype,
            initializer=tf.keras.initializers.Orthogonal,
        )
        
        super(TestRNNLayer, self).build(input_shape)
        
        self.built = True

    def call(self, inputs, training=False ):
        
        tf.print('inputs:\n',inputs,'\nshape:',inputs.shape,'  dtype:',inputs.dtype,'\n')
        tf.print('state:\n', self.state ,'\nshape:', self.state.shape ,'  dtype:', self.state.dtype ,'\n')
        
        to_cell = tf.linalg.matmul( inputs , self.state )
#        tf.print('to_cell:\n', to_cell,'\nshape:', to_cell.shape,'  dtype:', to_cell.dtype,'\n')
        
#        state_T = tf.linalg.matrix_transpose( self.state )
#        tf.print('state_T:\n', state_T ,'\nshape:', state_T.shape ,'  dtype:', state_T.dtype ,'\n')
        
        to_out = tf.linalg.matmul( to_cell , self.output_kernel )
#        tf.print('to_out:\n', to_out,'\nshape:', to_out.shape,'  dtype:', to_out.dtype,'\n')
        
        return to_out



    

''' TESTING SECTION ''' 
if __name__ == "__main__":
    
    # Build Dataset
    
    batches=15
    features = 32
    out_size = 17
    
    mkygls = get_mackey_glass(steps=batches*features)[1::]
    
    inpt_data = [ mkygls[i:features+i] for i in range(len(mkygls)-features) ]
    inpt_data = np.asarray( inpt_data ).astype( dtype=np.float32 )[0:-32]
    inpt_data = np.expand_dims( inpt_data , 1 )
#    print( '\ninpt_data:\n' , inpt_data , inpt_data.shape)
    
    out_data = [ mkygls[i:out_size+i] for i in range( features, len(mkygls)-features) ]
    out_data = np.asarray( out_data ).astype( dtype=np.float32 )
    out_data = np.expand_dims( out_data , 1 )
    
    rem = inpt_data.shape[0] % batches
    if rem != 0:
        inpt_data[0:-rem]
        out_data[0:-rem]
    
    # Build Model
    inpt = tf.keras.Input( shape=(1, features), batch_size=batches , dtype=tf.dtypes.as_dtype(inpt_data.dtype) )
    test = TestRNNLayer( state_size = 12 , output_size = out_size )(inpt)
    model = tf.keras.Model( inputs=inpt , outputs=test )
    model.compile(
        optimizer=tf.keras.optimizers.legacy.RMSprop(
            learning_rate=0.001
        ),
        loss=tf.keras.losses.MeanSquaredError()
    )
    model.summary()
    
    # Train Model 
    model.fit(
        x=inpt_data,
        y=out_data,
        batch_size=batches
    )
    
    # Predict Model
    
    # Test Model










