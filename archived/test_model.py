import os

import numpy as np

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


class TestRNNCell(tf.keras.layers.Layer):
    
    def __init__(
                    self,
                    units,
                    activation='relu',
                    **kwargs
                    ):
        super().__init__(**kwargs)
        
        # Internal hidden state size. Determines the recurrent kernel mapping size.
        self.units = units
        
        # The activations function checks if activation is a string, itterable, or callable, 
        #   and returns either the associated activation, itterable of activations, 
        #   or callable object. located in the keras module.
        self.activation = activation
        
        self.state_size = self.units
        self.output_size = self.units
        
    
    def build(self, input_shape):
        
        super().build(input_shape)
        
        if self.activation is 'relu':
            self.activation = tf.keras.activations.relu
        
        # Generate random seed.
        _2rnds = np.random.randint(1,high=65537,size=(2),dtype=int)
        
        glorot1 = tf.keras.initializers.GlorotNormal(_2rnds[0])
        self.kernel = self.add_weight(
                                       name="kernel",
                                       shape=(input_shape[-1], self.units),
                                       initializer=glorot1,
                                       trainable=True
                                      )

        glorot2 = tf.keras.initializers.GlorotNormal(_2rnds[1])
        self.recurrent_kernel = self.add_weight(
                                                 name="recurrent_kernel",
                                                 shape=(self.units, self.units),
                                                 initializer=glorot2,
                                                 trainable=False
                                                )
        
        self.built=True


    def call(self, inputs, states, training=None):
#        print('inputs.shape',inputs.shape)
        
        prev_output = states[0] if tf.nest.is_nested(states) else states
#        print('prev_output.shape',prev_output.shape)
        
        h = tf.tensordot(inputs,self.kernel,1)
#        print('h.shape',h.shape)
        
        # Apply recurrent mapping
        output = tf.tensordot(prev_output,self.recurrent_kernel,1)
        
        # Apply Summation to input mapping
        output = h + output
        
        # Apply activation function.
        if self.activation is not None: output = self.activation(output)
        
        # Check if it's nested and apply output.
        new_state = [output] if tf.nest.is_nested(states) else output
        
        return output, new_state


    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            dtype = inputs.dtype
        
        def create_zeros(unnested_state_size):
            flat_dims = tf.TensorShape(unnested_state_size).as_list()
            init_state_size = [batch_size] + flat_dims
            return tf.zeros(init_state_size,dtype=dtype)
            
        if tf.nest.is_nested(self.state_size):
            return tf.nest.map_structure(create_zeros, self.state_size)
        else:
            return create_zeros(self.state_size)


    def get_config(self):
        config = {
                    "units": self.units,
                    "activation": tf.keras.activations.serialize(self.activation)
                    }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))




@tf.keras.utils.register_keras_serializable('test_model')
class TestRNNLayer(tf.keras.layers.RNN):
    
    ''' TestRNNLayer '''
    def __init__(self, **kwargs):
        
        cell = TestRNNCell(units=25,activation='relu')
        
        super(TestRNNLayer, self).__init__(cell, **kwargs)

        self.input_spec = [tf.keras.layers.InputSpec(ndim=3)]


    ''' TestRNNLayer '''
    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super(TestRNNLayer, self).call(
                                                inputs,
                                                mask=mask,
                                                training=training,
                                                initial_state=initial_state
                                               )


    ''' TestRNNLayer '''
    @property
    def units(self):
        return self.cell.units


    ''' TestRNNLayer '''
    def get_config(self):
        base_config = super(TestRNNLayer,self).get_config()
        if 'cell' in base_config: del base_config['cell']
        return base_config

    ''' TestRNNLayer '''
    @classmethod
    def from_config(self, config):
        return self(**config)   # Same as implementation in SimpleRNN() Layer see: C:\Users\isuan\Desktop\Research\Lib\site-packages\tensorflow\python\keras\layers\recurrent.py


