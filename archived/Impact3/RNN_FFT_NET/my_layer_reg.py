
import sys

import numpy as np
import scipy
import tensorflow as tf
from keras.utils import losses_utils

import pymanopt


''' '''
def check_unitary_mat(_str='',p_mat=None):
    
    if p_mat.shape[-1] == 1:
        _mat = np.eye(p_mat.shape[-2],dtype=complex)
        _mat = tf.convert_to_tensor(_mat)
        _mat = tf.cast(_mat,dtype=p_mat.dtype)
        _mat *= p_mat
    else:
        _mat = p_mat
    
    _det = tf.abs(tf.linalg.det(_mat))
    tf.print(' ----> '+_str+' det:',_det,'\n')
    
    return _det


''' '''
def printComplex(name,arr,print_shape=True):
    tf.print('\n',name+':')
    tf.print('\treal:',tf.math.real(arr))
    tf.print('\timag:',tf.math.imag(arr))
    if print_shape:
        tf.print('shape:',arr.shape,'    dtype:',arr.dtype)
    tf.print('\n')


''' '''
def id_check(name,arr):
    
    if tf.rank(arr) < 2:
        tf.print(name+' ID - check: FAILED -> rank < 2/\n')
    else:
        _arr = tf.reshape(arr,[arr.shape[-2],arr.shape[-2]])
        _mult = tf.matmul(tf.transpose(tf.math.conj(_arr)),_arr) / arr.shape[-2]
        _id = tf.math.round(tf.cast(_mult,dtype=tf.float64))
        printComplex('ID Check'+name,_id)

''' '''
def print_both(name,arr):
    printComplex(name,arr,print_shape=True)
    check_unitary_mat(name,arr)

''' '''
@tf.custom_gradient
def print_gradient(_str,input):
    
    _str += ':'
    
    _nothing = input
    
    def grad(_dg):
        if _nothing.dtype is tf.complex128:
            print_both('(global) '+_str+'print_gradient -> _input',input)
            print_both('(global) '+_str+'print_gradient -> _out dg',_dg)
        else:
            tf.print('(global) '+_str+'print_gradient -> _out dg:\n',_dg,'\nshape:',_dg.shape,'\n')
        _di = input
        return _di, None
    
    return _nothing, grad

''' ---- LOSS FUNCTION ---- '''
class MyLoss(tf.keras.losses.Loss):

    ''' '''
    def __init_(self, reduction=tf.keras.losses.Reduction.NONE, name=None, **kwargs):
        super().__init_(reduction=reduction, name=name, **kwargs)
    
    ''' '''
    @tf.custom_gradient
    def call(self, true, pred):
        
        _pred = tf.cast(pred,true.dtype)
        _true = tf.reshape(true,_pred.shape)
#        tf.print('_true:\n',_true,'\nshape:',_true.shape,'  dtype:',_true.dtype,'\n')
#        tf.print('_pred:\n',_pred,'\nshape:',_pred.shape,'  dtype:',_pred.dtype,'\n')
        
        if (_true.dtype == tf.complex64) or (_true.dtype == tf.complex128):
            _loss = tf.math.exp(tf.math.log(_pred) - tf.math.log(_true))
        else:
            _loss = tf.math.abs(_pred - _true)
        
        def grad(_dg): # _dg returns a set of 1's in tf.float64 format
            _dt = tf.reshape(_loss,_dg.shape)
            _dp = tf.reshape(_loss,_dg.shape)
            return _dt, _dp
        
#        tf.print('_loss:\n',_loss,'\n')
        
        return _loss, grad # Returns complex tensor as error


''' RNN CELL'''
class MyRNNCell(tf.keras.layers.Layer):
    
    ''' ''' 
    def __init__(self,**kwargs):
        super(MyRNNCell, self).__init__(**kwargs)
        self.units = 8
    
    ''' '''
    def build(self, input_shape):
        
        super(MyRNNCell,self).build(input_shape=input_shape)
        
        init_ones = tf.keras.initializers.Ones()
        init_zeros = tf.keras.initializers.Zeros()

        '''
        self.B = tf.Variable(
                            initial_value = init_ones(shape=(self.units,self.units),dtype=tf.complex64),
                             dtype=tf.complex64,
                             trainable=True
                             )
        '''
        
        self.built = True
    
    ''' '''
    def call(self, input, training=False):
        
        _input = tf.reshape(input,[1])
        
        
        
        return input


''' '''
class MyRNN(tf.keras.layers.Layer):
    
    ''' '''
    def __init__(self,**kwargs):
        
        super(MyRNN, self).__init__(**kwargs)
        
        self.cell = MyRNNCell(dtype=self.dtype)
        
        
    ''' '''
    def build(self,input_shape):
        
        super(MyRNN,self).build(input_shape=input_shape)
        
        self.cell.build(input_shape)
        
        self.built = True
    
    ''' '''
    def call(self, input, training=False):
        
        ret = self.cell.call(input)
        
        return ret

    
''' '''
class MyModel(tf.keras.Model):

    ''' '''
    def __init__(self, input, output):
        
        super(MyModel, self).__init__(input, output)
        
        self.my_loss = MyLoss()

    
    ''' '''
    def train_step(self, data):
        
        to_print = False
        
        #tf.print('\n___________________________________________________START__________________________________________________\n')

        input, image = data
        _input = tf.cast(input,dtype=tf.float64)
        _image = tf.cast(image,dtype=tf.float64)
        
        with tf.GradientTape() as tape:
        
            #tf.print('\n-----------------------------------------------------------------------------------------------------\nPrediction:\n')
            
            #tf.print('\nMyModel -> input:\n', _input,'\nshape:',_input.shape,'  dypte:',_input.dtype,'\n')
            
            _pred = tf.cast(self(_input, training=True),dtype=_input.dtype)  # Forward pass

            tf.print('\nMyModel -> pred:\n', _pred,'\nshape:',_pred.shape,'  dypte:',_pred.dtype,'\n')
            tf.print('\nMyModel -> image:\n', _image,'\nshape:',_image.shape,'  dypte:',_image.dtype,'\n')
            
            # Can skip the built-in loss function entirely and just make your own, 
            #       no reason to use their rediculous wrapper.
            #_loss = self.my_loss.call(_image, _pred)
            _loss = self.compiled_loss(_image,_pred)
            #tf.print('loss',_loss,'\n')
            
            #tf.print('\n-----------------------------------------------------------------------------------------------------\nBackward Gradient:\n')

            # Compute gradients 
            _vars = self.trainable_variables
#            for v in _vars:
#                printComplex('MyModel:train_step -> trainable_variables',v)
            
            _grad = tape.gradient(_loss, _vars)
            #tf.print('MyModel:train_step -> Gradients:\n')
            #for g in _grad:
            #    printComplex('MyModel:train_step -> Gradients:\n',g)
                

        _wgts = self.trainable_weights
        #tf.print('Wgts trainable_weights:\n',_wgts,'\n')
        
        self.optimizer.apply_gradients(zip(_grad, _wgts))

        # Updated trainable weights
        #tf.print('Wgts POST-apply_gradients:')
        #for _w in _wgts:
        #    printComplex('MyModel:train_step -> _w', _w, print_shape=True)
        
        #tf.print('\n___________________________________________________END__________________________________________________\n\n')
        
        return {'loss':_loss}

# init_ones = tf.keras.initializers.Ones()
# init_zeros = tf.keras.initializers.Zeros()

#self.B = tf.Variable(
#                    initial_value = init_ones(shape=(self.units,self.units),dtype=tf.complex64),
#                     dtype=tf.complex64,
#                     trainable=True
#                     )















