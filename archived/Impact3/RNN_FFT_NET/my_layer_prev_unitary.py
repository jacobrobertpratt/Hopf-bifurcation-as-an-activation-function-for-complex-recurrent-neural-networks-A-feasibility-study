
import sys

import numpy as np
import scipy
import tensorflow as tf
from keras.utils import losses_utils

import pymanopt


''' '''
@tf.function
def check_unitary_mat(_str='',p_mat=None):
    
    if len(_str) > 0:
        tf.print('Check Unitary '+_str+':\n -> shape:',p_mat.shape,'\n -> rank:',tf.rank(p_mat))
    else:
        tf.print('Check Unitary Mat:\n -> shape:',p_mat.shape,'\n -> rank:',tf.rank(p_mat))
    
    if p_mat.shape[-1] == 1:
        _mat = np.eye(p_mat.shape[-2],dtype=complex)
        _mat = tf.convert_to_tensor(_mat)
        _mat = tf.cast(_mat,dtype=p_mat.dtype)
        _mat *= p_mat
    else:
        _mat = p_mat
    
    _det = tf.abs(tf.linalg.det(_mat))
    tf.print(' ----> det:',_det,'\n')
    
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
def print_both(name,arr):
    printComplex(name,arr,print_shape=True)
    check_unitary_mat(name,arr)
    
    
''' '''
def generate_unitary(input,units):

    # Convert to unitary differential manifold
    _nm1 = units - 1
    _reshape = tf.reshape(tf.cast(input,dtype=tf.float64),[units])
    _dy = tf.concat([[0],(_reshape[1::] - _reshape[0:_nm1])],0)
    _cpx = tf.complex(tf.math.cos(_dy),tf.math.sin(_dy))
    
    # Find unitary scale factor
    _div = tf.math.divide(tf.math.pow(_reshape,units),tf.reduce_prod(_reshape))
    _cpx *= tf.cast(tf.math.pow(_div,(1.0 / units)),dtype=_cpx.dtype)  # Mulitplies unitary _dy matrix with unitary scale factor
    return tf.reshape(_cpx,[units,1])



''' ---- LOSS FUNCTION ---- '''
class MyLoss(tf.keras.losses.Loss):

    ''' '''
    def __init_(self, reduction=tf.keras.losses.Reduction.NONE, name=None, **kwargs):
        super().__init_(reduction=reduction, name=name, **kwargs)

    ''' '''
    def call(self, true, pred):
        
        _pred = tf.reshape(tf.cast(pred,true.dtype),[pred.shape[-2]])
        _true = tf.reshape(true,[true.shape[-2]])
        
        #_mult_conj = tf.math.conj(_true) * _pred
        if (_true.dtype == tf.complex64) or (_true.dtype == tf.complex128):
            #tf.print('MyLoss -> IS COMPLEX VALUED')
            _diff = tf.math.exp(tf.math.log(_pred) - tf.math.log(_true))
        else:
            _diff = tf.math.abs(_pred - _true)  # L1 error
            #tf.print('MyLoss -> IS -NOT- COMPLEX VALUED')
            
        #printComplex('_exp_diff',_exp_diff)
        #check_unitary_mat('_exp_diff',tf.linalg.tensor_diag(_exp_diff))
        
        _out_loss = tf.reshape(_diff,true.shape)
        
        return _out_loss # Returns complex tensor as error

    
''' RNN CELL'''
class MyRNNCell(tf.keras.layers.Layer):
    
    ''' ''' 
    def __init__(self, p_units, **kwargs):
        
        super(MyRNNCell, self).__init__(**kwargs)
        
        self.units = p_units
    
    
    ''' '''
    def build(self, input_shape):
        
        super(MyRNNCell,self).build(input_shape=input_shape)

        init_ones = tf.keras.initializers.Ones()
        init_zeros = tf.keras.initializers.Zeros()
        
        # State(t) -> state(t+1) transition map
        self.A = tf.Variable(
                             initial_value = init_ones (
                                                        shape=(1,input_shape[-2],input_shape[-1]),
                                                        dtype=tf.complex128,
                                                        ),
                             dtype=tf.complex128,
                             trainable=True
                             )

        # Hidden state of manifold
        self.x = tf.Variable(
                             initial_value = init_ones (
                                                        shape=(1,input_shape[-2],input_shape[-1]),
                                                        dtype=tf.complex128,
                                                        ),
                             dtype=tf.complex128,
                             trainable=False
                             )
        
        self.B = tf.Variable(
                             initial_value = init_ones (
                                                        shape=(1,input_shape[-2],input_shape[-1]),
                                                        dtype=tf.complex128,
                                                        ),
                             dtype=tf.complex128,
                             trainable=True
                             )
        
        self.built = True


    @tf.custom_gradient
    def manifold_update(self, manifold, input):
        
        _manifold = tf.cast(manifold,dtype=tf.complex128)
        _input = tf.cast(input,dtype=tf.complex128)
        
        _map = tf.math.conj(_manifold) * _input # tf.math.exp(tf.math.log(_input) - tf.math.log(_manifold))
        
        def _grad(_dg):
            #print_both('MyRNNCell -> manifold_update:_dg',_dg)
            _cpx_dg = tf.cast(_dg,dtype=tf.complex128)
            _dm = tf.math.conj(_input) * _cpx_dg
            _di = tf.math.conj(_manifold) * _cpx_dg
            return _dm, _di
            
        return _map, _grad



    ''' '''
    def call(self, input, training=False):
        
        if training is False:
            pass
        
        # Cast input to be same type as it's mapped matrix B
        _input = tf.cast(input, self.x.dtype)
        
        # Update manifold transition based on input state (input a manifold transition)
        _A = self.A
        # print_both('_A', _A)
        
        # Update state
        _x = tf.math.conj(_A) * _input
        #print_both('x',_x)
        
        self.x.assign(_x)
        
        return _x


''' '''
class MyRNN(tf.keras.layers.Layer):
    
    ''' '''
    def __init__(self, units, **kwargs):
        
        super(MyRNN, self).__init__(**kwargs)

        self.units = units
        
        self.cell = MyRNNCell(self.units)
    
    
    ''' '''
    def build(self,input_shape):
        
        super(MyRNN,self).build(input_shape=input_shape)
        
        self.cell.build(input_shape[1::])
        
        init_ones = tf.keras.initializers.Ones()
        init_zeros = tf.keras.initializers.Zeros()
        init_eye = tf.cast(np.eye(self.units,dtype=complex),dtype=tf.complex128)
        
        self.A = tf.Variable(
                             initial_value = init_ones (
                                                         shape=(self.units,1),
                                                         dtype=tf.complex128,
                                                        ),
                             dtype=tf.complex128,
                             trainable=True
                            )
        
        self.B = tf.Variable(
                             initial_value = init_ones (
                                                         shape=(self.units,1),
                                                         dtype=tf.complex128,
                                                        ),
                             dtype=tf.complex128,
                             trainable=True
                            )

        self.F = tf.Variable(
                             initial_value = init_ones (
                                                         shape=(self.units),
                                                         dtype=tf.complex128,
                                                        ),
                             dtype=tf.complex128,
                             trainable=True
                            )

        # State is in tangent space
        self.x = tf.Variable(
                             initial_value = init_ones (
                                                         shape=(self.units,1),
                                                         dtype=tf.complex128,
                                                        ),
                             dtype=tf.complex128,
                             trainable=False
                            )

        # Counter used for training
        self.unit_window = tf.Variable(
                                        initial_value = init_zeros (
                                                                    shape=(self.units,self.units),
                                                                    dtype=tf.complex128,
                                                                    ),
                                        dtype=tf.complex128,
                                        trainable=False
                                        )
        
        # Counter used for training
        self.window_counter = tf.Variable(0, dtype=tf.int64, trainable=True)
        
        # To store last manifold for log-diff window calculations
        self.last_cell = tf.Variable( 
                                     initial_value = init_zeros(shape=(self.units), dtype=tf.complex128),
                                     dtype=tf.complex128,
                                     trainable=False 
                                     )
        
        # Reps. current exponential map of log-diff window
        self.Exp_window = tf.Variable( 
                                       initial_value = tf.cast(np.eye(self.units,dtype=complex),dtype=tf.complex128),
                                       dtype=tf.complex128,
                                       trainable=False 
                                      )
        init_eye = np.eye(self.units,dtype=complex)
        init_eye = np.roll(init_eye,1,axis=1)
        self.perm = tf.Variable( 
                                initial_value = tf.cast(init_eye,dtype=tf.complex128),
                                dtype=tf.complex128,
                                trainable=False 
                                )

        self.built = True
    
    ''' '''
    def generate_unitary(self, input):
        
        units = self.units
        
        # Convert to unitary differential manifold
        _nm1 = units - 1
        _reshape = tf.reshape(tf.cast(input,dtype=tf.float64),[units])
        _dy = tf.concat([[0],(_reshape[1::] - _reshape[0:_nm1])],0)
        _cpx = tf.complex(tf.math.cos(_dy),tf.math.sin(_dy))
        
        # Find unitary scale factor 
        _div = tf.math.divide(tf.math.pow(_reshape,units),tf.reduce_prod(_reshape))
        _cpx *= tf.cast(tf.math.pow(_div,(1.0 / units)),dtype=_cpx.dtype)  # Mulitplies unitary _dy matrix with unitary scale factor
        
        return tf.reshape(_cpx,[1,units,1])
    
    
    ''' '''
    def call(self, input, training=False):
        
        # Convert input to scaled unitary
        _unit = self.generate_unitary(input)
        #print_both('input',_unit)
        
        _rshp_unit = tf.reshape(_unit,[self.units])

        # Unitized FFT
        _unit_fft = tf.signal.fft(_rshp_unit)
#        _unit_fft = tf.math.divide(tf.math.pow(_unit_fft,self.units),tf.reduce_prod(_unit_fft))
#        _unit_fft = tf.math.pow(_unit_fft,(1.0 / self.units))
        _unit_fft = tf.reshape(_unit_fft,[self.units])
        _unit_fft = _unit_fft / tf.math.reduce_sum(_unit_fft) # Normalize over summation
#        printComplex('unit_fft',_unit_fft)
#        tf.print('red_prod:',tf.math.abs(tf.math.reduce_sum(_unit_fft)),'\n')
        
        _det_arr = []
        if self.window_counter >= self.units:
            
            # Convert input FFT to unitary
            _expm_list = []
            _Ewin_to_Log = tf.linalg.logm(self.Exp_window)
            for u in range(self.units):
                _conj = tf.math.conj(_Ewin_to_Log) * _unit_fft[u]
                _skew_symd = _conj - tf.transpose(_conj)
                _expm_list.append(tf.linalg.expm(_skew_symd))
            _exp_tens = tf.convert_to_tensor(_expm_list) # If summed expm(a) Determinant = 1
            #print_both('_exp_tens[0]',tf.matmul(_exp_tens[0],tf.math.conj(_exp_tens[0])))
            
            
            
#            _exp_dot = tf.math.reduce_sum(tf.math.conj(_exp_tens) * tf.math.log(_unit), axis=0)
#            _exp_T = tf.linalg.expm(_exp_dot)
#            _A = _exp_T - tf.transpose(_exp_T)
#            print_both('_A',tf.linalg.expm(_A)) # Guarenteed to be Unitary
            
#            for d in _det_arr:
#                printComplex('det',d)
            
            #_A = self.A * tf.transpose(tf.math.conj(self.Exp_window)) * _unit_fft
            #_B = tf.transpose(tf.math.conj(self.Exp_window)) * _unit_fft
            #print_both('_A', _A)

#            _AA = tf.linalg.expm(_A - tf.transpose(tf.math.conj(_A)))
#            print_both('_AA',_AA)
            
        else:
            _A = self.A
            _B = self.B
            #printComplex('test',_test[0][0])
            #tf.print('abs',tf.math.abs(_test[0]),'\n')
            #_prod = tf.math.reduce_prod(_test,axis=1)
            
            
        #print_both('self.A',self.A)
        #print_both('self.B',self.B)
        _unit = self.A * _unit    # This Works keep it.
        #print_both('_unit',_unit)
        
        #_unit_fft_rshp = tf.reshape(_unit_fft,[1,self.units,1])
        _cell = self.cell.call(_unit, training=training)
        _rshp_cell = tf.reshape(_cell,[self.units])
        
        if training is True:
            
            # Wait one iteration for last_cell to be set
            if self.window_counter > 0:
                
                _curr_log_diff = tf.math.log(_rshp_cell) - tf.math.log(_rshp_unit)
                self.unit_window.assign(tf.roll(self.unit_window,1,0))
                self.unit_window[0].assign(_curr_log_diff)
            
            # Assign last cell
            self.last_cell.assign(_rshp_cell)
            
            # Increment counter
            if self.window_counter < self.units:
                self.window_counter.assign_add(1)
            else:
                # Calcualte change manifold
                _expm_window = tf.linalg.expm(self.unit_window)
                _adj_expm = tf.matmul(tf.math.conj(self.Exp_window),_expm_window) # Use conj(Exp_window) so it doesn't explode,  don't use transpose (still explodes)
                self.Exp_window.assign(_adj_expm)
                
        
        #_cell_ifft = tf.signal.ifft(_cell)
        #printComplex('_cell_ifft',_cell_ifft)
        
        # Adjust input from manifold map to create output estimation
        _update = tf.cast(input,dtype=tf.float64) + tf.cast(tf.math.angle(_cell),dtype=tf.float64)
        
        return _update

    
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

        _input, _image = data
        
        ''' Predict, find loss, and calculate gradients '''
        with tf.GradientTape() as tape:
        
            #tf.print('\n-----------------------------------------------------------------------------------------------------\nPrediction:\n')
            
            #tf.print('\nMyModel -> input:\n', _input,'\nshape:',_input.shape,'  dypte:',_input.dtype,'\n')
            
            _pred = tf.cast(self(_input, training=True),dtype=_input.dtype)  # Forward pass

            #tf.print('\nMyModel -> pred:\n', _pred,'\nshape:',_pred.shape,'  dypte:',_pred.dtype,'\n')
            #tf.print('\nMyModel -> image:\n', _image,'\nshape:',_image.shape,'  dypte:',_image.dtype,'\n')
            
            # Can skip the built-in loss function entirely and just make your own, 
            #       no reason to use their rediculous wrapper.
            _loss = self.my_loss.call(_image, _pred)
            #tf.print('loss',_loss,'\n')

            #tf.print('\n-----------------------------------------------------------------------------------------------------\nBackward Gradient:\n')

            # Compute gradients 
            _vars = self.trainable_variables
            
            _grad = tape.gradient(_loss, _vars)
            #tf.print('MyModel:train_step -> Gradients:')
            #for _g in _grad:
            #    printComplex('MyModel:train_step -> _g', _g, print_shape=True)
            #    check_unitary_mat('MyModel:train_step -> _g',_g)


        ''' Update model with found gradients '''
        _wgts = self.trainable_weights
        #tf.print('Wgts POST-apply_gradients:')
        #for _w in _wgts:
        #    printComplex('MyModel:train_step -> _w', _w, print_shape=True)
        
        self.optimizer.apply_gradients(zip(_grad, _wgts))

        # Updated trainable weights
        #tf.print('Wgts POST-apply_gradients:')
        #for _w in _wgts:
        #    printComplex('MyModel:train_step -> _w', _w, print_shape=True)
        
        #tf.print('\n___________________________________________________END__________________________________________________\n\n')
        
        return {'loss':_loss}
    


''' FFT Stuff '''
#_input = np.zeros(shape=(self.units,1),dtype=float)
#_input = tf.convert_to_tensor(_input)
#_input = tf.cast(_input,dtype=tf.complex64)
#if (input.dtype is not tf.complex64) or (input.dtype is not tf.complex128):
#    _input = tf.complex(tf.math.cos(input),tf.math.sin(input))
#_input = tf.signal.fft(_input)
#_input -= tf.math.reduce_mean(_input)
#_input = tf.math.divide(_input, tf.math.l2_normalize(_input))

''' '''
# All Ones
# init_ones = tf.keras.initializers.Ones()
# init_zeros = tf.keras.initializers.Zeros()
# init_eye = tf.cast(np.eye(self.units,dtype=complex),dtype=tf.complex128)

# Create a standard complex grid from 0 -> 2*PI
#_rng1 = np.arange(0,2*np.pi,2*np.pi/self.units,dtype=float)
#_cos = np.cos(_rng1)
#_sin = np.sin(_rng1)
#_mesh = np.meshgrid(_cos,_sin)
#_cpx_ini = np.zeros(shape=(self.units,self.units),dtype=complex)
#_cpx_ini.real = _mesh[0]
#_cpx_ini.imag = _mesh[1]
#_cpx_ini = tf.convert_to_tensor(_cpx_ini,dtype=tf.complex64)

#self.M = tf.Variable(
#                     initial_value = _cpx_ini,
#                     dtype=tf.complex64,
#                     trainable=True
#                     )

#self.B = tf.Variable(
#                    initial_value = init_ones(shape=(self.units,self.units),dtype=tf.complex64),
#                     dtype=tf.complex64,
#                     trainable=True
#                     )
















