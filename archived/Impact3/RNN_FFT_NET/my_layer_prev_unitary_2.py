
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
def print_gradient(input,_str):
    
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
            _dt = tf.reshape(_diff,_dg.shape)
            _dp = tf.reshape(_diff,_dg.shape)
            return _dt, _dp
        
#        tf.print('_loss:\n',_loss,'\n')
        
        return _loss, grad # Returns complex tensor as error


''' RNN CELL'''
class MyRNNCell(tf.keras.layers.Layer):
    
    ''' ''' 
    def __init__(self,
                 units,
                 **kwargs):
        
        super(MyRNNCell, self).__init__(**kwargs)
        
        self.units = units
        
        self.kernel_initilizer = tf.keras.initializers.GlorotNormal()
        
        self.state_size = 32

    
    ''' '''
    def build(self, input_shape):
        
        super(MyRNNCell,self).build(input_shape=input_shape)
        
        # Used to reverse the unitary calculation
        self.unit_prod = None
        
        init_ones = tf.keras.initializers.Ones()
        init_zeros = tf.keras.initializers.Zeros()

        # roots are the unitary roots of the unit circle given the internal state size
        rng = np.arange(self.state_size)
        roots = np.exp(-2j * 2 * np.pi * rng / self.state_size)
        self.roots = tf.constant(roots,dtype=tf.complex128)
        
        dft = roots.reshape(-1,1) ** rng.reshape(1,-1)
        self.dft = tf.constant(dft.T,dtype=tf.complex128)
        
        self.dends = tf.Variable(
                                 initial_value = init_ones(
                                                           shape=(self.state_size),
                                                           dtype=tf.complex128,
                                                           ),
                                 dtype=tf.complex128,
                                 trainable=False
                                 )
        
        self.tang_hist = tf.Variable(
                                     initial_value = init_ones(
                                                               shape=(self.state_size,self.state_size),
                                                               dtype=tf.complex128,
                                                               ),
                                     dtype=tf.complex128,
                                     trainable=True
                                     )
                                     
        self.built = True
    
    ''' '''
    def call(self, input, training=False):
        
        _fft_wave = input[0]
#        printComplex('fft_wave',_fft_wave)
#        tf.print(tf.math.abs(tf.math.conj(_fft_wave)*_fft_wave))
        
        _input = tf.reshape(input[1::],[self.state_size])
#        printComplex('_input',_input)
        
        _input_vs_dends = _input / self.dends
#        printComplex('_input_vs_dends',_input_vs_dends)
        
        # Combine input signals and cell state
        # ... 
        
        # Update state values
        # ...
        
        # Degrade state values over time if needed.
        # ... 
        
        # Activate state base on new cell state
        # ... 
        
        # Map activation to output synaptic weights (syns will be learned ...)
        # ... 
        
        self.dends.assign(_input_vs_dends)
        
        return input #tf.reshape(input,input.shape)


''' '''
class MyRNN(tf.keras.layers.Layer):
    
    ''' '''
    def __init__(self, units, **kwargs):
        
        super(MyRNN, self).__init__(**kwargs)

        self.units = units
        self.state_size = 32
        self.hist_size = 32
        self.cell = MyRNNCell(self.units)
    
    ''' '''
    def build(self,input_shape):
        
        super(MyRNN,self).build(input_shape=input_shape)
        
        self.cell.build(input_shape[1::])
        
        init_ones = tf.keras.initializers.Ones()
        init_zeros = tf.keras.initializers.Zeros()
        
        self.counter = tf.Variable([0],dtype=tf.float32,trainable=False)
        
        self.last_y = tf.Variable([0],dtype=tf.float64,trainable=False)
        
        self.state = tf.Variable(
                                 initial_value=init_ones(
                                                         shape=(self.state_size),
                                                         dtype=tf.complex128,
                                                         ),
                                 dtype=tf.complex128,
                                 trainable=False
                                 )

        self.y_hist = tf.Variable(
                                  initial_value=init_ones(
                                                          shape=(self.state_size),
                                                          dtype=tf.float64,
                                                          ),
                                  dtype=tf.float64,
                                  trainable=False
                                  )
        
        self.dy_hist = tf.Variable(
                                   initial_value=init_ones(
                                                            shape=(self.state_size),
                                                            dtype=tf.float64,
                                                            ),
                                   dtype=tf.float64,
                                   trainable=False
                                   )
        
        rng = np.exp(-2j * 2 * np.pi * np.arange(self.state_size) / self.state_size)
        self.rng = tf.constant(rng,dtype=tf.complex128)
        
        dft = rng.reshape(-1,1) ** np.arange(1,self.state_size+1).reshape(1,-1)
        self.dft = tf.constant(dft,dtype=tf.complex128)
        
        self.built = True

    def print_cpx_inform(self,_str, _cpx):
        printComplex(_str,_cpx)
        tf.print(_str+' abs:\n',tf.math.abs(_cpx))
        tf.print(_str+' angle:\n',tf.math.angle(_cpx))
        tf.print(_str+' -> reduce_sum:',tf.math.reduce_sum(tf.math.abs(_cpx)),'  reduce_prod:',tf.math.reduce_prod(tf.math.abs(_cpx)),'\n')
        
    def print_real_inform(self,_str, _input):
        tf.print(_str,_input)
        tf.print(_str+' -> reduce_sum:',tf.math.reduce_sum(tf.math.abs(_input)),'  reduce_prod:',tf.math.reduce_prod(tf.math.abs(_input)),'\n')
    
    ''' '''
    def call(self, input, training=False):
        
        tf.print('\n\n\n\n------------------------------------------------------------------------------\n',self.counter[0],
                    '\n------------------------------------------------------------------------------\n')
 
        _input = tf.reshape(input,[1])

        ''' MAP INPUT TO NORMED PRODUCT SPACE '''
        
        # Get input and add to input history
        _y = _input
        
        _y_hist = self.y_hist + _y
        self.y_hist.assign(tf.roll(_y_hist,1,0))
        self.y_hist[0].assign(_y[0])
#        tf.print('self.y_hist:\n',self.y_hist,'\n')
        
        _hist_adj = tf.math.reduce_sum(self.y_hist) - self.y_hist
        _hist_prod = tf.math.reduce_prod(_hist_adj)
        _hist_norm = _hist_adj / (_hist_prod ** (1.0 / self.state_size))
#        _cy = tf.cast(tf.math.exp(_hist_norm),dtype=tf.complex128)
#        _cy = tf.complex(tf.math.cos(_hist_norm),tf.math.sin(_hist_norm))
        _cy = tf.cast(_hist_norm,dtype=tf.complex128)
#        self.print_cpx_inform('_cy',_cy)
        
        # Get change of input and add to input history
        _dy = _y - self.last_y
        
        _dy_hist = self.dy_hist + _dy
        self.dy_hist.assign(tf.roll(_dy_hist,1,0))
        self.dy_hist[0].assign(_dy[0])
#        tf.print('self.dy_hist:\n',self.dy_hist,'\n')
        
#        _dy_hist_min = tf.math.reduce_min(self.dy_hist)
#        _dy_hist_adj = self.dy_hist - _dy_hist_min
#        _dy_hist_max = tf.math.reduce_max(_dy_hist_adj)
#        _dy_hist_adj = _dy_hist_adj / _dy_hist_max
#        tf.print('_dy_hist_adj:\n',_dy_hist_adj,'\n')
        
        _cdy = (self.dy_hist * (2 * np.pi)) % (2 * np.pi)
        _cdy = tf.complex(tf.math.cos(_cdy), tf.math.sin(_cdy))
#        self.print_cpx_inform('_cdy',_cdy)
        
        _cpx = _cy * _cdy
#        self.print_cpx_inform('_cpx',_cpx)
        
        # Dampen state history ... Maybe (i.e., reduce the older input values in the history by some function f(*))
        # ... 
        
        # Adjust state Values to be unitary ... if needed (i.e., want to keep the same level of energy ... prefer to have area under curve = 0.0)
        _state = tf.math.exp(tf.math.log(self.state) + tf.math.log(_cpx))       # Can apply alpha & beta weights here
        self.print_cpx_inform('A state',_state)
        
        ''' STATE TO CELL MAPPING '''
        
        # Determine which parts of the state map to the cell (some type of permutation or change matrix)
        _fft = tf.signal.fft(_cpx)
        _fft /= tf.math.reduce_sum(_fft)
        self.print_cpx_inform('_fft',_fft)
        tf.print(tf.math.reduce_sum(tf.math.abs(_fft)))
        
        # Map to cell(s)
        _cell_input = tf.concat([[_fft[0]],_state],0)
        _cell = self.cell.call(_cell_input, training=training)
#        self.print_cpx_inform('cell',_cell)

        ''' CELL TO STATE MAPPING 
                Cell(s) return values and remap to, and update, the state value to represent 
                their filtered/corrected value. '''
        # Take current state of network and update based on cell(s) response(s)
        _state = tf.math.exp(tf.math.log(_state) - tf.math.log(_cpx))
        self.print_cpx_inform('B state',_state)
        
        ''' UPDATE STATE VALUES AND HISTORY'''
        
        self.state.assign(_state)

        self.last_y.assign(_y)

        self.counter.assign_add([1.0])
        
        # Map back to input-dtype values
        _out_ang = tf.math.angle(_state)
        _out = self.y_hist * tf.math.abs(_state) * (tf.math.cos(_out_ang) + tf.math.sin(_out_ang))
#        tf.print('out',_out,'\n')
        
        return tf.reshape(_out[0],[1,1,1])

    
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
            #_grad = tape.gradient(_loss, _vars)
            #tf.print('MyModel:train_step -> Gradients:')
            #for _g in _grad:
            #    if _g is not None:
            #        printComplex('MyModel:train_step -> _g', _g,print_shape=True)
            #        check_unitary_mat('MyModel:train_step -> _g', _g)
                

        ''' Update model with found gradients '''
        _wgts = self.trainable_weights
        #tf.print('Wgts POST-apply_gradients:')
        #for _w in _wgts:
        #    printComplex('MyModel:train_step -> _w', _w, print_shape=True)
        
        #self.optimizer.apply_gradients(zip(_grad, _wgts))

        # Updated trainable weights
        #tf.print('Wgts POST-apply_gradients:')
        #for _w in _wgts:
        #    printComplex('MyModel:train_step -> _w', _w, print_shape=True)
        
        #tf.print('\n___________________________________________________END__________________________________________________\n\n')
        
        return {'loss':_loss}
    

''' '''
def map_input_to_state(self, input):

    units = self.hist_size

    # Convert to unitary differential manifold
    _nm1 = units - 1
    _reshape = tf.reshape(tf.cast(input,dtype=tf.float64),[units])
    _dy = tf.concat([[0],(_reshape[1::] - _reshape[0:_nm1])],0)
    _cpx = tf.complex(tf.math.cos(_dy),tf.math.sin(_dy))
    
    # Find unitary scale factor
    _prod = tf.reduce_prod(_reshape)
    _div = tf.math.divide(tf.math.pow(_reshape,units),_prod)
    _cpx *= tf.cast(tf.math.pow(_div,(1.0 / units)),dtype=_cpx.dtype)  # Multiplies unitary _dy matrix with unitary scale factor
    
    return tf.reshape(_cpx,[units,1])

def generate_unitary(self, input): # TESTED
    
    _input = tf.reshape(input,[self.units])
    _input = tf.cast(_input,dtype=tf.float64)
    
    _cpx = tf.complex(tf.math.cos(_input),tf.math.sin(_input))
    self.unit_prod = tf.reduce_prod(_cpx)
    _cpx = _cpx / self.unit_prod
    
    return tf.reshape(_cpx, [1,self.units,1])

def revert_unitary(self, input): # TESTED
    
    _input = tf.reshape(input,[self.units])
    
    _cpx = _input * self.unit_prod
    _out = tf.math.abs(_cpx) * tf.math.angle(_cpx)
    
    return tf.reshape(_out,[1,self.units,1])

def revert_fouier(self, fft):
    
    # Cast and check inputs
    _fft = tf.reshape(fft,[self.units])
    _fft *= self.fft_sum
    _ifft = tf.signal.ifft(_fft)
    
    return tf.reshape(_ifft,[1,self.units,1])

def map_to_state(self, input):
    
    #tf.print('self.dft_weights:\n',self.dft_weights,'\n')
    #tf.print('input:\n',input,'\n')
    
    ''' Cast input for later multiplication '''
    _input = tf.cast(input,dtype=tf.complex128)
    
    ''' Construct DFT matrix (n x n)
        - self.root is initialized as the roots of unity given the input 'root_size', (if root_size is None, root_size=self.units)
        - self.root_weights is the adjustment from the previous _root mapping. 
            Both values should stay unitary. (Look into adjusting their values to represent the appropriate mapping from input to signal space) '''
    _root = tf.math.conj(self.root_weights) * self.root
    self.root.assign(_root)
    
    ''' Construct the sudo-DFT matrix that represents the mapping from input space to my signal space (This defines how the input is mapped to this cell)
        - _root is provided from above
        - self.rng is a constant value constructed in build() used to expand _root into a Vandermonde matrix (represents the sudo-DFT) '''
    _dft = tf.reshape(_root,[self.root_size,1]) ** self.rng
    #printComplex('dft',_dft)
    #id_check('MyRNNCell:map_to_state -> dft',_dft)
    
    tf.print('\n-----------------------------------------------------------------------------------------------------\nFrame:',self.cnt[0],'\n')
    
    ''' Map the input values to our 'state-space' (i.e., sudo-DFT space, i.e., sudo-frequency space)
        - input is cast to a complex128 dtype but with only real values.
        '''
    _fft = tf.matmul(_dft, tf.reshape(_input,[self.units,1]))
    _fft /= tf.reduce_sum(_fft)
    
    printComplex('_fft', _fft, print_shape=True)
    #tf.print('_fft sum:',tf.math.abs(tf.reduce_sum(_fft)),'\n')
    
    printComplex('self.x', self.x)
    #tf.print('self.x sum:',tf.math.abs(tf.reduce_sum(self.x)),'\n')
    
    ''' Determine what signals mapped from the input contribute to the cells internal signal '''
#        _numor = (_fft * tf.math.conj(self.x))
#        _denom = (tf.math.conj(self.x) * self.x)
#        _u = tf.math.abs(_numor / _denom)
    _u =  tf.math.abs(self.x / _fft)
#        tf.print('_u:',_u,'\n')
    printComplex('_u:',_u)
    
    if self.cnt == 128:
        self.x.assign(tf.reshape(_fft,[1,self.units,1]))
        
    self.cnt.assign_add([1])            
        
    return tf.reshape(_fft,[1,self.units,1])


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

'''
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
'''

''' ------------------------------------------------------------------------------------------------------------- '''
''' TESTING SUDO CELL OUTPUT '''

#_rng = np.ones(shape=32,dtype=float)
#_sudo_cell = 0.9 * _rng

#_sudo_mag = np.random.rand((self.state_size))
#np.random.shuffle(_sudo_mag)
#_sudo_mag = tf.cast(_sudo_mag,dtype=tf.float64)

#_sudo_mag = tf.math.reduce_sum(_sudo_mag) - _sudo_mag
#_sudo_prod = tf.math.reduce_prod(_sudo_mag)
#_sudo_norm = _sudo_mag / (_sudo_prod ** (1.0 / self.state_size))
#_prod = tf.cast(_sudo_norm,dtype=tf.complex128)
#printComplex('sudo prod',_prod)
#tf.print('SUDO -> _prod:\nreduce_sum:',tf.math.reduce_sum(tf.math.abs(_prod)),'  reduce_prod:',tf.math.abs(tf.math.reduce_prod(_prod)),'\n')

#_sudo_cell =  tf.complex(np.cos(_sudo_cell),np.sin(_sudo_cell)) * _prod

#printComplex('_sudo_cell',_sudo_cell)
#tf.print('_sudo_cell abs:\n',tf.math.abs(_sudo_cell))
#tf.print('_sudo_cell angle:\n',tf.math.angle(_sudo_cell))
#tf.print('_sudo_cell -> reduce_sum:',tf.math.reduce_sum(tf.math.abs(_sudo_cell)),'  reduce_prod:',tf.math.abs(tf.math.reduce_prod(_sudo_cell)),'\n')

#_log_sudo = tf.math.log(_sudo_cell)
#_log_state = tf.math.log(_state)
#printComplex('Log _log_sudo',_log_sudo)
#printComplex('Log _log_state',_log_state)

#_log_sudo_mult = _log_sudo
#_log_state_mult = _log_state
#printComplex('Log _log_sudo_mult',_log_sudo_mult)
#printComplex('Log _log_state_mult',_log_state_mult)

#_summed = _log_state_mult + _log_sudo_mult
#printComplex('Log _summed',_summed)

#_state = tf.math.exp(_summed)

''' ------------------------------------------------------------------------------------------------------------- '''


'''




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





#@tf.custom_gradient
def map_unitary_weights(self, out_wgts, input):
    
    _wgts = tf.reshape(out_wgts,[self.units])
    _input = tf.reshape(input,[self.units])
    
    _out = tf.math.conj(_input) * _wgts
    
    def grad(_dg):
        #print_both('map_unitary_weights -> dg',_dg)
        _dw = input
        _ds = out_wgts
        return _dw, _ds
    
    return tf.reshape(_out,input.shape) #, grad

#@tf.custom_gradient
def add_log_and_map(self, Ax, Bu):
    
    _x = tf.math.exp(tf.math.log(Ax) + tf.math.log(Bu))
    
    def grad(_dg):
        print_both('add_log_and_map -> _dg',_dg)
        _dAx = Bu
        _dBu = Ax
        return _dAx, _dBu
    
    return _x #, grad



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


'''














