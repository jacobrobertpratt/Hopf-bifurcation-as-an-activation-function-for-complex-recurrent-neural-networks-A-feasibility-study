
import sys

import numpy as np
import scipy
import tensorflow as tf
import tensorflow_probability as tfp

from tfdiffeq import odeint
from tfdiffeq import plot_phase_portrait, plot_vector_field, plot_results

from keras.utils import losses_utils



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
        
        return _loss, grad # Returns complex tensor as error


''' RNN CELL'''
class MyRNNCell(tf.keras.layers.Layer):
    
    ''' ''' 
    def __init__(self, state_size, cell_id=0, **kwargs):
        
        super(MyRNNCell, self).__init__(**kwargs)
        
        self.cell_id = cell_id
        
        self.state_size = state_size
        
        self.freq = self.cell_id / self.state_size
    
    
    ''' '''
    def build(self, input_shape):
        
        super(MyRNNCell,self).build(input_shape=input_shape)
        
        init_ones = tf.keras.initializers.Ones()
        init_zeros = tf.keras.initializers.Zeros()
        init_glorot = tf.keras.initializers.GlorotUniform(seed=None)
        init_eye = tf.keras.initializers.Identity()
        
        do_wgt_train = True
        
        # Complex valued phase and radius state (for each cell)
        self.x = tf.Variable(
                              initial_value=init_ones(
                                                      shape=[1,self.state_size],
                                                      dtype=tf.complex128,
                                                      ),
                              name='x_c'+str(self.cell_id),
                              dtype=tf.complex128,
                              trainable=False
                             )
        
        '''
        self.mu = tf.Variable(
                                initial_value=init_ones(
                                                        shape=[1,self.state_size],
                                                        dtype=tf.float64,
                                                        ),
                                name='mu_c'+str(self.cell_id),
                                dtype=tf.float64,
                                trainable=do_wgt_train
                               )
        '''
        
        self.mu2 = tf.Variable(
                                initial_value=init_ones(
                                                        shape=[1,self.state_size],
                                                        dtype=tf.float64,
                                                        ),
                                name='mu2_c'+str(self.cell_id),
                                dtype=tf.float64,
                                trainable=False
                               )
        
        self.A = tf.Variable(
                             initial_value=init_glorot(
                                                       shape=[self.state_size,self.state_size],
                                                       dtype=tf.float64,
                                                       ),
                             name='A_c'+str(self.cell_id),
                             dtype=tf.float64,
                             trainable=do_wgt_train
                             )
        
        self.B = tf.Variable(
                             initial_value=init_glorot(
                                                       shape=[self.state_size,self.state_size],
                                                       dtype=tf.float64,
                                                       ),
                             name='B_c'+str(self.cell_id),
                             dtype=tf.float64,
                             trainable=do_wgt_train
                             )
                             
        '''
        self.C = tf.Variable(
                             initial_value=init_glorot(
                                                       shape=[self.state_size,self.state_size],
                                                       dtype=tf.float64,
                                                       ),
                             name='C_c'+str(self.cell_id),
                             dtype=tf.float64,
                             trainable=do_wgt_train
                             )
        
        self.D = tf.Variable(
                             initial_value=init_glorot(
                                                       shape=[self.state_size,self.state_size],
                                                       dtype=tf.float64,
                                                       ),
                             name='D_c'+str(self.cell_id),
                             dtype=tf.float64,
                             trainable=do_wgt_train
                             )
        '''
        
        self.theta = tf.Variable(
                                  initial_value=[[-2*np.pi*self.freq]],
                                  name='theta_c'+str(self.cell_id),
                                  dtype=tf.float64,
                                  trainable=False
                                 )
        
        self.ones = tf.constant(np.ones(shape=(1,self.state_size)),dtype=tf.float64)
        
        self.counter = tf.Variable([0],dtype=tf.float64,trainable=False)
        
        self.built = True
    
    @tf.function
    def odefunc_limit_cycle(self,t,x,w,mu):
        r, p = tf.unstack(x)
        dr_dt = r * (mu - (r * r))              # mu values is squared relative to radius
        dp_dt = w * self.ones
        return tf.stack([dr_dt,dp_dt])

    @tf.function
    def odefunc(self,t,x,w,mu):
        ode = x
        for j in range(6):
            ode = self.odefunc_limit_cycle(0,ode,w,mu)
        return ode

    @tf.function
    def map_to_limit_cycle(self,x_0,input,mu,wgtA,wgtB):
        
        t_0 = self.counter[0]
        t_1 = self.counter[0] + 1.0
        
        r_0 = tf.math.abs(x_0)
        p_0 = tf.math.angle(x_0)
        
#        r_i = tf.keras.activations.tanh(tf.matmul((input * tf.math.cos(input)),wgtB))
#        p_i = tf.keras.activations.tanh(tf.matmul((input * tf.math.sin(input)),wgtD))

        r_i = tf.matmul((input * tf.math.cos(input)),wgtA)
        p_i = tf.matmul((input * tf.math.sin(input)),wgtB)
        
        _state = tf.stack([r_i,p_i])
        
        _mu = tf.keras.activations.relu(mu)
        
#        ode = self.odefunc(0,_state,self.theta,_mu)
        
        # TESTING ... 
        
        t = tf.linspace(t_0,t_1,num=10)
        res = tfp.math.ode.DormandPrince().solve(self.odefunc_limit_cycle,t_0,_state,t,constants={'w':self.theta,'mu':mu})
        _ode = res.states[-1]
#        tf.print('_ode:\n',_ode,'\nshape:',_ode.shape,'\n')
#        _ode = _ode[0]
#        tf.print('_ode[0]:\n',_ode,'\nshape:',_ode.shape,'\n')
        
        dr_dt, dp_dt = tf.unstack(_ode)
#        tf.print('_dr_dt:\n',_dr_dt,'\nshape:',_dr_dt.shape,'\n')
#        tf.print('_dp_dt:\n',_dp_dt,'\nshape:',_dp_dt.shape,'\n')

        # ... TESTING.
        
#        dr_dt, dp_dt = tf.unstack(ode)
        
        r_1 = dr_dt
        p_1 = dp_dt

        x_1 = tf.complex(r_1 * tf.math.cos(p_1), r_1 * tf.math.sin(p_1))
        
        return x_1
    
    ''' '''
    def call(self, input, training=False):
        
        _input = tf.reshape(input,[1,self.state_size])
        
        '''
        if training is True:
            x_t = self.map_to_limit_cycle_training(self.x,_input,self.mu)
        else:
            x_t = self.map_to_limit_cycle(self.x,_input,self.mu)
        '''
        
        x_t = self.map_to_limit_cycle(self.x,_input,self.mu2,self.A,self.B)
        
        # Update non-trainable variables
        self.x.assign(x_t)
        self.counter.assign_add([1.0])

        ang = tf.math.angle(x_t)
        out = tf.math.abs(x_t) * (tf.math.cos(ang) + tf.math.sin(ang))
        
        return out


''' '''
class MyRNN(tf.keras.layers.Layer):
    
    ''' '''
    def __init__(   self,
                    state_size=32,
                    cell_count=1,
                    **kwargs):
        
        super(MyRNN, self).__init__(**kwargs)

        self.state_size = state_size
        
        # Creates state_size number of RNNCells with each cell being state_size large.
#        self.cells = [MyRNNCell(self.state_size,cell_id=j,name='rnncell_'+str(j)) for j in range(self.state_size)]
#        self.cell = MyRNNCell(self.state_size,cell_id=0,name='rnncell_'+str(0))
        self.cell = MyRNNCell(self.state_size,cell_id=1,name='rnncell_'+str(1))
#        self.cell = MyRNNCell(self.state_size,cell_id=16,name='rnncell_'+str(16))
    
    
    ''' '''
    def build(self,input_shape):
        
        super(MyRNN,self).build(input_shape=input_shape)
        
#        for cell in self.cells:
#            cell.build(input_shape[1::])
        
        self.cell.build(input_shape[1::])
        
        self.counter = tf.Variable([0],dtype=tf.float32,trainable=False)
        
        self.built = True

    
    ''' '''
    def call(self, input, training=False):
        
#        tf.print('\n------------------------------------------------------------------------------------------\n',self.counter)
#        tf.print('input:\n',input,'\n')
        
        ''' INPUT TO COMPLEX FUNCTION '''
        _input = tf.reshape(input,[1,self.state_size])
#        tf.print('_input:\n',_input,'\n')
        
        ''' MAP INPUT TO CELLS '''
#        cells = self.map_to_cells(input,training=training)
#        tf.print('cells',cells,'\n')
        
        ''' MAP CELL OUTPUT BACK TO STATE '''
#        out = tf.math.reduce_mean(cells,axis=1)
#        tf.print('out',out,'\n')
        
        cell_res = self.cell.call(_input,training=training)
        
        ''' CELL RESULTS TO OUTPUT VALUE(S) '''
        # ... 
        
        ''' UPDATE AUXILIARY VARIABLES '''
        self.counter.assign_add([1.0])
        
        return tf.reshape(cell_res,[1,1,self.state_size])

    
''' '''
class MyModel(tf.keras.Model):

    ''' '''
    def __init__(self, input, output):
        
        super(MyModel, self).__init__(input, output)
        
        self.my_loss = MyLoss()

    
    ''' '''
    def train_step(self, data):
        
        #tf.print('\n___________________________________________________START__________________________________________________\n')

        input, image = data
        _input = tf.cast(input,dtype=tf.float64)
        image = tf.cast(image,dtype=tf.float64)
        
        with tf.GradientTape() as tape:
        
            #tf.print('\n-----------------------------------------------------------------------------------------------------\nPrediction:\n')
            
            #tf.print('\nMyModel -> input:\n', _input,'\nshape:',_input.shape,'  dypte:',_input.dtype,'\n')
            
            pred = tf.cast(self(_input, training=True),dtype=_input.dtype)  # Forward pass

            #tf.print('\nMyModel -> pred:\n', pred,'\nshape:',pred.shape,'  dypte:',pred.dtype,'\n')
            #tf.print('\nMyModel -> image:\n', image,'\nshape:',image.shape,'  dypte:',image.dtype,'\n')
            
            # Can skip the built-in loss function entirely and just make your own, 
            #       no reason to use their rediculous wrapper.
            #_loss = self.my_loss.call(_image, _pred)
            _loss = self.compiled_loss(image,pred)
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




''' EXTRA STUFF '''

'''
@tf.function
def map_to_cells(self, input_map, training=False):
    
    state_size = tuple(c.state_size for c in self.cells)
    nest_cells = tf.nest.pack_sequence_as(state_size,tf.nest.flatten(self.cells))
    
    new_cells = []
    for cell, nest in zip(self.cells,nest_cells):
        new_cell = cell.call(input_map,training=training)
        new_cells.append(new_cell)
    _new_map = tf.convert_to_tensor(new_cells)
    
    return _new_map
'''

#### init
'''
self.x = tf.Variable(
                      initial_value=init_ones(
                                               shape=(self.state_size,1),
                                               dtype=tf.float64,
                                              ),
                       name='x_c'+str(self.cell_id),
                       dtype=tf.float64,
                       trainable=False
                      )

init_glorot_A = init_glorot(shape=(self.state_size,self.state_size),dtype=tf.float64)
#        init_glorot_A -= tf.math.reduce_mean(init_glorot_A)
self.A = tf.Variable(
                      initial_value=init_glorot_A,
                       name='wgt_A_c'+str(self.cell_id),
                       dtype=tf.float64,
                       trainable=True
                      )

init_glorot_B = init_glorot(shape=(self.state_size,self.state_size),dtype=tf.float64)
#        init_glorot_B -= tf.math.reduce_mean(init_glorot_B)
self.B = tf.Variable(
                      initial_value=init_glorot_B,
                      name='wgt_B_c'+str(self.cell_id),
                      dtype=tf.float64,
                      trainable=True
                     )
'''
#### CALLABLE

'''
@tf.function
def odefunc_Ax_Bu(self,t,x,A,B,u):
    dx = tf.keras.activations.tanh(tf.matmul(A,x) + tf.matmul(B,u))
    return dx

@tf.custom_gradient
def map_to_state(self,A,x,B,u):
    
    x_t = self.odefunc_Ax_Bu(0, x, A, B, u)
    
    def grad(dL):
        
        
        t_0 = self.counter[0]
        t_1 = self.counter[0] + 1.0
        
        t = tf.linspace(t_0, t_1, num=8)
        _x = tfp.math.ode.DormandPrince().solve(self.odefunc_Ax_Bu,t_0,x,solution_times=t,constants={'A':A,'B':B,'u':u})

        dA = tf.reshape(tf.convert_to_tensor(tf.gradients(_x.states,A)),A.shape) * dL
        dx = tf.reshape(tf.convert_to_tensor(tf.gradients(_x.states,x)),x.shape) * dL
        dB = tf.reshape(tf.convert_to_tensor(tf.gradients(_x.states,B)),B.shape) * dL
        du = tf.reshape(tf.convert_to_tensor(tf.gradients(_x.states,u)),u.shape) * dL
        
        return dA, dx, dB, du
    
    return x_t, grad
'''

#### Gradient

'''
def grad(dL):
    
    t_0 = self.counter[0]
    t_1 = self.counter[0] + self.state_size
    
    t = tf.linspace(t_0, t_1, num=10)
    _dr, _dt = tfp.math.ode.DormandPrince().solve(self.odefunc_limit_cycle,t_0,x,solution_times=t,constants={'mu':mu,'theta':theta})
    
    drho = tf.reshape(tf.convert_to_tensor(tf.gradients(_dr.states,rho)),rho.shape) * dL
    dmu = tf.reshape(tf.convert_to_tensor(tf.gradients(_dr.states,mu)),mu.shape) * dL
    dtheta = tf.reshape(tf.convert_to_tensor(tf.gradients(_dt.states,theta)),theta.shape) * dL
    
    return drho, dmu, dtheta
'''
