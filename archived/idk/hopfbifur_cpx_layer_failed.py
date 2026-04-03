import os

import numpy as np

from scipy.stats import unitary_group

import tensorflow as tf

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# keras
#from keras import backend
#from keras.layers.rnn import rnn_utils

''' Local Imports '''
import proj_utils as utils
from proj_utils import _print, _cpx_ptrace, _cpx_pdet, _cpx_eigvals
from activations import HopfBifur
from initializers import Unitary, Hermitian, RandConjSymmVects, UnitaryState
from initializers import ARange, RandomUnitComplex, SkewHermitian

'''
TODO:
- Add multiple cells
- Copy this code over to the use_RNN_class version to see if performance is different.
- Try and see what the output of the ODESover looks like and if we can use some regularizer
    on that data for the output state. 
- Make stuff into complex next.
'''

def _print( msg , arr , **kwargs):
    
    def p_func( msg , arr , **kwargs):
        if arr.dtype.is_complex:
            tf.print('\n'+msg+':\nreal:\n', tf.math.real(arr),'\nimag:\n',tf.math.imag(arr),'\nshape:',arr.shape,'  dtype:',arr.dtype,'\n',**kwargs)
        else:
            tf.print('\n'+msg+':\n', arr,'\nshape:', arr.shape,'  dtype:', arr.dtype,'\n',**kwargs)
    
    if tf.nest.is_nested(arr):
        tf.nest.map_structure(p_func,[msg]*len(arr) , arr)
    else:
        p_func( msg , arr , **kwargs)
    

@tf.keras.utils.register_keras_serializable('hopfbifur_cpx_layer')
class HopfBifurCpxRNNCell(tf.keras.layers.Layer):
    
    '''
    notKerasRNNCell processes one timestep of a Hopf-Bifurcation recurrent network.
    That is the input is mapped with the internal weights, and updated 
    '''
    def __init__(
                    self,
                    state_size,
                    activation=None,
                    **kwargs
                  ):
        
        super(HopfBifurCpxRNNCell, self).__init__(**kwargs)
        
        self.state_size = state_size
#        print('self.state_size',state_size)
        
        self.activation = HopfBifur(dtype=self.dtype) if activation is None else activation
        
        
    def build(self, input_shape):
        
        batch_dim = input_shape[0]
        input_dim = input_shape[-1]
        
        print( 'C input_shape:' , input_shape )
        print( 'C batch_dim:' , batch_dim )
        print( 'C input_dim:' , input_dim )
        print( 'C state_size:' , self.state_size )
        
        '''
        # Hidden State
        self.cell_state = tf.Variable(
            initial_value = tf.zeros( shape=(batch_dim, self.state_size), dtype=self.dtype),
            trainable = False,
            dtype = self.dtype,
            name = 'cell_state'
        )
#        print('self.cell_states:\nReal:\n',tf.math.real(self.cell_state),'\nImag:\n',tf.math.imag(self.cell_state),'\n')
#        exit(0)
        '''
        
        self.G = self.add_weight(
            name='G_herm',
            shape=( batch_dim , self.state_size, self.state_size ),
            dtype=self.dtype,
            initializer=Hermitian( self.state_size ),
            trainable=True
        )
        
        self.H = self.add_weight(
            name='H_herm',
            shape=(batch_dim , self.state_size, self.state_size),
            dtype=self.dtype,
            initializer=Hermitian( self.state_size ),
            trainable=True
        )
        
        super( HopfBifurCpxRNNCell , self ).build( input_shape )
        
        self.built=True
    
    
    @tf.custom_gradient
    def map_unitary_kernel(self, input, kernel):
        output = tf.linalg.matmul( input , kernel )
        def grad(gL):
            gL_cT = tf.transpose(tf.math.conj(gL),[0,2,1])
            J = tf.linalg.matmul( gL_cT, input)
            J_cT = tf.math.conj(tf.transpose(J,[0,2,1]))
            lMJ = tf.linalg.matmul( kernel, J_cT)
            rMJ = tf.linalg.matmul( lMJ, kernel)
            J_sub_rMJ = tf.math.divide( tf.math.subtract(J, rMJ), 2.)
            lMJcT = tf.math.conj(tf.transpose(kernel,[0,2,1]))
            gk = tf.linalg.matmul( lMJcT, J_sub_rMJ)
            gi = tf.linalg.matmul( gL, lMJcT)
            return gi, gk
        return output, grad
    
    
    @tf.custom_gradient
    
    def map_hermitian_kernel(self, input, kernel):
        
        output = tf.linalg.matmul( input , kernel )
        
        def grad( gL ):
            
            input = tf.expand_dims( input , 0)
            
            evals, evecs = tf.linalg.eigh( kernel )
            
            evals = tf.expand_dims( tf.math.real(evals) , 1 )
            
            exp_evals = tf.math.exp( evals )
            
            exp_evals_diff = (exp_evals - tf.transpose(exp_evals , [0,2,1] ))
            
            evals_diff = (evals - tf.transpose(evals , [0,2,1] ))
            
            G = tf.math.divide_no_nan(exp_evals_diff, evals_diff)
            
            vgL = tf.linalg.matmul(tf.transpose(input , [0,2,1] ),tf.math.conj(gL))
            
            gLv = tf.linalg.matmul(tf.transpose(gL , [0,2,1] ),tf.math.conj(input))
            vgL_gLv = tf.math.divide(tf.math.add(vgL,gLv), 2.)
            V = tf.linalg.matmul(tf.transpose(tf.math.conj(evecs) , [0,2,1] ), tf.linalg.matmul(vgL_gLv, evecs))
            VG = tf.math.multiply(V, tf.cast(G, dtype=V.dtype))
            gk = tf.linalg.matmul(evecs, tf.linalg.matmul(VG, tf.transpose(tf.math.conj(evecs) , [0,2,1] )))
            gi = tf.linalg.matmul( gL , tf.math.conj( tf.transpose( kernel ) , [0,2,1] ) )
            
            return gi, gk
            
        return output, grad


    

    def call( self, inputs, states , training=None ):
        
        mapped_inputs = self.map_hermitian_kernel( inputs, self.G)
        mapped_states = self.map_hermitian_kernel( states, self.H)
        new_states = mapped_inputs + mapped_states
        
        output = self.activation( new_states )
        
        return output, new_states
        
    
    def get_config(self):
        base_config = super(HopfBifurCpxRNNCell, self).get_config()
        config = {
            "units": self.units,
            "activation": tf.keras.activations.serialize(self.activation)
        }
        return dict(list(base_config.items()) + list(config.items()))








@tf.keras.utils.register_keras_serializable('hopfbifur_cpx_layer')
class HopfBifurCpxRNNLayer(tf.keras.layers.Layer):
    
    def __init__(
                   self,
                   output_size,
                   state_size,
                   **kwargs
                  ):
        
        super(HopfBifurCpxRNNLayer, self).__init__(**kwargs)
        
        # Collect output size to use for output map
        self.output_dim = output_size
        
        # Internal cell size for mappings
        self.state_size = state_size
        
        self.cell = HopfBifurCpxRNNCell(
            self.state_size,
            dtype=self.dtype,
            activation=tf.keras.activations.tanh,
            name='hopfcell'
        )


    def build(self, input_shape):
        
        if isinstance(input_shape, list): input_shape = input_shape[0]
        
        super(HopfBifurCpxRNNLayer, self).build(input_shape)
        
        batch_dim = input_shape[0]
        time_dim = input_shape[1]
        input_dim = input_shape[-1]
        
        # From tf.keras.layers.basernn
        def get_step_input_shape(shape):
            if isinstance(shape, tf.TensorShape):
                shape = tuple(shape.as_list())
            # remove the timestep from the input_shape
            return (shape[0],) + shape[2:]
        
        step_input_shape = tf.nest.map_structure(
            get_step_input_shape, input_shape
        )
        
        # allow cell (if layer) to build before we set or validate state_spec.
        if not self.cell.built:
            with tf.name_scope(self.cell.name):
                self.cell.build(step_input_shape)
                assert self.cell.built, 'Input state failed to build'
        
        # From cell after being built
        state_dim = self.cell.state_size
        
        # Internal State of Layer
        state_init = np.exp(-2.j*np.pi*np.arange( 0 , state_dim + 1)/(state_dim + 1)).reshape(-1,1)
        state_init = ((state_init[1::])**np.arange(1,batch_dim+1)).T
        state_init = np.reshape(state_init , [batch_dim , 1, state_dim] )
        self.states = tf.Variable(
            initial_value=state_init.copy(),
            trainable=False,
            name='state',
            dtype=self.dtype
        )
#        _print( 'L states' , self.state )
#        exit(0)
        
        self.M = self.add_weight(
            name='M_unit',
            shape=(batch_dim, input_dim, input_dim),
            dtype=self.dtype,
            initializer=Unitary(input_dim),
            trainable=True
        )
#        _print( 'L self.M' , self.M )
#        exit(0)
#        tf.cast( ([rng]*batch_dim) , dtype=self.dtype )) , [batch_dim, input_dim] ),
        rng = np.arange( 1 , input_dim + 1 )
        self.range = tf.constant(
            tf.reshape(
                tf.cast( ([rng]*batch_dim).copy() , dtype=tf.float32 ),
                [ batch_dim , input_dim , 1]
            ),
            dtype=tf.float32
        )
#        _print( 'self.range' , self.range )
#        exit(0)
        
        self.batch_dict = {}
        for x in range(batch_dim): self.batch_dict[x] = None
        
        super(HopfBifurCpxRNNLayer, self).build( input_shape )
        
        self.built = True
        
        
    def call(self, inputs, training=None):
        
        tf.print( '\n' + '-'*100 + '\n')
        
        input = inputs[0] if tf.nest.is_nested(inputs) else inputs
        input = tf.cast(input, dtype=self.dtype)
        
#        tf.print( 'input' , input.shape )
#        tf.print( 'self.states' , self.states.shape )
#        tf.print( 'self.M' , self.M.shape )
#        tf.print( 'self.range' , self.range.shape )
        
        input_map = tf.linalg.matmul( input , self.M )  # (batch_size , 1 , input_size )
#        _print( 'input_map' , input_map )
        
        state_angle = tf.math.angle( self.states )  # ( batch , state_size )
#        _print( 'state_angle' , state_angle )
        
        # Outer Product
        state_proj = tf.einsum( 'bij,bki->bjk' , state_angle , self.range ) # (batch , state_size , state_size )
        state_proj = tf.math.exp( -1.j * state_proj )   # (batch , state_size , state_size )
#        _print( 'state_proj' , state_proj )
        
        input_state_map = tf.einsum( 'bik,bjk->bij' , input_map , state_proj )  # (batch , 1 , state_size )
#        _print( 'input_state_map' , input_state_map )
#        _print( 'self.states' , self.states )
        
        input_map = tf.unstack(input_state_map)
#        tf.print( 'input_map' , len(input_map) )
        
        state_map = tf.unstack(self.states)
#        tf.print( 'state_map' , len(state_map) )
        
        call_fn = self.cell.__call__
        mapped_cell = tf.nest.map_structure(
            lambda _x, _z : call_fn( _x , _z , training=training ),
            input_map,
            state_map
        )
        mapped_cell = list(zip(*mapped_cell))
#        tf.print( 'mapped_cell' , mapped_cell )
        
        cell_output = tf.stack( mapped_cell[0] )
        cell_states = tf.stack( mapped_cell[1] )
        _print( 'cell_output' , cell_output )
        _print( 'cell_states' , cell_states )
        
        outputs = tf.cast( cell_output , dtype = inputs.dtype )
        
        outputs = [outputs] if tf.nest.is_nested(inputs) else outputs
        
        return outputs
        
        
        
    @tf.function
    def rnn_call(self, cell, inputs, states, **kwargs):
        
        nested_states = [states] if not tf.nest.is_nested(states) else states
        
        # Get cell's calling function
        cell_call_fn = ( cell.__call__ if callable( cell ) else cell.call )
        
        def step( t_inputs, t_states ):
            t_states = ( t_states[0] if len(t_states) == 1 else t_states )
            output, new_states = cell_call_fn( t_inputs , t_states, **kwargs )
            if not tf.nest.is_nested(new_states): new_states = [new_states]
            return output, new_states
        
        lastout, outputs, states = tf.keras.backend.rnn(step, inputs, nested_states)
        
        _lastout = lastout[0] if tf.nest.is_nested(lastout) else lastout
#        _print('_lastout',_lastout)
        
        _outputs = outputs[0] if tf.nest.is_nested(outputs) else outputs
#        _print('_outputs',_outputs)
        
        _states = states[0] if tf.nest.is_nested(states) else states
#        _print('_states',_states)
        
        return ( _lastout , _outputs , _states )
    
    '''
    def get_config(self):
        
        base_config = super(HopfBifurCpxRNNLayer, self).get_config()
        
        if 'cell' in base_config: del base_config['cell']
        return base_config
    '''
    
    '''
    @classmethod
    def from_config(self, config):
        return self(**config)
    '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
'''




class HopfCell(tf.keras.layers.Layer):
    
    def build(self, input_shape):
        
        super(HopfCell,self).build(input_shape=input_shape)

        in_sz = input_shape[-1]
        self.in_sz = in_sz
        
        root = np.exp(-2j*np.pi*np.arange(in_sz)/in_sz).reshape(-1,1)
        self.root = tf.constant(root,dtype=tf.complex128)
        
        # Normal dft
        dft = root**np.arange(in_sz)
        self.dft = tf.constant(dft,dtype=tf.complex128)
        
        # Unitary dft
        udft = dft / np.sqrt(in_sz)
        self.udft = tf.constant(udft,dtype=tf.complex128)
        
        zeros = np.zeros(shape=(in_sz,1))
        self.zeros  = tf.constant(zeros,dtype=tf.complex128)
        
        ones = np.ones(shape=(in_sz,1))
        self.ones  = tf.constant(ones,dtype=tf.complex128)
        
        self.eye = tf.constant(np.eye(in_sz),dtype=tf.complex128)
        
        def make_unitary_zero_det_weight(sz,nme='U',do_save=True):
            
            def gen_wgt_matrix(t_sz):
                wgt = unitary_group.rvs(t_sz)
                wgt = tf.linalg.logm(wgt)
                wgt = (wgt + tf.math.conj(wgt)[::-1,::-1])/2.
                wgt = wgt / tf.math.pow(tf.linalg.det(wgt),(1/in_sz))
                wgt = tf.linalg.expm(wgt)
                return wgt
            
            if do_save is True:
                if os.path.exists('rnn_mat_'+nme+'.npy'):
                    wgt = np.load('rnn_mat_'+nme+'.npy',allow_pickle=True)
                    # Check saved shape and replace if not the same.
                    if wgt.shape[-1] != sz:
                        wgt = gen_wgt_matrix(sz)
                        np.save('rnn_mat_'+nme+'.npy',wgt,allow_pickle=True)
#                        tf.print('Generating new matrix for '+nme)
                else:
                    wgt = gen_wgt_matrix(sz)
                    np.save('rnn_mat_'+nme+'.npy',wgt,allow_pickle=True)
#                    tf.print('Generating new matrix for '+nme)
            else:
                wgt = gen_wgt_matrix(sz)
#                tf.print('Generating new matrix for '+nme)
            return wgt
        
        def make_random_normal_vec(sz,nme='z',do_save=True):
            if do_save is True:
                if os.path.exists('rnn_vec_'+nme+'.npy'):
                    vec = np.load('rnn_vec_'+nme+'.npy',allow_pickle=True)
                    # Check saved shape and replace if not the same.
                    if vec.shape[0] != sz:
                        vec = np.random.rand(sz).reshape(-1,1)
                        np.save('rnn_vec_'+nme+'.npy',vec,allow_pickle=True)
#                        tf.print('Generating new vector for '+nme)
                else:
                    vec = np.random.rand(sz).reshape(-1,1)
                    np.save('rnn_vec_'+nme+'.npy',vec,allow_pickle=True)
#                    tf.print('Generating new vector for '+nme)
            else:
                vec = np.random.rand(sz).reshape(-1,1)
#                tf.print('Generating new vector for '+nme)
            return vec
        
        def make_range_vec(sz,nme='z',do_save=True):
            if do_save is True:
                if os.path.exists('rnn_vec_'+nme+'.npy'):
                    vec = np.load('rnn_vec_'+nme+'.npy',allow_pickle=True)
                    # Check saved shape and replace if not the same.
                    if vec.shape[0] != sz:
                        vec = np.arange(1,sz+1).reshape(-1,1)
                        np.save('rnn_vec_'+nme+'.npy',vec,allow_pickle=True)
#                        tf.print('Generating new vector for '+nme)
                else:
                    vec = np.arange(1,sz+1).reshape(-1,1)
                    np.save('rnn_vec_'+nme+'.npy',vec,allow_pickle=True)
#                    tf.print('Generating new vector for '+nme)
            else:
                vec = np.arange(1,sz+1).reshape(-1,1)
#                tf.print('Generating new vector for '+nme)
            return vec
        
        # Set to True if we want to reuse the initialization matrices for common training
        _save = False
        _train = True
        _sz = in_sz
        

#        A = make_unitary_zero_det_weight(_sz,nme='A',do_save=_save)
#        A = (A + tf.math.conj(A).T)/2.
#        self.A = tf.Variable(   A,
#                                name='A_hrm',
#                                dtype=tf.complex128,
#                                trainable=_train
#                                )
        
#        B = make_unitary_zero_det_weight(_sz,nme='B',do_save=_save)
#        B = (B + tf.math.conj(B).T)/2.
#        self.B = tf.Variable(   B,
#                                name='B_hrm',
#                                dtype=tf.complex128,
#                                trainable=_train
#                                )
       
#        G = make_unitary_zero_det_weight(_sz,nme='G',do_save=_save)
#        self.G = tf.Variable(   G,
#                                name='G_unt',
#                                dtype=tf.complex128,
#                                trainable=_train
#                                )
        
        U = make_unitary_zero_det_weight(_sz,nme='U',do_save=_save)
        self.U = tf.Variable(   U,
                                name='U_unt',
                                dtype=tf.complex128,
                                trainable=_train
                                )
        
        V = make_unitary_zero_det_weight(_sz,nme='V',do_save=_save)
        self.V = tf.Variable(   V,
                                name='V_unt',
                                dtype=tf.complex128,
                                trainable=_train
                                )
        
        z = make_random_normal_vec(in_sz,'z',do_save=_save)
        z = self.U @ z
        self.z = tf.Variable(
                              initial_value=z,
                              name='z',
                              dtype=tf.complex128,
                              trainable=False
                             )
        
        self.counter = tf.Variable([0],dtype=tf.float64,trainable=False)
        
        self.built = True


#    @tf.function
    @tf.custom_gradient
    def unitary_map(self,M,v):
        Mv = M @ v
        def grad(gL):
            J = gL*tf.math.conj(v).T
            gM = (tf.math.conj(M).T @ ((J - M @ tf.math.conj(J).T @ M)/2.))
            gv = tf.zeros_like(v)
            return gM, gv
        return Mv, grad

    @tf.custom_gradient
    def herm_map(self,M,v):
        Mv = M @ v
        def grad(gL):
            lmdas, W = tf.linalg.eigh(M)
            lmdas = tf.expand_dims(lmdas,1)
            exp_lmdas = tf.math.exp(lmdas)
            G = tf.math.divide_no_nan((exp_lmdas - exp_lmdas.T),(lmdas - lmdas.T))
            J = (v @ tf.math.conj(gL).T + gL @ tf.math.conj(v).T)/2.
            VG = (tf.math.conj(W).T @ J @ W) * G
            gM = W @ VG @ tf.math.conj(W).T
            gv = tf.math.conj(M).T @ gL
            return gM, gv
        return Mv, grad


    @tf.custom_gradient
    def cos_sin_map(self,C,S,z,x):
    # C = (U + tf.math.conj(U).T)/2.    &    S = (U - tf.math.conj(U).T)/2.j
    #       where U is a unitary matrix.
        Mv = C @ z + S @ x
        def grad(gL):
            lmdas_c, W_c = tf.linalg.eigh(C)
            lmdas_s, W_s = tf.linalg.eigh(S)
            lmdas_c = tf.expand_dims(lmdas_c,1)
            lmdas_s = tf.expand_dims(lmdas_s,1)
            cos_lmdas = tf.math.cos(lmdas_c)
            sin_lmdas = tf.math.sin(lmdas_s)
            G_c = tf.math.divide_no_nan((cos_lmdas - cos_lmdas.T),(lmdas_c - lmdas_c.T))
            G_s = tf.math.divide_no_nan((sin_lmdas - sin_lmdas.T),(lmdas_s - lmdas_s.T))
            J_c = (z @ tf.math.conj(gL).T + gL @ tf.math.conj(z).T)/2.
            J_s = (x @ tf.math.conj(gL).T + gL @ tf.math.conj(x).T)/2.
            VG_c = (tf.math.conj(W_c).T @ J_c @ W_c) * G_c
            VG_s = (tf.math.conj(W_s).T @ J_s @ W_s) * G_s
            gC = W_c @ VG_c @ tf.math.conj(W_c).T
            gS = W_s @ VG_s @ tf.math.conj(W_s).T
            gz = tf.zeros_like(z)
            gx = tf.zeros_like(x)
            return gC, gS, gz, gx
        return Mv, grad


    @tf.custom_gradient
    def exp_map(self,a,b):
        N = tf.math.exp(tf.math.conj(a)*b)*(-1.+0.j)
        def grad(gL):
            ga = gL*N*b
            gb = gL*N*tf.math.conj(a)
            return ga, gb
        return N, grad


#    @tf.function
    def call(self,input,training=False):
        
#        tf.print('\n-----------------------------------------------------------------------------\n',self.counter[0],')\n')
        
        sz = (0.+self.in_sz)
        
        z = self.z
        x = input
#        _print('x',x)
        
#        A = self.A
#        B = self.B
        U = self.U
        V = self.V
#        E = self.E
        
#        Az = self.herm_map(A,z)
#        Bx = self.herm_map(B,x)
#        _print('Az',Az)
#        _print('Bx',Bx)
        
        Uz = self.unitary_map(U,z)
        Vx = self.unitary_map(V,x)
#        _print('Uz',Uz)
#        _print('Vx',Vx)
        
#        a = Az + Bx
#        a = self.cos_sin_map(A,B,z,x)
        a = Uz + Vx
        b = self.ones*(-1.+0.j)
#        b = self.exp_map(Uz,Vx)
#        _print('a',a,True)
#        _print('b',b,True)
        
#        _print('z',z,True)
        
        # Map to the limit cycle #
        
        z_t = self.hopf_ODE(z,a,b)
        
#        z_t = tf.math.tanh(a)
        
#        _print('z_t',z_t,True)
        
        # Pulling the real-value and re-mapping it to the DFT fixes floating point erros that explode the model.
        self.z.assign(z_t)
        self.counter.assign_add([1.])
        
        return tf.reshape(z_t,input.shape)
    
    
class HopfLayer(tf.keras.layers.Layer):
    
    def __init__(self,**kwargs):
        
        super(HopfLayer, self).__init__(**kwargs)
        
        # Creates state_size number of RNNCells with each cell being state_size large.
#        self.cells = [MyRNNCell(self.state_size,cell_id=j,name='rnncell_'+str(j)) for j in range(self.state_size)]
        self.cell = HopfCell(name='hopf')
        
        self.counter = tf.Variable([0],dtype=tf.float64,trainable=False)



    def build(self,input_shape):
        
        super(HopfLayer,self).build(input_shape=input_shape)
        
        in_sz = input_shape[-1]
        self.in_sz = in_sz
        
        self.cell.build(input_shape[1::])
        
        self.built = True


    def call(self, input, training=False):

#        tf.print('\n-----------------------------------------------------------------------------------------------------------------------\nCount',self.counter[0],'\n')
        
        _input = tf.reshape(input,[self.in_sz,1])
#        _print('_input',_input)
        
        
        # Make multiple calls ... maybe.
        y_t = self.cell.call(_input,training=training)
#        _print('y_t',y_t)
        
#        exp = tf.math.exp(1.j*tf.math.angle(y_t))
#        _print('exp',exp)
        

#        y_hat = tf.math.conj(y_t).T @ exp
        y_hat = tf.math.conj(y_t).T @ y_t
#        _print('y_hat',y_hat)

        self.counter.assign_add([self.in_sz])
        
        return tf.math.real(y_hat) # tf.reshape(y_hat,_input.shape)



class HopfModel(tf.keras.Model):


    def __init__(self, input, output):
        
        super(HopfModel, self).__init__(input, output)
        
#        self.my_loss = MyLoss()
        
#        self.my_accu = tf.keras.metrics.KLDivergence(name='kullback_leibler_divergence', dtype=None)
#        self.my_accu = tf.keras.metrics.CosineSimilarity(name='cosine_similarity', dtype=None, axis=-1)
        
        self.my_prec = None # tf.keras.metrics.Precision(thresholds=None, top_k=None, class_id=None, name='precision', dtype=None)
        
        self.results = {}
        
        self.my_loss = None
        
        self.my_accu = None
        
#        if self.my_loss is not None:
#            self.results['loss'] = tf.constant([0.],dtype=tf.float64)
        
#        if self.my_accu is not None:
#            self.results['accuracy'] = tf.constant([0.],dtype=tf.float64)

#        if self.my_prec is not None:
#            self.results['precision'] = tf.constant([0.],dtype=tf.float64)


    def train_step(self, data):
        
#        tf.print('\n___________________________________________________START__________________________________________________\n')

        input, image = data
        input = tf.cast(input,dtype=tf.float64)
        image = tf.cast(image,dtype=tf.float64)
        
        with tf.GradientTape() as tape:
            
#            tf.print('\n-----------------------------------------------------------------------------------------------------\nPrediction:\n')
            
#            _print('HopfModel -> input',input)
            
            pred = tf.cast(self(input, training=True),dtype=input.dtype)  # Forward pass
#            pred = tf.reshape(pred,image.shape)
#            _print('HopfModel -> pred',pred)
#            _print('HopfModel -> image',image)
            
            # My Local Loss for multi-dim loss stuffs
#            loss = tf.math.pow(image - pred,2)/(0.+pred.shape[-1])
#            _print('loss',loss)
            
            loss = self.compiled_loss(image,pred)
#            loss = self.myloss
#            loss = tf.math.pow(pred - image,2)/(0.+pred.shape[-1])
#            _print('HopfModel -> loss',loss)
            
            if self.my_loss is not None:
                self.results['loss'] = tf.math.reduce_mean(loss)
    #            _print('HopfModel -> accuracy',self.my_accu.result())
            
#            if self.my_accu is not None:
#                self.my_accu.update_state(image,pred)
#                self.results['accuracy'] = self.my_accu.result()
    #            _print('HopfModel -> accuracy',self.my_accu.result())
            
#            if self.my_prec is not None:
#                self.my_prec.update_state(image,pred)
#                self.results['precision'] = self.my_prec.result()
    #            _print('HopfModel -> precision',self.my_prec.result())
            
#            tf.print('\n-----------------------------------------------------------------------------------------------------\nBackward Gradient:\n')
            
            # Compute gradients 
            vars = self.trainable_variables
#            for v in vars:
#                _print('HopfModel -> train_vars',v)
#                unt = tf.linalg.expm(v)
#                untI = tf.matmul(tf.transpose(unt),unt)
#                _print('V untI',untI)
                
            grad = tape.gradient(loss, vars)
            #tf.print('HopfModel:train_step -> Gradients:\n')
#            for g in _grad:
#                _print('HopfModel -> grads',g)
#                unt = tf.linalg.expm(g)
#                untI = tf.matmul(tf.transpose(unt),unt)
#                _print('G untI',untI)
            
        wgts = self.trainable_weights
        #tf.print('Wgts trainable_weights:\n',_wgts,'\n')
        
        self.optimizer.apply_gradients(zip(grad,wgts))
        
        # Updated trainable weights
        #tf.print('Wgts POST-apply_gradients:')
        #for _w in _wgts:
        #    printComplex('HopfModel:train_step -> _w', _w, print_shape=True)
        
        #tf.print('\n___________________________________________________END__________________________________________________\n\n')
        
        return self.results


class MyLoss(tf.keras.losses.Loss):

    def __init_(self, reduction=tf.keras.losses.Reduction.NONE, name=None, **kwargs):
        super().__init_(reduction=reduction, name=name, **kwargs)

    @tf.custom_gradient
    def call(self, true, pred):
        
        _pred = tf.cast(pred,true.dtype)
        _true = tf.reshape(true,_pred.shape)

        _loss = tf.math.squared_difference(_pred,_true)
        
        def grad(dL): # dL returns [1.,1.,...,1.] same shape as _loss.shape
            dL_dT = dL * tf.gradients(_loss,true)
            dL_dP = dL * tf.gradients(_loss,pred)
            return dL_dT, dL_dP
        
        return _loss, grad # Returns complex tensor as error


'''