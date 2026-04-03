import os
import math
import collections
import time 

import numpy as np
from scipy.stats import unitary_group

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

''' Local Imports '''
import proj_utils as utils
from activations import HopfBifur
from initializers import Hermitian , Unitary, RandomUnitComplex

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
            tf.print('\n'+msg+':\nreal:\n', tf.math.real(arr),'\nimag:\n',tf.math.imag(arr),'\nshape:',arr.shape,'  dtype:',arr.dtype,'  type:',type(arr),'\n',**kwargs)
        else:
            tf.print('\n'+msg+':\n', arr,'\nshape:', arr.shape,'  dtype:', arr.dtype,'  type:',type(arr),'\n',**kwargs)
    
    if tf.nest.is_nested(arr):
        tf.nest.map_structure(p_func,[msg]*len(arr) , arr)
    else:
        p_func( msg , arr , **kwargs)







@tf.keras.utils.register_keras_serializable('hopfbifur_cpx_layer')
class HopfBifurCpxRNNCell(tf.keras.layers.Layer):
    
    def __init__(
                    self,
                    state_size,
                    activation=None,
                    **kwargs
                  ):
        
        super( HopfBifurCpxRNNCell , self ).__init__( **kwargs )
        
        # Size of input cells
        self.state_size = state_size
        
        # Set default activation or input activation.
        self.activation = HopfBifur( dtype=self.dtype ) if activation is None else activation
        
        
        
    def build(self, input_shape):
        
        batch_size = input_shape[0]
        input_size = input_shape[-1]
        
        self.sqrt_insize = tf.constant( np.sqrt([input_size]) , dtype=self.dtype )
        
        def get_cpx_identity(shape,dtype):
            return tf.cast( np.eye(shape[-1]) , dtype=dtype )
        
        self.A = self.add_weight(
            name='A_'+self.name,
            shape=( self.state_size , self.state_size ),
            dtype=self.dtype,
            initializer=get_cpx_identity,
            trainable=True
        )
        
        self.B = self.add_weight(
            name='B_'+self.name,
            shape=( self.state_size , self.state_size ),
            dtype=self.dtype,
            initializer=get_cpx_identity,
            trainable=True
        )
        
        
        
        
        # Hidden State
        h_init = np.random.rand( self.state_size )
        h_init = np.exp(-2.j*np.pi*h_init)*h_init
        h_init = (h_init + np.conjugate(h_init)[::-1])/2.
        h_init = tf.reshape( tf.cast( h_init , dtype=self.dtype ) , [ 1, self.state_size] )
        self.state = tf.Variable(
            name='state_'+self.name,
            shape=( 1 , self.state_size ),
            dtype=self.dtype,
            initial_value=h_init,
            trainable=False
        )
        
        self.G = self.add_weight(
            name='G_herm_'+self.name,
            shape=( self.state_size , self.state_size ),
            dtype=self.dtype,
            initializer=Hermitian(),
            trainable=True
        )
        
        self.H = self.add_weight(
            name='H_herm_'+self.name,
            shape=( self.state_size , self.state_size ),
            dtype=self.dtype,
            initializer=Hermitian(),
            trainable=True
        )
        
        
        super( HopfBifurCpxRNNCell , self ).build( input_shape )
        
        self.cnt = tf.Variable( [0] , trainable=False , dtype=tf.int32 )
        
        self.built=True



    @tf.function
    def project_to_state( self , state , input ):
        
        mag_st = tf.math.sqrt( tf.math.reduce_sum( state * tf.math.conj(state) ) )
        mag_in = tf.math.sqrt( tf.math.reduce_sum( input * tf.math.conj(input) ) )
        norm = mag_st * mag_in
        
        proj_input = tf.linalg.matmul(
            tf.linalg.matmul( tf.transpose( state ) , input ) / norm,
            tf.linalg.matrix_transpose( input , conjugate=True )
        )
        return tf.reshape( proj_input , state.shape )
        
        
        
    def call(self, inputs, states, training = False):
    
#        tf.print('\n'+'- '*20+' Cell '+'- '*20+'\nCount:',self.cnt[0],'\n')
        
        new_input = inputs[0] if tf.nest.is_nested( inputs ) else inputs
        prev_state = states[0] if tf.nest.is_nested( states ) else states
        
        ''' Have:
                1) Global state of the outer system
                2) internal state of the cell
            
            Cell Perspective:
                1) Recieves a set of frequencies from some global mass
                2) Determine if the input frequencies help the internal freq.
                    a) current internal freq. vs. input 
                3) 
        '''
        
#        input_map = self.map_hermitian_kernel( new_input , self.G )
        input_map = tf.linalg.matmul( new_input , self.G )
#        input_map = tf.linalg.matmul( new_input , self.A )
        
#        state_map = self.map_hermitian_kernel( prev_state , self.H )
        state_map = tf.linalg.matmul( prev_state , self.H )
#        state_map = tf.linalg.matmul( prev_state , self.B )
        
        new_state = input_map + state_map
#        _print( 'new_state' , new_state )
        
        new_state = tf.math.divide_no_nan(
            new_state,
            tf.norm( new_state , ord=2, axis=-1 )
        )
#        _print( 'new_state' , new_state )
        
        new_output = self.activation( new_state )
#        _print( 'new_output' , new_output )
        
        # check for shape compatibility & re-nest if the previous one was nested.
        ret_value = [ new_output ] if tf.nest.is_nested( inputs ) else new_output
        ret_state = [ new_state ] if tf.nest.is_nested( states ) else new_state
        
        self.cnt.assign_add([1])
        
        return ret_value , ret_state
        
        
    def get_config(self):
        base_config = super(HopfBifurCpxRNNCell, self).get_config()
        config = {
                    "units": self.units,
                    "activation": tf.keras.activations.serialize(self.activation)
                    }
        return dict(list(base_config.items()) + list(config.items()))
        
        
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
    def map_hermitian_kernel( self, x , A ):
        
        y = tf.linalg.matmul( x , A )
        
        def grad( gL ):
            
            # The following directional derivative calculation is based on the paper by Najfeld and Havel.
            # Cite: I. Najfeld and T. F. Havel, “Derivatives of the Matrix Exponential and Their Computation,”
            # Advances in Applied Mathematics, vol. 16, no. 3, pp. 321–375, Sep. 1995, doi: 10.1006/aama.1995.1017.
            
            x_cT = tf.transpose( x , conjugate=True )
            
            gL_cT = tf.linalg.matrix_transpose( gL , conjugate=True )
            
            evls , U = tf.linalg.eigh( A )      # Where U is a unitary matrix such that A = UVU*
            evls = tf.expand_dims( evls , 0 )
            evls_T = tf.transpose( evls )
            
            U_cT = tf.linalg.matrix_transpose( U , conjugate=True )
            
            exp_evls = tf.math.exp(evls)
            exp_evlsT = tf.transpose( exp_evls )
            phi = tf.math.divide_no_nan(
                (exp_evlsT - exp_evls),
                ( evls_T - evls )
            )
            phi = tf.math.real( phi )
            
            diag_exp_evls = tf.linalg.diag( exp_evls[0] )
            G = tf.cast( phi , dtype=diag_exp_evls.dtype ) + diag_exp_evls
            
            V = tf.math.divide(
                tf.math.add(
                    tf.linalg.matmul( tf.transpose( x ), tf.math.conj( gL ) ),
                    tf.linalg.matmul( tf.transpose( gL ), tf.math.conj( x ) )
                ), 2.
            )
            
            V_bar = tf.linalg.matmul( tf.linalg.matmul( U_cT , V ) , U )
            
            GV = tf.math.multiply( G , V_bar )
            
            Dv_A = tf.linalg.matmul( tf.linalg.matmul( U , GV ) , U_cT )
            
            gx = tf.linalg.matmul( gL , tf.transpose( A , conjugate=True ) )
            gA = Dv_A
            
            return gx , gA
        
        return y , grad






@tf.keras.utils.register_keras_serializable('hopfbifur_cpx_layer')
class HopfBifurCpxRNNLayer(tf.keras.layers.Layer):
    
    def __init__(
                   self,
                   state_size=None,
                   output_size=None,
                   activation=None,
                   **kwargs
                  ):
        
        super(HopfBifurCpxRNNLayer, self).__init__(**kwargs)
        
        # The state size of each cells, if None then defaults to square-root of input dimensions and cast as an integer.
        self.state_size = state_size
        
        self.output_size = output_size
        
        # To hold the cell(s) for the model.
        self.cell = None
        
        # Set the activation function
        self.activation = activation
        
    
    
    def build(self, input_shape):
        
        batch_size = input_shape[0]
        time_size = input_shape[-2]
        input_size = input_shape[-1]
        
        if self.output_size is None: self.output_size = input_size
        if self.state_size is None: self.state_size = int(math.sqrt(input_size))*2
        
        # Generate cell -> Only use 1 for now, until it's working better.
        if self.cell is None:
            self.cell = HopfBifurCpxRNNCell(
                state_size=self.state_size,
                activation=self.activation,
                dtype=self.dtype,
                name='hopf_cell'
            )
        
        # From tf.keras.layers.basernn
        def get_step_input_shape(shape):
            if isinstance(shape, tf.TensorShape):
                shape = tuple(shape.as_list())
            # remove the timestep from the input_shape
            return (shape[0],) + shape[2:]
        
        step_input_shape = tf.nest.map_structure( get_step_input_shape , input_shape )
        
        # Build cell(s)
        if not self.cell.built:
            with tf.name_scope(self.cell.name):
                self.cell.build( step_input_shape )         # Last build shape -> self.cells.build( input_shape[1::] )
                assert self.cell.built , 'HopfBifurCpxRNNLayer failed to build.'
        
        # Global State
        state_init = np.arange(0,self.state_size)/self.state_size
        state_init = np.exp(2.j*np.pi*state_init)
        state_init = (state_init + np.conjugate(state_init)[::-1])/2.
        state_init = tf.reshape( tf.cast( state_init , dtype=self.dtype ) , [ 1 , self.state_size] )
        self.state = tf.Variable(
            initial_value=state_init,
            shape=( 1 , self.state_size ),
            dtype=self.dtype,
            trainable=False,
            name='state_'+self.name
        )
        
        rnginit = np.linspace( 0 , 2*np.pi , input_size)
        rnginit = tf.reshape( rnginit , [ 1 , input_size ])
        rnginit = tf.cast( rnginit , dtype=self.dtype )
        self.inrng = tf.constant(
            rnginit,
            dtype=self.dtype,
            name='in_rng_'+self.name
        )
        
        rnginit = np.linspace( 0 , 2*np.pi , self.output_size)
        rnginit = tf.reshape( rnginit , [ 1 , self.output_size ])
        rnginit = tf.cast( rnginit , dtype=self.dtype )
        self.outrng = tf.constant(
            rnginit,
            dtype=self.dtype,
            name='out_rng_'+self.name
        )
        
        super( HopfBifurCpxRNNLayer, self ).build( input_shape )
        
        self.built = True
        
        
    @tf.function
    def generate_bases( self , state , range , conj=False ):
        angle = tf.cast( tf.math.angle( state ) , dtype=self.dtype )
        root = tf.linalg.matmul( angle , range , transpose_a = True )
        bases = tf.math.divide_no_nan(
            tf.math.exp( 1.j*root ),
            tf.math.sqrt( tf.cast( [self.state_size] , dtype=self.dtype ) )
        )
        # Possibly Normalize Here ... 
        return tf.reshape( bases , [ state.shape[-1] , range.shape[-1] ] )
        
        
    def call(self, inputs, training=False):
        
#        tf.print('\n'+'-'*25+' Layer '+'-'*25+'\n')
#        _print('inputs',inputs)
#        _print('state',self.state)
        
        cast_inputs = tf.cast( inputs, dtype=self.dtype )
        
        # TODO: Apply forward mapping with state stuff here
        #       return_state must be True for this.
        state_bases = self.generate_bases( self.state , self.inrng )
#        _print('state_bases',state_bases)
        
#        Ichk = tf.linalg.matmul( state_bases , state_bases , adjoint_a = True )
#        _print('1) Ichk',Ichk)
        
        proj_inputs = tf.linalg.matmul( cast_inputs , state_bases , transpose_b=True )
#        _print( 'proj_inputs' , proj_inputs )
        
        rnn_return = self.rnn_call(
            self.cell,
            proj_inputs,
            self.state,
            time_major = True,          # Provides our 'cell' layer with single input tensors instead of entire batch.
            return_all_outputs=True,    # If True -> seq_out has batch sized output; else, single output
            training = training
        )
        cell_out , seq_out , cell_state = rnn_return
        
#        tf.print('\n'+' *'*15+' RNN Returned '+'* '*15+'\n')
#        _print( 'cell_out', cell_out)
#        _print( 'seq_out', seq_out)
#        _print( 'cell_state', cell_state)
        tf.debugging.check_numerics( tf.math.real( cell_out ) , 'a cell_out element is nan')
        tf.debugging.check_numerics( tf.math.real( seq_out ) , 'a seq_out element is nan')
        tf.debugging.check_numerics( tf.math.real( cell_state ) , 'a cell_state element is nan')
        
        # Combine old state and new state
        cell_state = tf.reshape( cell_state , self.state.shape )
        state_change = tf.math.divide_no_nan(
            tf.math.conj(self.state) * cell_state,
            tf.math.sqrt( tf.math.conj(self.state) * self.state )
        )
#        _print( 'state_change' , state_change )
#        _print( 'self.state' , self.state )
        
        new_state = state_change * self.state
#        self.state.assign( new_state )
#        _print( 'new_state' , new_state )
        
        state_bases = self.generate_bases( cell_state , self.outrng )
#        _print('state_bases',state_bases)
        
        out_map = tf.linalg.matmul( seq_out , state_bases )
#        _print( 'out_map' , out_map )
        imgvals = tf.math.reduce_sum( tf.math.imag( out_map ) )
#        tf.print('imgvals:',imgvals)
#        tf.debugging.assert_less( imgvals , 1.e-3, 'imaginary values went above 1.e-3 issue with recurrance.' )
#        _print( 'imgvals' , imgvals )
        
        output = tf.math.real( out_map )
#        _print( 'output' , output )
        
        return output
        
        
    def get_config(self):
        base_config = super(HopfBifurCpxRNNLayer, self).get_config()
        if 'cells' in base_config: del base_config['cells']
        return base_config
        
        
    @classmethod
    def from_config(self, config):
        return self(**config)
        
        
    @tf.function
    def rnn_call(self, cell, inputs, states, time_major=False, return_all_outputs=False , training=False):
        
#        nested_states = [states] if not tf.nest.is_nested(states) else states
        flat_states = tf.nest.flatten( states )
        
        # Get cell's calling function
        cell_call_fn = ( cell.__call__ if callable(cell) else cell.call )
        
        # Callback for backend.rnn() operations.
        def _step(step_inputs, step_states):
            step_states = step_states[0]
            output, new_states = cell_call_fn(step_inputs, step_states, training=training)
            #if not tf.nest.is_nested(new_states): new_states = [new_states]
            return output, new_states
        
        new_lastouts, new_outputs, new_states = tf.keras.backend.rnn(
            _step,
            inputs,
            flat_states,
            time_major=time_major,
            return_all_outputs=return_all_outputs
        )
        
        _lastout = new_lastouts[0] if tf.nest.is_nested( new_lastouts ) else new_lastouts
#        _print('_lastout',_lastout)
        
        _outputs = new_outputs[0] if tf.nest.is_nested( new_outputs ) else new_outputs
#        _print('_outputs',_outputs)
        
        _states = new_states[0] if tf.nest.is_nested( new_states ) else new_states
#        _print('_states',_states)
        
        return (_lastout, _outputs, _states)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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