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
from initializers import Hermitian , HermitianV2 , Unitary, RandomUnitComplex

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
        
        self.I = self.add_weight(
            name='I_'+self.name,
            shape=( self.state_size , self.state_size ),
            dtype=self.dtype,
            initializer=get_cpx_identity,
            trainable=False
        )
        
        
        
        
        
        # Hidden State
        state_init = np.arange( self.state_size ) / self.state_size
        state_init = state_init + state_init[1]/2.
        state_init = np.exp(-2.j*np.pi*state_init)
        self.state = tf.Variable(
            name='state_'+self.name,
            shape=( 1 , self.state_size ),
            dtype=self.dtype,
            initial_value=tf.reshape(
                tf.cast( state_init.copy() , dtype=self.dtype ),
                [ 1, self.state_size ]
            ),
            trainable=False
        )
        
        '''
        self.U = self.add_weight(
            name='U_unit_'+self.name,
            shape=( self.state_size , self.state_size ),
            dtype=self.dtype,
            initializer=Unitary(),
            trainable=False
        )
        self.W = self.add_weight(
            name='W_unit_'+self.name,
            shape=( self.state_size , self.state_size ),
            dtype=self.dtype,
            initializer=Unitary(),
            trainable=False
        )
        '''
        
        self.G = self.add_weight(
            name='G_herm_'+self.name,
            shape=( self.state_size , self.state_size ),
            dtype=self.dtype,
            initializer=HermitianV2(),
            trainable=True
        )
        
        self.H = self.add_weight(
            name='H_herm_'+self.name,
            shape=( self.state_size , self.state_size ),
            dtype=self.dtype,
            initializer=HermitianV2(),
            trainable=True
        )
        
        super( HopfBifurCpxRNNCell , self ).build( input_shape )
        
        self.cnt = tf.Variable( [0] , trainable=False , dtype=tf.int32 )
        
        self.built=True
        
        
        '''
        Idea:
        1) Global state keeps both the phase and the amplitude of whatever is current.
        2) hidden state keeps only the phase of the cell
        3) Input, global state, and phase are mapped based on operators.
        4) How well the phase and amplitude match after mapping determines if the 
            cell contributes to & by how much ? idk.
         - Also the 1st derivative of a Hermitian operator is anti-hermitian.
        '''
        
        
    def call( self , inputs , states , training=False ):
        
#        tf.print('\n'+'- '*20+' Cell '+'- '*20+'\nCount:',self.cnt[0],'\n')
        
        x = inputs[0] if tf.nest.is_nested( inputs ) else inputs
        z = states[0] if tf.nest.is_nested( states ) else states
        h = self.state
        
        x_T = tf.linalg.adjoint( x )
        z_T = tf.linalg.adjoint( z )
        h_T = tf.linalg.adjoint( h )
        
#        _print( 'x_T' , x_T )
#        _print( 'mag(x_T)' , tf.math.abs( x_T ) )
#        _print( 'ang(x_T)' , tf.math.angle( x_T ) )
        
        Hx = self.hermitian_map( self.H , x_T )
#        _print( 'mag(Hx)' , tf.math.abs( Hx ) )
#        _print( 'ang(Hx)' , tf.math.angle( Hx ) )
        
#        Gz = self.hermitian_map( self.G , z_T )
#        _print( 'mag(Gz)' , tf.math.abs( Gz ) )
#        _print( 'ang(Gz)' , tf.math.angle( Gz ) )
        
        ## Keep new state as the old-state for now.
        new_z = z_T
        
        ## Output is the old state and input linearly mapped.
        f = Hx
        
        f = tf.reshape( f , x.shape )
        new_z = tf.reshape( new_z , z.shape )
        
#        _print( 'C f' , f )
#        _print( 'C mag(f)' , tf.math.abs( f ) )
#        _print( 'C ang(f)' , tf.math.angle( f ) )
        
#        _print( 'C z_t' , z_t )
#        _print( 'C mag(z_t)' , tf.math.abs( z_t ) )
#        _print( 'C ang(z_t)' , tf.math.angle( z_t ) )
        
        # check for shape compatibility & re-nest if the previous one was nested.
        y = [ f ] if tf.nest.is_nested( inputs ) else f
        new_z = [ new_z ] if tf.nest.is_nested( states ) else new_z
        
        self.cnt.assign_add([1])
        
        return y , new_z
        
        
    def get_config(self):
        base_config = super(HopfBifurCpxRNNCell, self).get_config()
        config = {
                    "units": self.units,
                    "activation": tf.keras.activations.serialize(self.activation)
                    }
        return dict(list(base_config.items()) + list(config.items()))
        

    
    @tf.function
    def gen_eigen_G( self , eigs ):
        eigs = tf.reshape( eigs , [eigs.shape[-1]] )
        exps = tf.math.exp( eigs )
        D = tf.linalg.diag( exps )
        eigs = tf.expand_dims( eigs , 0 )
        exps = tf.expand_dims( exps , 0 )
        eigs_diff = ( tf.linalg.adjoint( eigs ) - eigs )
        exps_diff = ( tf.linalg.adjoint( exps ) - exps )
        diff = tf.math.divide_no_nan( exps_diff , eigs_diff )
        G = diff + D
        return tf.reshape( G , [eigs.shape[-1]]*2 )

    @tf.custom_gradient
    def hermitian_map( self , H , x ):
        
        '''
            Use this method so eigenvalues can keep their order when updating. Also, 
            eases access to the unitary component.
            We trade 2x multiplications on the forward-pass for not needing an eigh() 
            function call on the backward pass.
            x:    ( 1 x n )     vector of length state-size.
            H:    ( n x n )     Hermitian matrix held as a weight paramter.
            lmda: ( n )         strictly positive array of length state_size.
            '''
        
        y = tf.linalg.matmul( H , x )
#        _print( 'y' , y )
        
        def grad( gL ):
            
            '''
            The following directional derivative calculation is based on the paper by Najfeld and Havel.
            Cite: I. Najfeld and T. F. Havel, “Derivatives of the Matrix Exponential and Their Computation,”
            Advances in Applied Mathematics, vol. 16, no. 3, pp. 321–375, Sep. 1995, doi: 10.1006/aama.1995.1017.
            '''
            
#            _print( 'gL' , gL , summarize = -1 )
            
            # Cast to a higher floating point to later cast down and remove floating point errors
            gL2 = tf.cast( gL , dtype=tf.complex128 )
            x2 = tf.cast( x , dtype=tf.complex128 )
            y2 = tf.cast( y , dtype=tf.complex128 )
            H2 = tf.cast( H , dtype=tf.complex128 )
            
            lmda , U = tf.linalg.eig( H2 )
            
            D = tf.linalg.diag( lmda )
            adjU = tf.linalg.adjoint( U )
            
            lmda2 = tf.cast( lmda , dtype=tf.complex128 )
            D2 = tf.cast( D , dtype=tf.complex128 )
            U2 = tf.cast( U , dtype=tf.complex128 )
            adjU2 = tf.cast( adjU , dtype=tf.complex128 )
            
#            V = ( tf.linalg.matmul( x2 , gL2 , adjoint_b=True ) + tf.linalg.matmul( gL2 , x2 , adjoint_b=True ) ) / 2. 
            V = tf.linalg.matmul( gL2 , x2 , adjoint_b=True )
            V = tf.math.divide_no_nan(
                V,
                tf.math.sqrt( tf.linalg.trace( tf.linalg.matmul( V , V , adjoint_a=True ) ) )
            )
            
            G = self.gen_eigen_G( lmda2 )
            
            Vbar = tf.linalg.matmul( tf.linalg.matmul( adjU2 , V ) , U2 )
            GV = tf.math.multiply( Vbar , G )
#            _print( 'GV' , GV )
            
            Dv_H2 = tf.linalg.matmul( tf.linalg.matmul( U2 , GV ) , adjU2 )
            gx2 = tf.linalg.matmul( H2 , gL2 , adjoint_a=True )
            
            gx = tf.cast( gx2 , dtype=x.dtype)
            gH = tf.cast( Dv_H2 , dtype=H.dtype )
            
            return gH , gx
        
        return y , grad
        
        
    @tf.custom_gradient
    def map_unitary_kernel( self , input , kernel ):
        
        output = tf.linalg.matmul( input , kernel )
        
        def grad( gL ):
            '''
            Following weight updates by projecting a 'closest' unitary matrix and implementing a different optimization function.
            B. Kiani, R. Balestriero, Y. LeCun, and S. Lloyd,
            “projUNN: efficient method for training deep networks with unitary matrices.”
            arXiv, Oct. 13, 2022. Accessed: Jun. 27, 2023. [Online]. Available: http://arxiv.org/abs/2203.05483
            '''
            gL_cT = tf.linalg.matrix_transpose( tf.math.conj( gL ) )
            J = tf.linalg.matmul( gL_cT , input )
            J_cT = tf.linalg.matrix_transpose( J , conjugate=True )
            KJK = tf.linalg.matmul( tf.linalg.matmul( kernel , J_cT ) , kernel )
            J_sub_rMJ = tf.math.divide( tf.math.subtract( J , KJK ) , 2. )
            lMJcT = tf.linalg.matrix_transpose( kernel , conjugate=True )
            gi = tf.linalg.matmul( gL , lMJcT )
            gk = tf.linalg.matmul( lMJcT , J_sub_rMJ )
            return gi , gk
            
        return output , grad




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
        
        assert self.state_size >= input_size, 'If state size is less than input size the input to state mapping is lossy.'
        
        self.sqrt_state_size = tf.constant(
            tf.math.sqrt(
                tf.cast(
                    [ self.state_size ],
                    dtype=self.dtype
                )
            ),
            dtype=self.dtype
        )
        
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
            return (shape[0],) + shape[2:]            # remove the timestep from the input_shape
        
        step_input_shape = tf.nest.map_structure( get_step_input_shape , input_shape )
        
        # Build cell(s)
        if not self.cell.built:
            with tf.name_scope(self.cell.name):
                self.cell.build( step_input_shape )         # Last build shape -> self.cells.build( input_shape[1::] )
                assert self.cell.built , 'HopfBifurCpxRNNLayer failed to build.'
        

        state_init = np.arange(self.state_size) / self.state_size 
        state_init = state_init + state_init[1] / 2.
        state_init = np.exp( -2.j*np.pi * state_init )
        state_init = tf.reshape( tf.cast( state_init , dtype=self.dtype ) , [ 1 , self.state_size] )
        self.state = tf.Variable(
            initial_value=state_init,
            shape=( 1 , self.state_size ),
            dtype=self.dtype,
            trainable=False,
            name='state_'+self.name
        )
#        _print('self.state',tf.transpose(self.state),summarize=-1)
#        exit(0)
        
        rnginit = np.arange( input_size ) + 0.5
        rnginit = tf.reshape( rnginit , [ 1 , input_size ])
        rnginit = tf.cast( rnginit , dtype=self.dtype )
        self.inrng = tf.constant(
            rnginit,
            dtype=self.dtype,
            name='in_rng_'+self.name
        )
        
        rnginit = np.arange( self.output_size ) + 0.5
        rnginit = tf.reshape( rnginit , [ 1 , self.output_size ])
        rnginit = tf.cast( rnginit , dtype=self.dtype )
        self.outrng = tf.constant(
            rnginit,
            dtype=self.dtype,
            name='out_rng_'+self.name
        )
        
        super( HopfBifurCpxRNNLayer, self ).build( input_shape )
        
        self.cnt = tf.Variable( [0] , trainable=False , dtype=tf.int32 )
        
        self.built = True
        
        
        
    @tf.function
    def generate_bases( self , state , range ):
        expang = tf.math.exp( 1.j * tf.math.angle( state ) )
        root = tf.linalg.matrix_transpose( expang )
        bases = tf.math.divide_no_nan(
            tf.math.pow( root , range ),
            self.sqrt_state_size
        )
        return tf.reshape( bases , [ state.shape[-1] , range.shape[-1] ] )
    
    
    def call(self, inputs, training=False ):
        
#        tf.print('\n'*8+'=='*25+' Layer '+'=='*25+'\nCount:',self.cnt[0],'\n')
#        _print( 'L inputs' , inputs )
        
        cast_inputs = tf.cast( inputs, dtype=self.dtype )
#        _print( 'L cast_inputs' , cast_inputs )
        
        # Apply forward mapping -> Maybe try Hilbert Transform instead (idk)
        input_bases = self.generate_bases( self.state , self.inrng )
#        _print( 'L input_bases' , input_bases )
        
        proj_inputs = tf.linalg.matmul( input_bases , cast_inputs , transpose_b=True )
        proj_inputs = tf.linalg.matrix_transpose( proj_inputs )
#        _print( 'L proj_inputs' , proj_inputs )
        
#        '''
        rnn_return = self.rnn_call(
            self.cell,
            proj_inputs,
            self.state,
            time_major=True,            # Provides our 'cell' layer with single input tensors instead of entire batch.
            return_all_outputs=True,    # If True -> seq_out has batch sized output; else, single output
            training = training
        )
        last_out , all_out , new_state = rnn_return
        
#        tf.print('\n'+' *'*15+' RNN Returned '+'* '*15+'\n')
#        _print( 'L last_out', last_out)
        
#        _print( 'L all_out', all_out)
#        _print( 'L ang(all_out)', tf.math.angle( all_out ) )
        
#        _print( 'L new_state', new_state )
#        _print( 'L ange(new_state)', tf.math.angle( new_state ) )
        
#        _print( 'L self.state', self.state )
        
        tf.debugging.check_numerics( tf.math.real( last_out ) , 'a last_out element is nan' )
        tf.debugging.check_numerics( tf.math.real( all_out ) , 'a all_out element is nan' )
        tf.debugging.check_numerics( tf.math.real( new_state ) , 'a new_state element is nan' )
#        '''
        
        self.state.assign( new_state )
        
        output_bases = self.generate_bases( new_state , self.outrng )
#        _print( 'output_bases' , output_bases)
        
        out_map = tf.linalg.matmul( output_bases , all_out , adjoint_a=True , transpose_b=True )
#        _print( 'L A) out_map' , out_map )
        
        out_map = tf.linalg.matrix_transpose( out_map )
#        _print( 'L B) out_map' , out_map )
        
        max_img_val = tf.math.reduce_max( tf.math.imag( out_map ) )
#        tf.debugging.assert_less( max_img_val , 1.e-3, 'imaginary values went above 1.e-5 issue with recurrance.' )
#        tf.print('L img:', max_img_val )
        
        output = tf.math.real( out_map )
#        _print( 'output' , output )
        
        self.cnt.assign_add([1])
        
        return output
        
        
    @tf.custom_gradient
    def print_grad( self , inpt ):
        out = inpt
        def grad( gL ):
            _print( 'gL' , gL )
            return gL
        return out , grad
        
        
    def get_config(self):
        base_config = super(HopfBifurCpxRNNLayer, self).get_config()
        if 'cells' in base_config: del base_config['cells']
        return base_config
        
        
    @classmethod
    def from_config(self, config):
        return self( **config )
        
        
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