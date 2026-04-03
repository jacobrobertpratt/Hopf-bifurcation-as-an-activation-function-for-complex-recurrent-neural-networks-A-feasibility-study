import os
import math
import collections
import time 

import numpy as np
import scipy
from scipy.stats import unitary_group
#from scipy.integrate import odeint, solve_ivp

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

''' Local Imports '''
from proj_utils import _print
from activations import HopfBifurA , HopfBifurB
from initializers import Hermitian , SkewHermitian , Unitary , RandStandardNormal

'''
TODO:
- Add multiple cells
- Copy this code over to the use_RNN_class version to see if performance is different.
- Try and see what the output of the ODESover looks like and if we can use some regularizer
    on that data for the output state. 
- Make stuff into complex next.
'''


@tf.keras.utils.register_keras_serializable('hopfbifur_cpx_layer')
class HopfBifurCpxRNNCell(tf.keras.layers.Layer):
    
    def __init__(
                    self,
                    state_size,
                    activation = None,
                    do_save= False,
                    **kwargs
                  ):
        
        super( HopfBifurCpxRNNCell , self ).__init__( **kwargs )
        
        self.do_save = do_save
        
        # Size of input cells
        self.state_size = state_size
        
        # Set default activation or input activation.
#        self.activation = HopfBifurA( dtype = self.dtype ) if activation is None else activation
        self.activation = HopfBifurB( dtype = self.dtype ) if activation is None else activation
        
        
    def build( self, input_shape ):
        
        self.input_size = input_shape[0]
        
#        print( 'input_shape' , input_shape )
#        print( 'input_size' , self.input_size )
#        print( 'state_size' , self.state_size )
        
        do_train = True    # True False
        
        U = Unitary( name = 'U' , save = self.do_save )
        U = U( ( self.state_size - 1 , self.state_size - 1 ) , self.dtype )
        
        W = Unitary( name = 'W' , save = self.do_save )
        W = W( ( self.state_size - 1 , self.state_size - 1 ) , self.dtype )
        
        def get_Cu( shape , dtype ):
            return ( U + tf.linalg.adjoint( U ) ) / 2.
        self.Cu = self.add_weight(
            name = 'Cu_herm_' + self.name,
            shape = ( self.state_size - 1 , self.state_size - 1 ),
            dtype = self.dtype,
            initializer = get_Cu,
            trainable = do_train
        )
        
        def get_Su( shape , dtype ):
            return ( U - tf.linalg.adjoint( U ) ) / 2.j
        self.Su = self.add_weight(
            name = 'Su_herm_' + self.name,
            shape = ( self.state_size - 1 , self.state_size - 1 ),
            dtype = self.dtype,
            initializer = get_Su,
            trainable = do_train
        )
        
        def get_Cw( shape , dtype ):
            return ( W + tf.linalg.adjoint( W ) ) / 2.
        self.Cw = self.add_weight(
            name = 'Cw_herm_' + self.name,
            shape = ( self.state_size - 1 , self.state_size - 1 ),
            dtype = self.dtype,
            initializer = get_Cw,
            trainable = do_train
        )
        
        def get_Sw( shape , dtype ):
            return ( W - tf.linalg.adjoint( W ) ) / 2.j
        self.Sw = self.add_weight(
            name = 'Sw_herm_' + self.name,
            shape = ( self.state_size - 1 , self.state_size - 1 ),
            dtype = self.dtype,
            initializer = get_Sw,
            trainable = do_train
        )
        
        ## CHANGED ## 
        self.step = tf.Variable( [ 0 ] , trainable = False , dtype = tf.int32 )
        
        super( HopfBifurCpxRNNCell , self ).build( input_shape )
        
        self.built = True
        
    def call( self , input , state , training = False ):
        
        ## CHANGED ## 
#        tf.print( '\n'+'- '*20+' Cell '+'- '*20+'\nCount:' , self.step[0] , '\n' )
        
        z = state[0] if tf.nest.is_nested( state ) else state
        x = input[0] if tf.nest.is_nested( input ) else input
#        _print( 'x' , x )
#        _print( 'z' , z )
        
        Cu = self.Cu
        Su = self.Su
        Cw = self.Cw
        Sw = self.Sw
        
        z01 , z1n = z[0:1] , z[1::]
        x01 , x1n = x[0:1] , x[1::]
        
        uz , nz = self.mke_unt_vec( z1n ) , self.mke_nrm_vec( z1n )
        ux , nx = self.mke_unt_vec( x1n ) , self.mke_nrm_vec( x1n )
        
        
        Cz = self.hermitian_map( Cu , uz )
        Sz = self.hermitian_map( Su , uz )
        
        Cx = self.hermitian_map( Cw , ux )
        Sx = self.hermitian_map( Sw , ux )
        
        Uz = Cz + 1.j*Sz
        Ux = Cx + 1.j*Sx
        
        ''' # HopfBifurA Version 1 - Kind of Learns
        dz = Uz + Ux
        dx = self.activation( dz , Uz * tf.math.conj( Ux ) , tf.zeros_like( dz ) )
        z_t = tf.concat( [ z01 , Uz * dz ] , 0 )
        x_t = tf.concat( [ x01 , Ux * dx ] , 0 )
#       '''

        ''' # HopfBifurB Version 1 - Kind of Learns
        dz = Ux + Uz
        dx = self.activation( dz , Ux , Uz )
        z_t = tf.concat( [ z01 , dz * Uz ] , 0 )
        x_t = tf.concat( [ x01 , dx * Ux ] , 0 )
#       '''

        ''' # HopfBifurB Version 2 -> Much Much Faster with time steps added
        dz = Ux + Uz
        dx = self.activation( self.step[0] , dz , Ux , Uz )
        z_t = tf.concat( [ z01 , dz * Uz ] , 0 )
        x_t = tf.concat( [ x01 , dx * Ux ] , 0 )
#       '''
        
#        ''' # HopfBifurB Version 3 ->
        c = ux * uz
        a = Ux * tf.math.conj( Uz )
        b = tf.ones_like( c ) - Ux * Ux - Uz * Uz
        dx = self.activation( self.step[0] , c , a , b )
        z_t = tf.concat( [ z01 , tf.math.conj( dx ) * Uz ] , 0 )
        x_t = tf.concat( [ x01 , dx * Ux ] , 0 )
#       '''
        
#        dx = tf.keras.activations.tanh( dz )
#        _print( 'z_t' , z_t )
#        _print( 'x_t' , x_t )
        
        z_out = [ z_t ] if tf.nest.is_nested( state ) else z_t
        x_out = [ x_t ] if tf.nest.is_nested( input ) else x_t
        
        ## CHANGED ## 
        self.step.assign( tf.math.floormod( self.step + 1 ,  ( self.state_size - 1 ) * 2 ) )
        
        return x_out , z_out
        

    @tf.function
    def make_Gmat( self , lmda ):
        func = tf.math.exp( lmda )
        _func = tf.expand_dims( func , 0 )
        _eigs = tf.expand_dims( lmda , 0 )
        G = tf.math.divide_no_nan(
            _func - tf.linalg.matrix_transpose( _func ),
            _eigs - tf.linalg.matrix_transpose( _eigs )
        ) + tf.linalg.diag( func )
        return G

    @tf.custom_gradient
    def hermitian_map( self , H , x ):
        
#        self.test_herm_mat( H )
        
        y = tf.linalg.matmul( H , x )
#        _print( 'A) H(y)' , y )
        
        def grad( gL ):
            ''' SOURCE:
            The following directional derivative calculation is based on the paper by Najfeld and Havel.
            Cite: I. Najfeld and T. F. Havel, “Derivatives of the Matrix Exponential and Their Computation,”
            Advances in Applied Mathematics, vol. 16, no. 3, pp. 321–375, Sep. 1995, doi: 10.1006/aama.1995.1017.
            '''
            
#            _print( 'y' , y )
#            _print( 'x' , x )
#            _print( 'H' , H )
#            _print( 'gL' , gL )
            
            lmda , W = tf.linalg.eig( H )
            adjW = tf.linalg.adjoint( W )
#            _print( 'W' , W )
#            _print( 'lmda' , lmda )
            
            nx = self.mke_nrm_mat( x )
#            ny = self.mke_nrm_mat( y )
#            nH = self.mke_nrm_mat( H )
            
#            V = tf.linalg.matmul( x , gL , adjoint_b = True ) + tf.linalg.matmul( gL , x , adjoint_b = True )
            V = self.extr( x , gL ) + self.extr( gL , x )
            V = self.mke_nrm_mat( V )
            
#            Vbar = tf.linalg.matmul( tf.linalg.matmul( adjW , V ) , W )
            GV = tf.math.multiply(
                self.make_Gmat( lmda ),
                tf.linalg.matmul( tf.linalg.matmul( adjW , V ) , W )
            )
            
            Dv_H = tf.linalg.matmul( tf.linalg.matmul( W , GV ) , adjW )
#            _print( 'Dv_H' , Dv_H )
            
#            gH = tf.linalg.adjoint( Dv_H )
#            gH = tf.linalg.adjoint( Dv_H ) @ self.mke_nrm_mat( Ex )
            
            gH = tf.linalg.matmul( Dv_H , self.mke_nrm_mat( self.extr( nx , nx ) ) , adjoint_a = True )
#            _print( 'gH' , gH )
            
#            gx = (-1.+0.j)*tf.linalg.matmul( H , gL , adjoint_a = True )
            gx = gL * tf.linalg.matmul( Dv_H , nx , adjoint_a = True )
#            _print( 'gx' , gx )
            
            return gH , gx
            
        return y , grad
        
    @tf.custom_gradient
    def print_grad( self , inpt ):
        out = inpt
        def grad( gL ):
            _print( 'gL' , gL )
            return gL
        return out , grad
        
    def get_config(self):
        base_config = super(HopfBifurCpxRNNCell, self).get_config()
        config = {
                    "units": self.units,
                    "activation": tf.keras.activations.serialize(self.activation)
                    }
        return dict(list(base_config.items()) + list(config.items()))
        
        
    @tf.function
    def mke_unt_vec( self , vec ):
        return tf.math.divide_no_nan( vec , tf.cast( tf.math.abs( vec ) , dtype = vec.dtype ) )
        
    @tf.function
    def mke_nrm_vec( self , vec ):
        nv , _ = tf.linalg.normalize( vec , ord = 2 )
        return nv
        
    @tf.function
    def mke_nrm_mat( self , mat ):
        nm , _ = tf.linalg.normalize( mat , ord = 2 , axis = [ -2 , -1 ] )
        return nm
        
    @tf.function
    def mke_unt_mat( self , mat ):
        return tf.math.divide_no_nan( mat , tf.cast( tf.math.abs( mat ) , dtype = mat.dtype ) )
        
    @tf.function
    def innr( self , a , b ):
        return tf.linalg.matmul( a , b, adjoint_a = True )
        
    @tf.function
    def extr( self , a , b ):
        return tf.linalg.matmul( a , b, adjoint_b = True )
        
    @tf.function
    def mke_vec_to_mat( self , vec ):
        re , im = tf.math.real( vec ) , tf.math.imag( vec )
        return tf.reshape( tf.concat( [ re , -im , im , re ] , 1 ) , [ vec.shape[0] , 2 , 2 ] )
        
    @tf.function
    def mke_mat_to_vec( self , mat ):
        return tf.complex( mat[::,0:1,0] , -mat[::,0:1,1] )
    
    @tf.function
    def test_unit_mat( self , U ):
        eye = tf.eye( U.shape[0] , dtype = U.dtype )
        ichkmax = tf.math.reduce_max( tf.math.abs( eye - tf.linalg.matmul( U , U , adjoint_b = True ) ) )
#        _print( 'ichkmax( U )' , ichkmax )
        tf.debugging.assert_less( ichkmax , 1.e-5, 'HopfBifurCell - > failed unitarity check.' )
        
    @tf.function
    def test_herm_mat( self , H ):
        hchkmax = tf.math.reduce_max( tf.math.abs( H - tf.linalg.adjoint( H ) ) )
#        _print( 'hchkmax( H )' , hchkmax )
#        tf.debugging.assert_less( hchkmax , 1.e-5, 'HopfBifurCell - > failed hermitian check.' )
        
    @tf.function
    def cpx_rot_to_rel( self , vec ):
        return vec * tf.math.conj( self.mke_unt_vec( vec ) )
        
        
        
        
@tf.keras.utils.register_keras_serializable('hopfbifur_cpx_layer')
class HopfBifurCpxRNNLayer(tf.keras.layers.Layer):
    
    def __init__(
            self,
            state_size,
            activation = None,
            **kwargs
        ):
        
        super( HopfBifurCpxRNNLayer , self ).__init__( **kwargs )
        
        # The state size of each cells, if None then defaults to square-root of input dimensions and cast as an integer.
        self.state_size = state_size
        
        # To hold the cell(s) for the model.
        self.cell = None
        
        # Set the activation function
        self.activation = activation
        
        
    def build( self , input_shape ):
        
        do_save = True
        
        self.batch_size = input_shape[0]
        self.input_size = input_shape[1]
        
        self.sqrisz = tf.constant(
            tf.cast( [ tf.math.sqrt( 0. + self.input_size ) ] , dtype = self.dtype ),
            dtype = self.dtype
        )
        self.sqrssz = tf.constant(
            tf.cast( [ tf.math.sqrt( 0. + self.state_size ) ] , dtype = self.dtype ),
            dtype = self.dtype
        )
        
        if do_save:
            state_init = None
            if not os.path.exists('state_layer.npy'):
                state_init = np.random.rand( self.input_size )
                np.save( 'state_layer.npy' , state_init , allow_pickle = True )
            else:
                state_init = np.load( 'state_layer.npy' , allow_pickle = True )
        else:
            state_init = np.random.rand( self.input_size )
        
        ssz = self.state_size + 1
        state_init = tf.cast( state_init , dtype = tf.float32 )
        state_init = tf.signal.rfft( state_init )
        self.fftshp = state_init.shape[0]
        state_init = state_init[0:ssz].reshape( -1 , 1 )
        state_init = tf.reshape( tf.cast( state_init , dtype = self.dtype ) , [ state_init.shape[0] , 1 ] )
        self.state = tf.Variable(
            initial_value = state_init,
            shape = ( state_init.shape[0] , 1 ),
            dtype = self.dtype,
            trainable = False,
            name = 'state_'+self.name
        )
        
        paddif = np.abs( self.state.shape[0] - self.fftshp )
        zropad = tf.cast( tf.reshape( [[0.]*paddif]*self.batch_size , [ self.batch_size , 1 , paddif ] ) , dtype = self.state.dtype )
        self.zropad = tf.constant( zropad , dtype = self.state.dtype )
#        _print('zropad',self.zropad)
#        exit(0)
        
        # Generate cell -> Only use 1 for now, until it's working better.
        if self.cell is None:
            self.cell = HopfBifurCpxRNNCell(
                state_size = self.state.shape[0],
                activation = self.activation,
                do_save = do_save,
                dtype = self.dtype,
                name = 'hopf_cell'
            )
        
        # From tf.keras.layers.basernn
        def get_step_input_shape(shape):
            if isinstance( shape , tf.TensorShape ):
                shape = tuple(shape.as_list())
            return (shape[0],) + shape[2:]            # remove the timestep from the input_shape
        step_input_shape = tf.nest.map_structure( get_step_input_shape , input_shape )
        
        # Build cell(s)
        if not self.cell.built:
            with tf.name_scope(self.cell.name):
                self.cell.build( ( self.state.shape[0] , 1 ) )
                assert self.cell.built , 'HopfBifurCpxRNNLayer failed to build.'
        
        super( HopfBifurCpxRNNLayer , self ).build( input_shape )
        
        self.cnt = tf.Variable( [ 0 ] , trainable = False , dtype = tf.int32 )
        
        self.built = True
        
    @tf.function
    def map_input( self , x , sz ):
        xT = tf.linalg.matrix_transpose( x )
        rfft = tf.signal.rfft( xT )[::,::,0:sz]
        return tf.linalg.matrix_transpose( rfft )
    
    @tf.function
    def map_output( self , x ):
        xT = tf.linalg.matrix_transpose( x )
        xTpad = tf.concat( [ xT , self.zropad ] , - 1 )
        rfft = tf.signal.irfft( xTpad )
        return tf.linalg.matrix_transpose( rfft )
    
    def call( self , input , training = False ):
        
#        tf.print( '\n'*8+'=='*25+' Layer '+'=='*25+'\nCount:',self.cnt[0],'\n' )
#        _print( 'input' , input )
        
        _state = self.state
        _input = self.map_input( input , self.state.shape[0] )
#        _print( '_input' , _input )
#        _print( '_state' , _state )
        
#        '''
        rnn_return = self.rnn_call(
            self.cell,
            _input,
            _state,
            time_major = True,            # Provides our 'cell' layer with single input tensors instead of entire batch.
            return_all_outputs = True,    # If True -> seq_out has batch sized output; else, single output
            training = training
        )
        _clast , _cout , _cstate = rnn_return
        '''
        _cout = _input
        _cstate = _state
#        '''
        
#        tf.print('\n'+' *'*15+' RNN Returned '+'* '*15+'\n')
#        _print( '_cout' , _cout )
#        _print( '_cstate' , _cstate )
        
        self.state.assign( _cstate )
        
#        tf.debugging.check_numerics( tf.math.real( _clstate ) , 'a _clstate element is nan' )
#        tf.debugging.check_numerics( tf.math.real( _cstates ) , 'a _cstates element is nan' )
#        tf.debugging.check_numerics( tf.math.real( _coutput ) , 'a _coutput element is nan' )
        
        _output = self.map_output( _cout )
#        _print( '_output' , _output )
        
#        max_img_val = tf.math.reduce_max( tf.math.imag( _output ) )
#        tf.debugging.assert_less( max_img_val , 1.e-5, 'imaginary values went above 1.e-5 issue with irfft() input map.' )
#        tf.print('maximg:', max_img_val )
        
        output = tf.cast( tf.math.real( _output ) , input.dtype )
#        _print( 'output' , output )
        
        self.cnt.assign_add( [ 1 ] )
        
        return output
        
    @tf.function
    def mke_unt_vec( self , vec ):
        return tf.math.divide_no_nan( vec , tf.cast( tf.math.abs( vec ) , dtype = vec.dtype ) )
        
    @tf.function
    def mke_nrm_vec( self , vec ):
        nv , _ = tf.linalg.normalize( vec , ord = 2 )
        return nv
        
    @tf.function
    def innr( self , a , b ):
        return tf.linalg.matmul( a , b, adjoint_a = True )
        
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
    
    
    
    
    
    
    
    
    