import os
import math
import collections
import time 

import numpy as np
import scipy
from scipy.stats import unitary_group

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

''' Local Imports '''
from proj_utils import _print
from activations import HopfBifur
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
                    **kwargs
                  ):
        
        super( HopfBifurCpxRNNCell , self ).__init__( **kwargs )
        
        # Size of input cells
        self.state_size = state_size
        
        # Set default activation or input activation.
        self.activation = HopfBifur( dtype = self.dtype ) if activation is None else activation
        
        
    def build( self, input_shape ):
        
        self.batch_size = input_shape[0]
        self.input_size = input_shape[-1]
        
        sz = self.state_size
        
        # True False
        do_save = False
        do_train = True
        do_conjsym = False
        
        self.S = self.add_weight(
            name = 'S_skew_' + self.name,
            shape = ( sz , sz ),
            dtype = self.dtype,
            initializer = SkewHermitian( name = 'S' , save = do_save ),
            trainable = do_train
        )
        
        self.G = self.add_weight(
            name = 'G_herm_' + self.name,
            shape = ( sz , sz ),
            dtype = self.dtype,
            initializer = Hermitian( name = 'G' , save = do_save ),
            trainable = do_train
        )
        
        self.H = self.add_weight(
            name = 'H_herm_' + self.name,
            shape = ( sz , sz ),
            dtype = self.dtype,
            initializer = Hermitian( name = 'H' , save = do_save ),
            trainable = do_train
        )
        
        self.ones = tf.constant( tf.ones( shape = ( sz , 1 ), dtype = self.dtype ), dtype = self.dtype )
        
        super( HopfBifurCpxRNNCell , self ).build( input_shape )
        
        self.cnt = tf.Variable( [0] , trainable = False , dtype = tf.int32 )
        
        self.built = True
        
        
    def call( self , input , state , training=False ):
        
#        tf.print( '\n'+'- '*20+' Cell '+'- '*20+'\nCount:' , self.cnt[0] , '\n' )
        
        z_in = state[0] if tf.nest.is_nested( state ) else state
        x_in = input[0] if tf.nest.is_nested( input ) else input
#        _print( 'z_in' , z_in )
#        _print( 'x_in' , x_in )
        
        S = self.S
        G = self.G
        H = self.H
        
        z = tf.linalg.matrix_transpose( z_in )
        x = tf.linalg.matrix_transpose( x_in )
#        _print( 'x' , x )
#        _print( 'z' , z )
        
        Gz = self.hermitian_map( G , z )
        Hx = self.hermitian_map( H , x )
        
        Hx2 = self.ones - Hx * tf.math.conj( Hx )
        Gz2 = ( 1.j*self.hermitian_map( G , z ) ) * tf.math.conj( -1.j*self.hermitian_map( G , z ) )
#        _print( 'Hx2' , Hx2 , summarize = -1 )
#        _print( 'Gz2' , Gz2 , summarize = -1 )
        
#        act = self.activation( z_tst , Hx2 , Gz2 )
#        _print( 'act' , act , summarize = -1 )
        
        z_t = Gz + Hx
        x_t = self.activation( z_t , Hx2 , Gz2 )
        _print( 'x_t' , x_t , summarize = -1 )
        _print( 'z_t' , z_t , summarize = -1 )
        
#        tf.debugging.check_numerics( tf.math.real( x_t ) , 'Cell -> x_t element is nan' )
#        tf.debugging.check_numerics( tf.math.real( z_t ) , 'Cell -> z_t element is nan' )
        
        z_t = tf.linalg.matrix_transpose( z_t )
        x_t = tf.linalg.matrix_transpose( x_t )
#        _print( 'z_t' , z_t )
#        _print( 'x_t' , x_t )

        z_out = [ z_t ] if tf.nest.is_nested( state ) else z_t
        x_out = [ x_t ] if tf.nest.is_nested( input ) else x_t
#        _print( 'z_out' , z_out )
#        _print( 'x_out' , x_out )
        
        self.cnt.assign_add( [ 1 ] )
        
        return x_out , z_out
        
        
    @tf.custom_gradient
    def hermitian_map( self , H , x ):
        
        y = tf.linalg.matmul( H , x )
#        y = ( y + tf.math.conj( y )[::-1] ) / 2.
        
        def grad( gL ):
            ''' SOURCE:
            The following directional derivative calculation is based on the paper by Najfeld and Havel.
            Cite: I. Najfeld and T. F. Havel, “Derivatives of the Matrix Exponential and Their Computation,”
            Advances in Applied Mathematics, vol. 16, no. 3, pp. 321–375, Sep. 1995, doi: 10.1006/aama.1995.1017.
            '''
            
#            _print( 'gL' , gL )
#            _print( 'y' , y , summarize=-1 )
#            _print( 'x' , x , summarize=-1 )
#            _print( 'hermitian_map -> gL' , gL , summarize=-1 )
            
            gL2 = tf.cast( gL , dtype = tf.complex128 )
            x2 = tf.cast( x , dtype = tf.complex128 )
            y2 = tf.cast( y , dtype = tf.complex128 )
            H2 = tf.cast( H , dtype = tf.complex128 )
            
            lmda2 , W2 = tf.linalg.eig( H2 )
            adjW2 = tf.linalg.adjoint( W2 )
            
            V = tf.linalg.matmul( gL2 , x2 , adjoint_b=True )
#            V_L = tf.linalg.matmul( x2 , gL2 , adjoint_b=True )
#            V_R = tf.linalg.matmul( gL2 , x2 , adjoint_b=True )
#            V = ( V_L + V_R ) / 2.
            
            func = tf.math.exp( lmda2 )
            _func = tf.expand_dims( func , 0 )
            _eigs = tf.expand_dims( lmda2 , 0 )
            G = tf.math.divide_no_nan(
                _func - tf.linalg.matrix_transpose( _func ),
                _eigs - tf.linalg.matrix_transpose( _eigs )
            ) + tf.linalg.diag( func )
            
            Vbar = tf.linalg.matmul( tf.linalg.matmul( adjW2 , V ) , W2 )
            GV = tf.math.multiply( G , Vbar )
            
            Dv_H2 = tf.linalg.matmul( tf.linalg.matmul( W2 , GV ) , adjW2 )
            
            gx2 = tf.linalg.matmul( H2 , gL2 , adjoint_a = True )
            
            gH = tf.reshape( tf.cast( Dv_H2 , dtype = H.dtype ) , H.shape )
            gx = tf.reshape( tf.cast( gx2 , dtype = x.dtype) , x.shape )
#            _print( 'gH' , gH )
#            _print( 'gx' , gx )
            
            return gH , gx
            
        return y , grad
        
        
    @tf.function
    def gen_rand_unit( self , M ):
        ''' From the paper:
        [1] F. Mezzadri, “How to generate random matrices from the classical compact groups.”
        arXiv, Feb. 27, 2007. doi: 10.48550/arXiv.math-ph/0609050. 
        Available: http://arxiv.org/abs/math-ph/0609050
        (On Page 11, see steps and python code)
        '''
        
        sz = M.shape[-1]
        re = tf.cast( np.random.randn( sz , sz ) , dtype = self.dtype )
        im = tf.cast( np.random.randn( sz , sz ) , dtype = self.dtype )
        O = ( re + 1.j*im ) / tf.math.sqrt( 2.0 )
        
        Q , R = tf.linalg.qr( O )
        E = tf.linalg.diag_part( R )    # Gets eigen values in a [ n-dim ] row-vector
        D = tf.linalg.diag( tf.math.divide_no_nan( E, tf.cast( tf.math.abs( E ) , dtype = E.dtype ) ) )
        
        return tf.math.sign( E ) * Q   # Same -> newQ = tf.math.multiply( Q , tf.linalg.matmul( D , Q ) )
        
    @tf.function
    def LSI_approx( self , M , rank ):
        ''' From the paper:
        [1] C. H. Papadimitriou, P. Raghavan, H. Tamaki, and S. Vempala,
        “Latent Semantic Indexing: A Probabilistic Analysis,”
        Journal of Computer and System Sciences,
        vol. 61, no. 2, pp. 217–235, Oct. 2000, doi: 10.1006/jcss.2000.1711.
        '''
        
        dims = 0. + M.shape[-1]
        
        B = self.gen_rand_unit( M )
        
        C = tf.linalg.matmul( B , M , adjoint_a = True ) * tf.math.sqrt( dims / rank )
        
        S , U , _ = tf.linalg.svd(
            tf.linalg.matmul( C , C , adjoint_b = True ),
            full_matrices = True
        )
        S = tf.expand_dims( S , 0 )
        nU = U / ( tf.math.sqrt( S ) + 1.e-8 )
        
        D = tf.linalg.matmul( C , tf.cast( nU , C.dtype ), adjoint_a = True )
        E = tf.linalg.matmul( M , D )
        
        return E , D
        
    @tf.function
    def calc_projUNN_T( self , M , G , lr = 0.1 ):
        ''' SOURCE:
            B. Kiani, R. Balestriero, Y. LeCun, and S. Lloyd,
            “projUNN: efficient method for training deep networks with unitary matrices.”
            arXiv, Oct. 13, 2022. Accessed: Jun. 27, 2023. [Online]. Available: http://arxiv.org/abs/2203.05483
            Note: Following weight updates by projecting a 'closest' unitary matrix and
                   implementing a different optimization function.
        '''
        
#        _print( 'M' , M )
#        _print( 'G' , G )
        
        adjM = tf.linalg.adjoint( M )
        
        a , b = self.LSI_approx( G , G.shape[-1] )
#        _print( 'a' , a )
#        _print( 'b' , b )
        
#        '''
        ## Paper Version from sudo code ##
        b_a = tf.concat( [ b , a ] , axis = -1 )
        Q , _ = tf.linalg.qr( b_a )
        adjQ = tf.linalg.adjoint( Q )
        
        _a = tf.expand_dims( a , -1 )
        _b = tf.expand_dims( b , -1 )
#        _print( '_a' , _a )
#        _print( '_b' , _b )
        
        ab = tf.linalg.matmul( _a , _b , adjoint_b = True )
        ba = tf.linalg.matmul( _b , _a , adjoint_b = True )
#        _print( 'ab' , ab )
#        _print( 'ba' , ba )
        
        QUabQ = tf.linalg.matmul( tf.linalg.matmul( adjQ , tf.linalg.matmul( adjM , ab ) ) , Q )
#        _print( 'QUabQ' , QUabQ )
        
        QbaUQ = tf.linalg.matmul( tf.linalg.matmul( adjQ , tf.linalg.matmul( ba , M ) ) , Q )
#        _print( 'QbaUQ' , QbaUQ)
        
        QmQ = ( QUabQ - QbaUQ )
#        _print( 'QmQ' , QmQ )
        
        K = tf.math.reduce_sum( QmQ , axis = 0 ) / 2.
#        _print( 'K' , K )
        
        e , E = tf.linalg.eig( K )
        
        '''
        ## Paper Version with github code ##
        a_hat = tf.linalg.matmul( adjM , a )
        b_a = tf.concat( [ b , a_hat ] , axis = -1 )
        Q , _ = tf.linalg.qr( b_a )
        adjQ = tf.linalg.adjoint( Q )
        prj_a = tf.linalg.matmul( adjQ , a_hat )
        prj_b = tf.linalg.matmul( adjQ , b )
        prj_ab = tf.linalg.matmul( prj_a , prj_b , adjoint_b=True ) 
        prj_ba = tf.linalg.matmul( prj_b , prj_a , adjoint_b=True )
        ab_ba = ( prj_ab - prj_ba ) / 2.
        _print( 'ab_ba' , ab_ba )
        e , E = tf.linalg.eig( ab_ba )
#       '''
#        _print( 'evls' , e )
#        _print( 'evcs' , E )
        
        s = tf.reshape( tf.math.expm1( -lr * e ) , [ e.shape[0] ] + [ 1 , 1 ] )
#        _print( 's' , s )
        
        E = tf.expand_dims( E , -1 )
        ExE = tf.linalg.matmul( E , E , adjoint_b = True ) * s
#        _print( 'ExE' , ExE )
        
        ExE = tf.math.reduce_sum( ExE , 0 )
        IExE = tf.eye( ExE.shape[-1] , dtype=ExE.dtype ) + ExE
#        _print( 'IExE' , IExE )
        
        R = tf.linalg.matmul( M , IExE )
#        _print( 'R' , R )
        
        return R
    
    @tf.custom_gradient
    def unitary_map( self , S , x ):
        
        U = tf.linalg.expm( S )
        y = tf.linalg.matmul( U , x )
        y = ( y + tf.math.conj( y )[::-1] ) / 2.
#        _print( 'y' , y )
        
        def grad( gL ):
            
#            _print( 'gL' , gL )
#            _print( 'S' , S )
            
#            U = tf.linalg.expm( S )
#            _print( 'U' , U )
            
            # Calculate Jacobian ( If that is what I calcualted ... I have no idea )
            dU = tf.linalg.matmul( S , U )
            J = tf.math.multiply_no_nan( gL , tf.linalg.adjoint( dU ) )
            J , _ = tf.linalg.normalize( J , ord = 2 , axis = [ -2 , -1 ] )
#            _print( 'J' , J )
            
            # Calculate the projUNN gradient for unitary matrix updates.
            W = self.calc_projUNN_T( U , J , lr = 0.1 )
            
            dW = tf.linalg.matmul( tf.linalg.logm( W ) , W )
            
            dS = tf.linalg.adjoint( dW ) - dU   # Same as sort of ->  dS = ( dW2 - dU )
            nx , _ = tf.linalg.normalize( x , ord = 2 )
            ngL , _ = tf.linalg.normalize( gL , ord = 2 )
#            gS = tf.linalg.adjoint( dS ) @ P @ dS 
#            gS = gS - tf.linalg.adjoint( dS ) @ tf.linalg.adjoint( P ) @ dS
            P = tf.linalg.matmul( ngL , nx , adjoint_b = True )
            adS = tf.linalg.adjoint( dS )
            adSP = adS @ P
            gS = adSP @ dS - adS @ tf.linalg.adjoint( adSP )
#            _print( 'gS' , gS )
            
            gx = gL * tf.linalg.matmul( dW , nx , adjoint_a = True )
#            _print( 'gx' , gx )
            
            return gS , gx
            
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
        
        
        
@tf.keras.utils.register_keras_serializable('hopfbifur_cpx_layer')
class HopfBifurCpxRNNLayer(tf.keras.layers.Layer):
    
    def __init__(
                   self,
                   state_size=None,
                   activation=None,
                   **kwargs
                  ):
        
        super(HopfBifurCpxRNNLayer, self).__init__(**kwargs)
        
        # The state size of each cells, if None then defaults to square-root of input dimensions and cast as an integer.
        self.state_size = state_size
        
        # To hold the cell(s) for the model.
        self.cell = None
        
        # Set the activation function
        self.activation = activation
        
        
    def build( self , input_shape ):
        
        self.batch_size = input_shape[0]
        self.time_size = input_shape[-2]
        self.input_size = input_shape[-1]
        
        sz = self.state_size
        
        do_save = False
        
#        if self.output_size is None: self.output_size = self.input_size
        if self.state_size is None: self.state_size = int(math.sqrt(self.input_size))*2
        
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
        
        state_init = np.random.rand( self.state_size )
#        state_init = np.arange(self.state_size) / self.state_size 
#        state_init = state_init + state_init[1] / 2.
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
        
        rnginit = np.arange( self.input_size ) + 0.5
        rnginit = tf.reshape( rnginit , [ 1 , self.input_size ])
        rnginit = tf.cast( rnginit , dtype = self.dtype )
        self.rng = tf.constant( rnginit, dtype=self.dtype, name='rng_'+self.name )
        
        self.sqrt_state_size = tf.constant( tf.math.sqrt( tf.cast( [ self.state_size ], dtype=self.dtype ) ), dtype=self.dtype )
        
        super( HopfBifurCpxRNNLayer , self ).build( input_shape )
        
        self.cnt = tf.Variable( [ 0 ] , trainable = False , dtype = tf.int32 )
        
        self.built = True
        
        
    @tf.function
    def generate_bases( self , state , range ):
        expang = tf.math.exp( 1.j * tf.math.angle( state ) )
        root = tf.linalg.matrix_transpose( expang )
        bases = tf.math.divide_no_nan( tf.math.pow( root , range ), self.sqrt_state_size )
        return tf.reshape( bases , [ state.shape[-1] , range.shape[-1] ] )
        
        
    def call( self , input , training = False ):
        
#        tf.print( '\n'*8+'=='*25+' Layer '+'=='*25+'\nCount:',self.cnt[0],'\n' )
#        _print( 'input' , input )
        
        _input = tf.cast( input , dtype = self.dtype )
        _state = self.state
#        _print( '_input' , _input )
#        _print( '_state' , _state )
        
#        _ibase = self.generate_bases( self.state , self.rng )
#        _ustate = self.state / tf.math.abs( self.state )
#        _ibase = tf.math.pow( ustate , self.rng , adjoint_a = True )
#        _input = tf.linalg.matmul( _ibase , _input , transpose_b = True )
#        _input = tf.linalg.matrix_transpose( _input )
#        _print( '_ibase' , _ibase )
#        _print( '_input' , _input )
        
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
#        _clast = tf.unstack( _rinput )[-1]
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
        
#        _obase = self.generate_bases( _cstate , self.rng )
#        _cout = tf.linalg.matmul( _obase , _cout , adjoint_a = True , transpose_b = True )
#        _output = tf.linalg.matrix_transpose( _cout )
        _output = _cout
#        _print( '_output' , _output )
        
#        max_img_val = tf.math.reduce_max( tf.math.imag( _output ) )
#        tf.print('maximg:', max_img_val )
#        tf.debugging.assert_less( max_img_val , 1.e-5, 'imaginary values went above 1.e-5 issue with recurrance.' )
        
        output = tf.cast( tf.math.real( _output ) , input.dtype )
        output = tf.reshape( output , input.shape )
#        _print( 'output' , output )
        
        self.cnt.assign_add( [ 1 ] )
        
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


@tf.function
def gcd( self , state ):
_stps = tf.cast( tf.math.abs( (2*np.pi / tf.math.angle( state ) ) * 1e6 ) , dtype = tf.int32 )
return tf.numpy_function( np.gcd.reduce , [ _stps ] , tf.int32 )[0]

@tf.function
def lcm( self , state ):
sz = self.state_size
_state = tf.reshape( state , [ sz ] )
ang = tf.math.abs( 2*np.pi / tf.math.angle( _state ) )[0:sz//2]
ang = ang / tf.math.reduce_min( ang )
#        _print( 'ang' , ang )

ang = tf.cast( ang , dtype = tf.int64 )
lcm = tf.numpy_function( np.lcm.reduce , [ ang ] , tf.int64 )
#        _print( 'lcm' , lcm )
return lcm
#        _ang = tf.cast( tf.math.abs( tf.math.angle( state ) ) , dtype = tf.int32 )
#        return tf.numpy_function( np.lcm.reduce , [ _ang ] , tf.int32 )[0]

@tf.custom_gradient
def make_base( self , ustate , size ):

gcd = self.gcd( ustate )
#        _print( 'gcd' , gcd )
#        _print( 'rng' , rng )

lcm = self.lcm( ustate )
#        _print( 'lcm' , lcm )

rng = tf.linspace( 0. , tf.cast( lcm , dtype = tf.float32 ) , size )
#        rng = tf.linspace( 0. , 2*np.pi*lcm , size )
#        rng = tf.range( size )
#        _print( 'rng' , rng )

rng = tf.expand_dims( tf.cast( [ rng ] , dtype = ustate.dtype ) , 0 )
#        _print( 'rng' , rng , summarize = -1 )

base = tf.math.pow( ustate , rng )
base = base / tf.math.abs( base ) # Clean-up floating point numbers
base = tf.reshape( base , [ self.state_size , size ] )

def grad( gL ):
gu = tf.ones_like( ustate )*gL
return gu , None

return base , grad



class HopfCell( tf.keras.layers.Layer ):
    
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