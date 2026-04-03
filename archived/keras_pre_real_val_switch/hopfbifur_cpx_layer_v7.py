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
                    do_save= False,
                    **kwargs
                  ):
        
        super( HopfBifurCpxRNNCell , self ).__init__( **kwargs )
        
        self.do_save = do_save
        
        # Size of input cells
        self.state_size = state_size
        
        # Set default activation or input activation.
        self.activation = HopfBifur( dtype = self.dtype ) if activation is None else activation
        
        
    def build( self, input_shape ):
        
        self.batch_size = input_shape[0]
        self.input_size = input_shape[-1]
        
        # True False
        do_train = False
        self.do_conjsym = False
        
        isz = self.input_size
        ssz = self.state_size - 1
        
        self.sqrisz = tf.constant( tf.cast( [ tf.math.sqrt( 0. + isz ) ] , dtype = self.dtype ) , dtype = self.dtype )
        self.sqrssz = tf.constant( tf.cast( [ tf.math.sqrt( 0. + ssz ) ] , dtype = self.dtype ) , dtype = self.dtype )
        
        self.A = self.add_weight(
            name = 'A_herm_' + self.name,
            shape = ( ssz , ssz ),
            dtype = self.dtype,
            initializer = Hermitian( name = 'A' , save = self.do_save ),
            trainable = do_train
        )
        
        self.B = self.add_weight(
            name = 'B_herm_' + self.name,
            shape = ( ssz , ssz ),
            dtype = self.dtype,
            initializer = Hermitian( name = 'B' , save = self.do_save ),
            trainable = do_train
        )
        
        def init_cos_sin_U( shape , dtype ):
            U = Unitary( name = 'U' , save = self.do_save ) # conjsym = self.do_conjsym )
            U = U( ( ssz , ssz ) , self.dtype )
            adjU = tf.linalg.adjoint( U )
            return tf.stack( [ ( U + adjU ) / 2. , ( U - adjU ) / 2.j ] )
        
        # Cosine and Sine components of U are the [0] and [1] dimensions.
        self.U = self.add_weight(
            name = 'U_stkd_' + self.name,
            shape = ( 2 , ssz , ssz ),
            dtype = self.dtype,
            initializer = init_cos_sin_U,
            trainable = do_train
        )
        
        def init_cos_sin_W( shape , dtype ):
            W = Unitary( name = 'W' , save = self.do_save ) # conjsym = self.do_conjsym )
            W = W( ( ssz , ssz ) , self.dtype )
            adjW = tf.linalg.adjoint( W )
            return tf.stack( [ ( W + adjW ) / 2. , ( W - adjW ) / 2.j ] )
        
        # Cosine and Sine components of U are the [0] and [1] dimensions.
        self.W = self.add_weight(
            name = 'W_stkd_' + self.name,
            shape = ( 2 , ssz , ssz ),
            dtype = self.dtype,
            initializer = init_cos_sin_W,
            trainable = do_train
        )
        
        self.sones = tf.constant( tf.ones( shape = ( ssz + 1 , 1 ), dtype = self.dtype ), dtype = self.dtype )
        
        self.eye = tf.constant ( tf.eye( ssz , dtype = self.dtype ) , dtype = self.dtype )
        
        self.cnt = tf.Variable( [ 0 ] , trainable = False , dtype = tf.int32 )
        
        super( HopfBifurCpxRNNCell , self ).build( input_shape )
        
        self.built = True
        
        
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
        
    @tf.function
    def mke_vec_to_mat( self , vec ):
        re , im = tf.math.real( vec ) , tf.math.imag( vec )
        return tf.reshape( tf.concat( [ re , -im , im , re ] , 1 ) , [ vec.shape[0] , 2 , 2 ] )
        
    @tf.function
    def mke_mat_to_vec( self , mat ):
        return tf.complex( mat[::,0:1,0] , -mat[::,0:1,1] )
    
    @tf.function
    def test_unit_mat( self , U ):
        ichkmax = tf.math.reduce_max( tf.math.abs( self.eye - tf.linalg.matmul( U , U , adjoint_b = True ) ) )
#        _print( 'ichkmax( U )' , ichkmax )
        tf.debugging.assert_less( ichkmax , 1.e-5, 'HopfBifurCell - > failed unitarity check.' )
        
    @tf.function
    def test_herm_mat( self , H ):
        hchkmax = tf.math.reduce_max( tf.math.abs( H - tf.linalg.adjoint( H ) ) )
#        _print( 'hchkmax( H )' , hchkmax )
        tf.debugging.assert_less( hchkmax , 1.e-5, 'HopfBifurCell - > failed hermitian check.' )
        
    @tf.function
    def cpx_rot_to_rel( self , vec ):
        return vec * tf.math.conj( self.mke_unt_vec( vec ) )
        
    def call( self , input , state , training = False ):
        
#        tf.print( '\n'+'- '*20+' Cell '+'- '*20+'\nCount:' , self.cnt[0] , '\n' )
        
        A = self.A
        B = self.B
        
        Cu , Su = tf.unstack( self.U )
        Cw , Sw = tf.unstack( self.W )
        
        z = state[0] if tf.nest.is_nested( state ) else state
        x = input[0] if tf.nest.is_nested( input ) else input
#        _print( 'x' , x , summarize = -1 )
#        _print( 'z' , z , summarize = -1 )
        
        z0 , zn = z[0:1] , z[1::]
        x0 , xn = x[0:1] , x[1::]
        
        uz , ux = self.mke_unt_vec( zn ) , self.mke_unt_vec( xn )
        nz , nx = self.mke_nrm_vec( zn ) , self.mke_nrm_vec( xn )
        
        z0_j = tf.linalg.matmul( uz , self.hermitian_map( A , uz ) , adjoint_a = True )
        x0_j = tf.linalg.matmul( ux , self.hermitian_map( B , ux ) , adjoint_a = True )
        
        Cz = self.hermitian_map( Cw , nz )
        Cx = self.hermitian_map( Cu , nx )
        Sz = 1.j*self.hermitian_map( Sw , nz )
        Sx = 1.j*self.hermitian_map( Sw , nx )
        
#        tf.print( '\n'+'- '*20+' Cell '+'- '*20+'\nCount:' , self.cnt[0] , '\n' )
        
        zn_j = Cz + 1.j*Sz
        xn_j = Cx + 1.j*Sx
        
        z_k = tf.concat( [ z0_j , zn_j ] , 0 )
        x_k = tf.concat( [ x0_j , xn_j ] , 0 )
        
        d_k = z_k + x_k
        x_t = tf.keras.activations.tanh( d_k )
#        _print( 'x_t' , x_t )
        
        z_t = z + d_k
#        _print( 'z_t' , z_t )
        
        z_out = [ z_t ] if tf.nest.is_nested( state ) else z_t
        x_out = [ x_t ] if tf.nest.is_nested( input ) else x_t
        
        self.cnt.assign_add( [ 1 ] )
        
        return x_out , z_out
        
        
    @tf.custom_gradient
    def hermitian_map( self , H , x ):
        
        y = tf.linalg.matmul( H , x )
#        _print( 'A) H(y)' , y )
        
        if self.do_conjsym:
            y = ( y + tf.math.conj( y )[::-1] ) / 2.
#            _print( 'B) H(y)' , y )
        
        def grad( gL ):
            ''' SOURCE:
            The following directional derivative calculation is based on the paper by Najfeld and Havel.
            Cite: I. Najfeld and T. F. Havel, “Derivatives of the Matrix Exponential and Their Computation,”
            Advances in Applied Mathematics, vol. 16, no. 3, pp. 321–375, Sep. 1995, doi: 10.1006/aama.1995.1017.
            '''
            
#            _print( 'y' , y )
#            _print( 'gL' , gL )
            
            gL2 = tf.cast( gL , dtype = tf.complex128 )
            x2 = tf.cast( x , dtype = tf.complex128 )
            y2 = tf.cast( y , dtype = tf.complex128 )
            H2 = tf.cast( H , dtype = tf.complex128 )
            
            lmda2 , W2 = tf.linalg.eig( H2 )
            adjW2 = tf.linalg.adjoint( W2 )
            
            if self.do_conjsym:
                V_L = tf.linalg.matmul( x2 , gL2 , adjoint_b = True )
                V_R = tf.linalg.matmul( gL2 , x2 , adjoint_b = True )
                V = ( V_L + V_R ) / 2.
            else:
#                V = tf.linalg.matmul( gL2 , x2 , adjoint_b = True )
                V_L = tf.linalg.matmul( x2 , gL2 , adjoint_b = True )
                V_R = tf.linalg.matmul( gL2 , x2 , adjoint_b = True )
                V = ( V_L + V_R ) / 2.
                
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
    def skewherm_map( self , S , x ):
        
        y = tf.linalg.matmul( S , x )
#        _print( 'A) S(y)' , y )
        
        if self.do_conjsym: y = ( y + tf.math.conj( y )[::-1] ) / 2.
        
        def grad( gL ):
            
            U = tf.linalg.expm( S )
            
#            _print( 'gL' , gL )
#            _print( 'S' , S )
#            _print( 'U' , U )
            
            # Calculate Jacobian ( If that is what I calcualted ... I have no idea )
            dU = tf.linalg.matmul( S , U )
            J = tf.math.multiply_no_nan( gL , tf.linalg.adjoint( dU ) )
            J , nJ = tf.linalg.normalize( J , ord = 2 , axis = [ -2 , -1 ] )
#            _print( 'J' , J )
#            _print( 'nJ' , nJ )
            
            # Calculate the projUNN gradient for unitary matrix updates.
            W = self.calc_projUNN_T( U , J , lr = 0.1 )
            
            dW = tf.linalg.matmul( tf.linalg.logm( W ) , W )
            
            dS = tf.linalg.adjoint( dW ) - dU   # Same as sort of ->  dS = ( dW2 - dU )
            nx , _ = tf.linalg.normalize( x , ord = 2 )
            ngL , _ = tf.linalg.normalize( gL , ord = 2 )
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
        
        do_save = False
        
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
                state_init = np.random.rand( self.state_size )
                np.save( 'state_layer.npy' , state_init , allow_pickle = True )
            else:
                state_init = np.load( 'state_layer.npy' , allow_pickle = True )
        else:
            state_init = np.random.rand( self.state_size )
        
        state_init = tf.cast( state_init , dtype = tf.float32 )
        state_init = tf.signal.rfft( state_init ).reshape( -1 , 1 ) / self.sqrisz
        state_init = tf.reshape( tf.cast( state_init , dtype = self.dtype ) , [ state_init.shape[0] , 1 ] )
        self.state = tf.Variable(
            initial_value = state_init,
            shape = ( state_init.shape[0] , 1 ),
            dtype = self.dtype,
            trainable = False,
            name = 'state_'+self.name
        )
        
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
                self.cell.build( step_input_shape )         # Last build shape -> self.cells.build( input_shape[1::] )
                assert self.cell.built , 'HopfBifurCpxRNNLayer failed to build.'
        
        super( HopfBifurCpxRNNLayer , self ).build( input_shape )
        
        self.cnt = tf.Variable( [ 0 ] , trainable = False , dtype = tf.int32 )
        
        self.built = True
        
        
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
        
        
    @tf.function
    def map_input( self , x ):
        xT = tf.linalg.matrix_transpose( x )
        rfft = tf.signal.rfft( xT )
        return tf.linalg.matrix_transpose( rfft )
    
    @tf.function
    def map_output( self , x ):
        xT = tf.linalg.matrix_transpose( x )
        rfft = tf.signal.irfft( xT )
        return tf.linalg.matrix_transpose( rfft )
    
    def call( self , input , training = False ):
        
#        tf.print( '\n'*8+'=='*25+' Layer '+'=='*25+'\nCount:',self.cnt[0],'\n' )
#        _print( 'input' , input )
        
#        _input = tf.linalg.matrix_transpose( tf.signal.rfft( tf.linalg.matrix_transpose( input ) ) / self.sqrisz )
        _input = self.map_input( input )
        _state = self.state
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
        
        _output = tf.linalg.matrix_transpose( tf.signal.irfft( tf.linalg.matrix_transpose( _cout ) ) )
#        tf.print( '_output.shape' , _output.shape )
#        _print( '_output' , _output )
        
        max_img_val = tf.math.reduce_max( tf.math.imag( _output ) )
        tf.debugging.assert_less( max_img_val , 1.e-5, 'imaginary values went above 1.e-5 issue with irfft() input map.' )
#        tf.print('maximg:', max_img_val )
        
        output = tf.cast( tf.math.real( _output ) , input.dtype )
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
    
    
    
    
    
    
    
    
    