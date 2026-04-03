import os

import numpy as np

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow.python.keras import backend
np_config.enable_numpy_behavior()

''' Local Imports '''
from proj_utils import _print
from activations import HopfActivRadius , HopfActivTheta
from initializers import Orthogonal , Hermitian














@tf.keras.utils.register_keras_serializable('hopf_theta_cell')
class HopfRNNCellTheta( tf.keras.layers.Layer ):
    '''
        PrattRNNCell processes one timestep of a Hopf-Bifurcation recurrent network.
        That is the input is mapped with the internal weights, and updated 
        '''
        
    def __init__(
            self,
            state_size = 32,
            output_size = 32,
            activation = None,
            regularizer = None,
            **kwargs
        ):
        
        super( HopfRNNCellTheta , self ).__init__( **kwargs )
        
        self.units = state_size
        self.state_size = state_size
        self.output_size = output_size
        self.activation = activation
        
        if isinstance( activation , str ) and 'hopf' in activation:
            self.hopfact = HopfActivTheta()
        else:
            self.hopfact = None
            
        
    def build( self , input_shape ):
        
        '''
        print( 'C) units:' , self.units )
        print( 'C) state_size:' , self.state_size )
        print( 'C) output_size:' , self.output_size )
        print( 'C) input_shape:' , input_shape )
#       '''
#        exit(0)
        
        do_save = False
        do_train = True
        
        fxsz = tf.signal.rfft( tf.ones( shape = input_shape , dtype = self.dtype ) ).shape
        fzsz = tf.signal.rfft( tf.ones( shape = ( input_shape[0] , self.units ) , dtype = self.dtype ) ).shape
        
#        '''
#        with tf.device('/cpu:0'):
        nme = 'H'
        self.H = self.add_weight(
            shape = ( fzsz[-1] - 1 , fzsz[-1] - 1 ),
            initializer = Hermitian( name = nme, save = do_save ),
            trainable = do_train,
            dtype = tf.complex64,
            name = nme + '_' + self.name
        )
#        _print( nme , self.H )
        
        nme = 'G'
        self.G = self.add_weight(
            shape = ( fzsz[-1] - 1 , fzsz[-1] - 1 ),
            initializer = Hermitian( name = nme, save = do_save ),
            trainable = do_train,
            dtype = tf.complex64,
            name = nme + '_' + self.name
        )
#        _print( nme , self.G )
        
        nme = 'M'
        self.M = self.add_weight(
            shape = ( input_shape[-1] , self.units ),
#            initializer = Orthogonal( name = nme, save = do_save ),
            initializer = tf.keras.initializers.GlorotNormal(),
            trainable = do_train,
            dtype = self.dtype,
            name = nme + '_' + self.name
        )
#        _print( nme , self.M )
        
        nme = 'N'
        self.N = self.add_weight(
            shape = ( self.units , self.units ),
#            initializer = Orthogonal( name = nme, save = do_save ),
            initializer = tf.keras.initializers.GlorotNormal(),
            trainable = do_train,
            dtype = self.dtype,
            name = nme + '_' + self.name
        )
#        _print( nme , self.N )
        
        self.ones = tf.constant(
#            tf.ones( shape = ( input_shape[0] , fzsz[-1] - 1 ), dtype = self.dtype ),
#            tf.ones( shape = ( input_shape[0] , fzsz[-1] ), dtype = self.dtype ),
            tf.zeros( shape = ( input_shape[0] , self.units ), dtype = self.dtype ),
            dtype = self.dtype
        )
#        _print( 'fones' , self.fones )
        
        self.zeros = tf.constant(
#            tf.zeros( shape = ( input_shape[0] , fzsz[-1] - 1 ), dtype = self.dtype ),
#            tf.zeros( shape = ( input_shape[0] , fzsz[-1] ), dtype = self.dtype ),
            tf.zeros( shape = ( input_shape[0] , self.units ), dtype = self.dtype ),
            dtype = self.dtype
        )
#        _print( 'fzeros' , self.fzeros )
#        exit(0)
        
        self.step = tf.Variable( [ 0 ] , trainable = False , dtype = tf.int32 )
        
        super( HopfRNNCellTheta , self ).build( input_shape )
        
        self.built = True
        
        
    @tf.custom_gradient
    def herm_map( self , x , H ):
        
        HT = tf.linalg.matrix_transpose( H )
        y = backend.dot( x , H ) #  + backend.dot( x , HT )
        
#        _print( 'y' , y )
        
        def grad( gL ):
            ''' SOURCE:
            The following directional derivative calculation is based on the paper by Najfeld and Havel.
            Cite: I. Najfeld and T. F. Havel, “Derivatives of the Matrix Exponential and Their Computation,”
            Advances in Applied Mathematics, vol. 16, no. 3, pp. 321–375, Sep. 1995, doi: 10.1006/aama.1995.1017.
            '''
            
            lmda , W = tf.linalg.eig( H )
            adjW= tf.linalg.adjoint( W )
            adjx = tf.linalg.adjoint( x )
            
#            _print( 'adjx' , adjx )
#            _print( 'gL' , gL )
            
            J = backend.dot( adjx , gL )
#            _print( 'J' , J )
            
            func = tf.math.exp( lmda )
            _func = tf.expand_dims( func , 0 )
            _eigs = tf.expand_dims( lmda , 0 )
            G = tf.math.divide_no_nan(
                _func - tf.linalg.matrix_transpose( _func ),
                _eigs - tf.linalg.matrix_transpose( _eigs )
            ) + tf.linalg.diag( func )
            
            Vbar = tf.linalg.matmul( tf.linalg.matmul( adjW , J ) , W )
            GV = tf.math.multiply( G , Vbar )
            
            Dv_H = tf.linalg.matmul( tf.linalg.matmul( W , GV ) , adjW )
            
            gH = ( Dv_H + tf.linalg.adjoint( Dv_H ) ) / 2.
#            _print( 'gH' , gH )
            
            gx = backend.dot( gL , HT ) + backend.dot( gL , H )
            
            return gx , gH
            
        return y , grad
        
        
    def call( self , inputs , states , training = False ):
        
        tf.print( '\n'+'- '*20+' Cell '+'- '*20+'\nCount:' , self.step[0] , '\n' )
        
        input = inputs[0] if tf.nest.is_nested( inputs ) else inputs
        state = states[0] if tf.nest.is_nested( states ) else states
        _print( 'input' , input )
        _print( 'state' , state )
        
        xM = backend.dot( input , self.M )
        zN = backend.dot( state , self.N )
        
        r = ( xM + zN ) / 2.
        u = tf.math.sqrt( xM**2 + zN**2 )
        y_t , _ = self.hopfact( self.step , r , self.zeros , u , self.zeros )
        
        '''
        ## Complex Mapping ##
        if self.hopfact is not None:
            
            cx = tf.signal.rfft( xM )
            cz = tf.signal.rfft( zN )
#            _print( 'cx' , cx )
#            _print( 'cz' , cz )
            
            cx0 , cx_ = cx[::,0:1] , cx[::,1::]
            cz0 , cz_ = cz[::,0:1] , cz[::,1::]
            
            H = self.H
#            HT = tf.linalg.matrix_transpose( H )
            G = self.G
#            GT = tf.linalg.matrix_transpose( G )
            
#            Hchk = H - tf.linalg.adjoint( H )
#            Hchk = tf.math.reduce_max( tf.math.abs( Hchk ) )
#            _print( 'Hchk' , Hchk )
            
#            cx_H = backend.dot( cx_ , H )
#            cz_G = backend.dot( cz_ , G )
#            cx_H = backend.dot( cx_ , H ) + backend.dot( cx_ , HT )
#            cz_G = backend.dot( cz_ , G ) + backend.dot( cz_ , GT )
            cx_H = self.herm_map( cx_ , H )
            cz_G = self.herm_map( cz_ , G )
            cn_t = cx_H + cz_G
#            _print( 'cn_t' , cn_t )
            
            r = tf.math.abs( cx_H ) + tf.math.abs( cz_G )
            u = tf.math.abs( cn_t )**2
#            r = tf.math.abs( cn_t )
#            u = tf.math.abs( cx_H )
#            _print( 'r' , r )
#            _print( 'u' , u )
            
            w = self.zeros # tf.math.angle( cz_G )
            v = self.zeros
            
#            _print( 'w' , w )
#            _print( 'v' , v )
            
            rn_t , wn_t = self.hopfact( self.step , r , w , u , v )
            cn_t = rn_t * tf.math.exp( 1.j * wn_t )
#            _print( 'cn_t' , cn_t )
            
            c0_t = ( cx0 + cz0 ) / 2.
            cy_t = tf.concat( [ c0_t , cn_t ] , 1 )
            y_t = tf.signal.irfft( cy_t )
            
        else:
            
            r = xM + zN
            u = tf.math.sqrt( xM**2 + zN**2 )
            
            y_t = self.hopfact( self.step , r , self.zeros , u , self.zeros )
            if self.activation is not None: y_t = self.activation( y_t )
#        _print( 'y_t' , y_t )
#        '''
        
#        tf.debugging.check_numerics( y_t , 'y_t for theta output is nan value' )
        
        z_t = [ y_t ] if tf.nest.is_nested( states ) else y_t
#        _print( 'y_t' , y_t )
#        _print( 'z_t' , z_t )
        
        self.step.assign( tf.math.floormod( self.step + 1 ,  self.units ) )
        
        return y_t , z_t
        
    def get_initial_state( self , inputs , batch_size , dtype ):
        ini_state = tf.zeros( shape = ( batch_size , self.units ) , dtype = dtype )
        return ini_state
#   '''
    
    def get_config( self ):
        config = {
            "units" : self.units,
            "activation" : tf.keras.activations.serialize( self.activation )
        }
        base_config = super().get_config()
        return dict( list( base_config.items() ) + list( config.items() ) )


@tf.keras.utils.register_keras_serializable( 'hopf_theta_layer' )
class HopfRNNLayerTheta( tf.keras.layers.Layer ):
    
    def __init__(
                  self,
                  state_size,
                  output_size = None,
                  activation = None,
                  return_sequences = False,
                  time_major = False,
                  stateful = False,
                  **kwargs
                ):
        
        super( HopfRNNLayerTheta , self ).__init__( **kwargs )
        
        self.state_size = state_size
        self.output_size = output_size
        self.return_sequences = return_sequences
        self.stateful = stateful
        self.time_major = time_major
        self.activation = activation
        
        
        
    def build( self , input_shape ):
        
        '''
        print( 'L) input_shape:' , input_shape )
        print( 'L) state_size:', self.state_size )
        print( 'L) output_size:', self.output_size )
        print( 'L) return_sequences:', self.return_sequences )
        print( 'L) stateful:', self.stateful )
        print( 'L) time_major:', self.time_major )
        print( 'L) activation:', self.activation )
#       '''
        
        self.layer = tf.keras.layers.RNN(
            HopfRNNCellTheta(
                activation = self.activation,
                state_size = self.state_size,
                output_size = self.output_size,
                dtype = self.dtype
            ),
            return_sequences = self.return_sequences,
            stateful = self.stateful,
            time_major = self.time_major,
            dtype = self.dtype
        )
#        self.layer.build( input_shape )
        
        super( HopfRNNLayerTheta , self ).build( input_shape )
        
        self.built = True
        
        
    def call( self , inputs , training = False ):
        
        input = tf.cast( inputs , dtype = self.dtype )
        _print( 'input' , input )
        
        input = input.transpose( ( 0 , 2, 1 ) )
        input = tf.signal.rfft( input )
        
        output = self.layer.call( input , training = training )
        _print( 'output', output )
        
        return tf.cast( output , dtype = inputs.dtype )
        
        
    def get_config(self):
        config = super( HopfRNNLayerTheta , self ).get_config()
        config.update( { 'state_size' : self.state_size } )
        return config
        
        
    @classmethod
    def from_config( self , config ):
        return self( **config )






@tf.keras.utils.register_keras_serializable('hopf_radial_cell')
class HopfRNNCellRadial( tf.keras.layers.Layer ):
    '''
        PrattRNNCell processes one timestep of a Hopf-Bifurcation recurrent network.
        That is the input is mapped with the internal weights, and updated 
        '''
        
    def __init__(
            self,
            state_size = 32,
            output_size = 32,
            activation = None,
            regularizer = None,
            **kwargs
        ):
        
        super( HopfRNNCellRadial , self ).__init__( **kwargs )
        
        self.units = state_size
        self.state_size = state_size
        self.output_size = output_size
        
        self.activation = HopfActivRadius()
        
        
    def build( self , input_shape ):
        
#        print( '(C) input_shape:' , input_shape )
#        exit(0)
        
        do_train = True
        
        self.A = self.add_weight(
            shape = ( input_shape[-1] , self.units ),
            initializer = tf.keras.initializers.GlorotNormal(),
            trainable = do_train,
            dtype = self.dtype,
            name = 'A_' + self.name
        )
#        _print( 'A' , self.A )
        
        self.B = self.add_weight(
            shape = ( self.units, self.units ),
            initializer = tf.keras.initializers.GlorotNormal(),
            trainable = do_train,
            dtype = self.dtype,
            name = 'B_' + self.name
        )
#        _print( 'B' , self.B )
#        exit(0)
        
        self.step = tf.Variable( [ 0 ] , trainable = False , dtype = tf.int32 )
        
        super( HopfRNNCellRadial , self ).build( input_shape )
        
        self.built = True
        
        
    def call( self , inputs , states , training = False ):
        
#        tf.print( '\n'+'- '*20+' Cell '+'- '*20+'\nCount:' , self.step[0] , '\n' )
        
        state = states[0] if tf.nest.is_nested( states ) else states
        input = inputs[0] if tf.nest.is_nested( inputs ) else inputs
#        _print( 'input' , input )
#        _print( 'state' , state )
        
        A = self.A
        B = self.B
#        _print( 'A' , A )
#        _print( 'B' , B )
        
        x_i = backend.dot( input , self.A )
#        _print( 'x_i' , x_i )
        
        z_i = backend.dot( state , self.B )
#        _print( 'z_i' , z_i )
        
        z_j = ( x_i + z_i ) / 2.
#        _print( 'z_j' , z_j )
        
        y_t = self.activation( z_j , 2 * tf.math.abs( z_j ) )
#        _print( 'y_t' , y_t )
        
        z_t = [ z_j ] if tf.nest.is_nested( states ) else z_j
#        _print( 'z_t' , z_t )
        
        self.step.assign( self.step + 1 ) # tf.math.floormod( self.step + 1 ,  self.state_size ) )
        
        return y_t , z_t
        
        
    def get_config( self ):
        config = {
            "units" : self.units,
            "activation" : tf.keras.activations.serialize( self.activation )
        }
        base_config = super().get_config()
        return dict( list( base_config.items() ) + list( config.items() ) )

@tf.keras.utils.register_keras_serializable( 'hopf_radial_layer' )
class HopfRNNLayerRadial( tf.keras.layers.Layer ):
    
    def __init__(
                  self,
                  state_size,
                  output_size = None,
                  activation = None,
                  return_sequences = False,
                  time_major = False,
                  stateful = False,
                  **kwargs
                ):
        
        super( HopfRNNLayerRadial , self ).__init__( **kwargs )
        
        self.state_size = state_size
        self.output_size = output_size
        self.return_sequences = return_sequences
        self.stateful = stateful
        self.time_major = time_major
        self.activation = activation
        
        
        
    def build( self , input_shape ):
        
        '''
        print( '(L) input_shape:' , input_shape )
        print( '(L) state_size:', self.state_size )
        print( '(L) output_size:', self.output_size )
        print( '(L) return_sequences:', self.return_sequences )
        print( '(L) stateful:', self.stateful )
        print( '(L) time_major:', self.time_major )
        print( '(L) activation:', self.activation )
        exit(0)
        '''
        
        self.layer = tf.keras.layers.RNN(
            HopfRNNCellRadial(
                activation = self.activation,
                state_size = self.state_size,
                output_size = self.output_size,
                dtype = self.dtype
            ),
            return_sequences = self.return_sequences,
            stateful = self.stateful,
            time_major = self.time_major,
            dtype = self.dtype
        )
        
        self.layer.build( input_shape )
        
        super( HopfRNNLayerRadial , self ).build( input_shape )
        
        self.built = True
        
        
    def call( self , inputs , training = False ):
        
        input = tf.cast( inputs , dtype = self.dtype )
#        _print( 'input' , input )
        
        output = self.layer.call( input , training = training )
        
#        tf.print('\n' + '- '*10 + ' __ RETURNED __ ' + 10*' -' + '\n')
#        _print( 'inputs' , inputs )
#        _print( 'outputs' , output )
        
        return tf.cast( output , dtype = inputs.dtype )
        
        
    def get_config(self):
        config = super( HopfRNNLayerRadial , self ).get_config()
        config.update( { 'state_size' : self.state_size } )
        return config
        
        
    @classmethod
    def from_config( self , config ):
        return self( **config )





@tf.keras.utils.register_keras_serializable('hopf_base_cell')
class HopfRNNCellBase( tf.keras.layers.Layer ):
    '''
        PrattRNNCell processes one timestep of a Hopf-Bifurcation recurrent network.
        That is the input is mapped with the internal weights, and updated 
        '''
        
    def __init__(
            self,
            state_size = 32,
            output_size = 32,
            activation = None,
            regularizer = None,
            **kwargs
        ):
        
        super( HopfRNNCellBase , self ).__init__( **kwargs )
        
        self.units = state_size
        self.state_size = state_size
        self.output_size = output_size
        
        self.activation = tf.keras.activations.linear if activation is None else activation
        
        
    def build( self , input_shape ):
        
#        print( '(C) input_shape:' , input_shape )
#        exit(0)
        
        do_train = True
        
        self.A = self.add_weight(
            shape = ( input_shape[-1] , self.units ),
            initializer = tf.keras.initializers.GlorotNormal(),
            trainable = do_train,
            dtype = self.dtype,
            name = 'A_' + self.name
        )
#        _print( 'A' , self.A )
        
        self.B = self.add_weight(
            shape = ( self.units, self.units ),
            initializer = tf.keras.initializers.GlorotNormal(),
            trainable = do_train,
            dtype = self.dtype,
            name = 'B_' + self.name
        )
#        _print( 'B' , self.B )
#        exit(0)
        
        self.step = tf.Variable( [ 0 ] , trainable = False , dtype = tf.int32 )
        
        super( HopfRNNCellBase , self ).build( input_shape )
        
        self.built = True
        
        
    def call( self , inputs , states , training = False ):
        
#        tf.print( '\n'+'- '*20+' Cell '+'- '*20+'\nCount:' , self.step[0] , '\n' )
        
        state = states[0] if tf.nest.is_nested( states ) else states
        input = inputs[0] if tf.nest.is_nested( inputs ) else inputs
#        _print( 'input' , input )
#        _print( 'state' , state )
        
        A = self.A
        B = self.B
#        _print( 'A' , A )
#        _print( 'B' , B )
        
        x_i = backend.dot( input , self.A )
#        _print( 'x_i' , x_i )
        
        z_i = backend.dot( state , self.B )
#        _print( 'z_i' , z_i )
        
        z_j = ( x_i + z_i ) / 2.
#        _print( 'z_j' , z_j )
        
        y_t = self.activation( z_j )
#        y_t = self.activation( z_j , 2*tf.math.abs( z_j ) )
#        _print( 'y_t' , y_t )
        
        z_t = [ z_j ] if tf.nest.is_nested( states ) else z_j
#        _print( 'z_t' , z_t )
        
        self.step.assign( self.step + 1 ) # tf.math.floormod( self.step + 1 ,  self.state_size ) )
        
        return y_t , z_t
        
        
    def get_config( self ):
        config = {
            "units" : self.units,
            "activation" : tf.keras.activations.serialize( self.activation )
        }
        base_config = super().get_config()
        return dict( list( base_config.items() ) + list( config.items() ) )

@tf.keras.utils.register_keras_serializable( 'hopf_base_layer' )
class HopfRNNLayerBase( tf.keras.layers.Layer ):
    
    def __init__(
                  self,
                  state_size,
                  output_size = None,
                  activation = None,
                  return_sequences = False,
                  time_major = False,
                  stateful = False,
                  **kwargs
                ):
        
        super( HopfRNNLayerBase , self ).__init__( **kwargs )
        
        self.state_size = state_size
        self.output_size = output_size
        self.return_sequences = return_sequences
        self.stateful = stateful
        self.time_major = time_major
        self.activation = activation
        
        
        
    def build( self , input_shape ):
        
        '''
        print( '(L) input_shape:' , input_shape )
        print( '(L) state_size:', self.state_size )
        print( '(L) output_size:', self.output_size )
        print( '(L) return_sequences:', self.return_sequences )
        print( '(L) stateful:', self.stateful )
        print( '(L) time_major:', self.time_major )
        print( '(L) activation:', self.activation )
        exit(0)
        '''
        
        self.layer = tf.keras.layers.RNN(
            HopfRNNCellBase(
                activation = self.activation,
                state_size = self.state_size,
                output_size = self.output_size,
                dtype = self.dtype
            ),
            return_sequences = self.return_sequences,
            stateful = self.stateful,
            time_major = self.time_major,
            dtype = self.dtype
        )
        
        self.layer.build( input_shape )
        
        super( HopfRNNLayerBase , self ).build( input_shape )
        
        self.built = True
        
        
    def call( self , inputs , training = False ):
        
        input = tf.cast( inputs , dtype = self.dtype )
#        _print( 'input' , input )
        
        output = self.layer.call( input , training = training )
        
#        tf.print('\n' + '- '*10 + ' __ RETURNED __ ' + 10*' -' + '\n')
#        _print( 'inputs' , inputs )
#        _print( 'outputs' , output )
        
        return tf.cast( output , dtype = inputs.dtype )
        
        
    def get_config(self):
        config = super( HopfRNNLayerBase , self ).get_config()
        config.update( { 'state_size' : self.state_size } )
        return config
        
        
    @classmethod
    def from_config( self , config ):
        return self( **config )

