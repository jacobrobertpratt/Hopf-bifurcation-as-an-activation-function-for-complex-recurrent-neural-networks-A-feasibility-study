import os

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow.python.keras import backend
np_config.enable_numpy_behavior()

''' Local Imports '''
from proj_utils import _print
from activations import HopfActivRadius , HopfActivTheta
from initializers import Orthogonal


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
        
        
    def build( self , input_shape ):
        
#        print( 'units:' , self.units )
#        print( 'state_size:' , self.state_size )
#        print( 'output_size:' , self.output_size )
#        print( 'input_shape:' , input_shape )
#        exit(0)
        
        do_save = True
        do_train = True
        
        nme = 'M'
        self.M = self.add_weight(
            shape = ( input_shape[-1] , self.units ),
            initializer = Orthogonal( name = nme, save = do_save ),
#            initializer = tf.keras.initializers.Orthogonal(),
            trainable = do_train,
            dtype = self.dtype,
            name = nme + '_' + self.name
        )
#        _print( nme , self.M )
        
        nme = 'N'
        self.N = self.add_weight(
            shape = ( self.units , self.units ),
            initializer = Orthogonal( name = nme, save = do_save ),
#            initializer = tf.keras.initializers.Orthogonal(),
            trainable = do_train,
            dtype = self.dtype,
            name = nme + '_' + self.name
        )
#        _print( nme , self.N )
        
        self.step = tf.Variable( [ 0 ] , trainable = False , dtype = tf.int32 )
        
        super( HopfRNNCellTheta , self ).build( input_shape )
        
        self.built = True
    
    ''' SimpleRNN Call Method Example
    def call( self , inputs , states , training = False ):
        
#        tf.print( '\n'+'- '*20+' Cell '+'- '*20+'\nCount:' , self.step[0] , '\n' )
#        _print( 'inputs' , inputs )
        
        state = states[0] if tf.nest.is_nested( states ) else states
        input = inputs[0] if tf.nest.is_nested( inputs ) else inputs
#        _print( 'state' , state )
        
        M = self.M
        N = self.N
#        _print( 'M' , M )
#        _print( 'N' , N )
        
        Mx = backend.dot( M , inputs )
#        _print( 'Mx' , Mx )
        
        Nz = backend.dot( state , N )
#        _print( 'Nz' , Nz )
        
        y_t = Mx + Nz
#        _print( 'y_t' , y_t )
        
        y_t = self.activation( y_t )
#        _print( 'y_t' , y_t )
        
#        tf.debugging.check_numerics( y_t , 'y_t for theta output is nan value' )
        
        z_t = [ y_t ] if tf.nest.is_nested( states ) else y_t
#        _print( 'z_t' , z_t )
        
        self.step.assign( tf.math.floormod( self.step + 1 ,  self.units ) ) # unit size
        
        return y_t , z_t
    '''
    
    
    
    def call( self , inputs , states , training = False ):
        
#        tf.print( '\n'+'- '*20+' Cell '+'- '*20+'\nCount:' , self.step[0] , '\n' )
        
        state = states[0] if tf.nest.is_nested( states ) else states
        input = inputs[0] if tf.nest.is_nested( inputs ) else inputs
#        _print( 'state' , state )
#        _print( 'input' , input )
        
        
        M = self.M
        N = self.N
#        _print( 'M' , M )
#        _print( 'N' , N )
        
        Mx = backend.dot( input , M )
#        _print( 'Mx' , Mx )
        
        Nz = backend.dot( state , N )
#        _print( 'Nz' , Nz )
        
        y_t = Mx + Nz
#        _print( 'y_t' , y_t )
        
        if self.activation is not None:
            y_t = self.activation( y_t )
#        _print( 'y_t' , y_t )
        
        
        
#        tf.debugging.check_numerics( y_t , 'y_t for theta output is nan value' )
        
        z_t = [ y_t ] if tf.nest.is_nested( states ) else y_t
#        _print( 'z_t' , z_t )
        
        self.step.assign( tf.math.floormod( self.step + 1 ,  self.units ) ) # unit size
        
        return y_t , z_t
        
        
    '''
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
        
        self.layer.build( input_shape )
        
        super( HopfRNNLayerTheta , self ).build( input_shape )
        
        self.built = True
        
        
    def call( self , inputs , training = False ):
        input = tf.cast( inputs , dtype = self.dtype )
        output = self.layer.call( input , training = training )
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

