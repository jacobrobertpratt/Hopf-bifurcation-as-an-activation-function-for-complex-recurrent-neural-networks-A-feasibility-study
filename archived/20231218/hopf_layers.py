import os

import numpy as np

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow.python.keras import backend
np_config.enable_numpy_behavior()

''' Local Imports '''
from proj_utils import _print
from activations import HopfActivRadius , HopfActivTheta , HopfActivCpx
from activations import CpxReLU
from initializers import Eye , Orthogonal , GlorotNorm , GlorotUnif , Hermitian , Unitary



@tf.custom_gradient
def herm_map( H , x ):
    
    HT = tf.linalg.matrix_transpose( H )
#    y = backend.dot( x , H )
    y = tf.linalg.matmul( H , x )
#    y = ( tf.linalg.matmul( H , x ) + tf.linalg.matmul( H , x , transpose_a = True ) ) / 2.
#        y = ( backend.dot( x , H ) + backend.dot( x , HT ) ) / 2.
#    y = ( backend.dot( H , x ) + backend.dot( HT , x ) ) / 2.
#        _print( 'y' , y )
    
    def grad( gL ):
        ''' SOURCE:
        The following directional derivative calculation is based on the paper by Najfeld and Havel.
        Cite: I. Najfeld and T. F. Havel, “Derivatives of the Matrix Exponential and Their Computation,”
        Advances in Applied Mathematics, vol. 16, no. 3, pp. 321–375, Sep. 1995, doi: 10.1006/aama.1995.1017.
        '''
        
        lmda , W = tf.linalg.eigh( H )
        adjW= tf.linalg.adjoint( W )
        adjx = tf.linalg.adjoint( x )
#            _print( 'adjx' , adjx )
#            _print( 'gL' , gL )
        
        J = backend.dot( gL , adjx )
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
#        _print( 'gH' , gH )
        
        gx = backend.dot( HT , gL ) + backend.dot( H , gL )
        
        return gH , gx
        
    return y , grad
    
    
@tf.custom_gradient
def unit_map( self , U , x ):
    
    y = tf.linalg.matmul( U , x )
    
    def grad( gl ):
        
        _print( 'gL' , gL )
        
        '''
        lr = 1.e-3 / 2.
        G = tf.linalg.matmul( gl , tf.linalg.adjoint( x ) )
        A = tf.linalg.matmul( G , U , adjoint_a = True  ) - tf.linalg.matmul( U , G , adjoint_a = True )
        I = tf.eye( A.shape[0] , dtype = A.dtype )
        L = tf.linalg.inv( I + lr*A )
        R = I - lr*A
        gU = tf.linalg.matmul( tf.linalg.matmul( L , R ) , U )
        nx , _ = tf.linalg.normalize( x , ord = 2 , axis = [-2,-1] )
        dU = tf.linalg.matmul( tf.linalg.logm( U ) , U )
        '''
        
        gU = tf.zeros_like( U )
        gx = tf.zeros_like( x )
        
        return gU , gx
    
    return y , grad
    
        
        
@tf.keras.utils.register_keras_serializable( 'hopf_theta_cell' )
class HopfRNNCellTheta( tf.keras.layers.Layer ):
    '''
    PrattRNNCell processes one timestep of a Hopf-Bifurcation recurrent network.
    That is the input is mapped with the internal weights, and updated 
    '''
    
    def __init__(
            self,
            units,
            activation = None,
            **kwargs
        ):
        
        super( HopfRNNCellTheta , self ).__init__( **kwargs )
        
        self.units = units
        self.state_size = units
        self.output_size = units
        self.activation = activation
        self.hopfact = None
        
        if isinstance( activation , str ):
            if 'hopf' in activation: self.hopfact = HopfActivCpx()   # self.hopfact = HopfActivTheta()
            if 'relu' in activation: self.activation = CpxReLU()
            if 'tanh' in activation: self.activation = tf.keras.activations.tanh
            
            
    def build( self , input_shape ):
        
        '''
        print( self.name + ' units:' , self.units )
        print( self.name + ' state_size:' , self.state_size )
        print( self.name + ' output_size:' , self.output_size )
        print( self.name + ' input_shape:' , input_shape )
        exit(0)
#       '''
        
        do_save = True
        do_train = True
        
        nme = 'A'
        self.A = self.add_weight(
#            shape = ( self.units , self.units ),
            shape = ( input_shape[0] , input_shape[0] ),
#            initializer = Hermitian( name = nme, save = do_save ),
#            initializer = Unitary( name = nme, save = do_save ),
            initializer = Orthogonal( name = nme, save = do_save ),
#            initializer = GlorotNorm( name = nme, save = do_save ),
#            initializer = GlorotUnif( name = nme, save = do_save ),
#            initializer = Eye( name = nme , save = do_save ),
            trainable = do_train,
            dtype = tf.complex64,
            name = nme + '_' + self.name
        )
#        _print( self.name + ' A' , self.A )
        
        nme = 'B'
        self.B = self.add_weight(
#            shape = ( self.units , self.units ),
#            shape = ( input_shape[-1] , self.units ),
            shape = ( input_shape[0] , input_shape[0] ),
#            initializer = Hermitian( name = nme , save = do_save ),
#            initializer = Unitary( name = nme , save = do_save ),
#            initializer = Orthogonal( name = nme , save = do_save ),
            initializer = GlorotUnif( name = nme, save = do_save ),
#            initializer = GlorotNorm( name = nme, save = do_save ),
#            initializer = Eye( name = nme , save = do_save ),
            trainable = do_train,
            dtype = tf.complex64,
            name = nme + '_' + self.name
        )
#        _print( self.name + ' B' , self.B )
        
        self.step = tf.Variable( [ 0 ] , trainable = False , dtype = tf.int32 )
        
        super( HopfRNNCellTheta , self ).build( input_shape )
        
        self.built = True
        
    def call( self , inputs , states , training = False ):
        
#        tf.print( '\n'+'- '*20+' Cell '+'- '*20+'\nCount:' , self.step[0] , '\n' )
        
        x = inputs[0] if tf.nest.is_nested( inputs ) else inputs
        z = states[0] if tf.nest.is_nested( states ) else states
#        _print( self.name + ' x' , x )
#        _print( self.name + ' z' , z )
        
        z_0 , z_n = z[::,0:1] , z[::,1::]
#        _print( 'z_0' , z_0 )
#        _print( 'z_n' , z_n )
        
        A = self.A
        B = self.B
#        _print( self.name + ' A' , A )
#        _print( self.name + ' B' , B )
        
#        Az = backend.dot( A , z )
#        Bx = backend.dot( B , x )
#        Az = backend.dot( z , self.A )
#        Bx = backend.dot( x , self.B )
        Az = tf.linalg.matmul( A , z_n )
        Bx = tf.linalg.matmul( B , x )
#        _print( self.name + ' Az' , Az )
#        _print( self.name + ' Bx' , Bx )
        
        y_t = Az + Bx
#        _print( 'y_t' , y_t )
        
        if self.hopfact is not None:
            y_t = self.hopfact( z_n , y_t , tf.ones_like( y_t ) , self.step , self.units )
        elif self.activation is not None:
            y_t = self.activation( y_t )
        
        y_t = tf.concat( [ z_0 , y_t ] , -1 )
#        _print( 'y_t' , y_t )
        
#        tf.debugging.check_numerics( tf.math.abs( y_t ) , 'var: abs(y_t)  ->  phase RNN layer has nan value' )
        
        z_t = [ y_t ] if tf.nest.is_nested( states ) else y_t
#        _print( 'y_t' , y_t )
#        _print( 'z_t' , z_t )
        
        self.step.assign( tf.math.floormod( self.step + 1 ,  ( self.units + 1 ) ) )
#        self.step.assign_add( [ 1 ] )
        
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
                  units,
                  activation = None,
                  return_sequences = False,
                  time_major = False,
                  stateful = False,
                  **kwargs
                ):
        
        super( HopfRNNLayerTheta , self ).__init__( **kwargs )
        
        # To hold the cell(s) for the model.
        self.cell = None
        
        self.units = units
        self.return_sequences = return_sequences
        self.stateful = stateful
        self.time_major = time_major
        self.activation = activation
        
        
    def build( self , input_shape ):
        
        '''
        print( self.name + ' input_shape:' , input_shape )
        print( self.name + ' units:', self.units )
        print( self.name + ' return_sequences:', self.return_sequences )
        print( self.name + ' stateful:', self.stateful )
        print( self.name + ' time_major:', self.time_major )
        print( self.name + ' activation:', self.activation )
        exit(0)
#       '''
        
        inshp = list( input_shape )
        self.bsz = inshp[0]
        self.seqlen = inshp[1]
        
        # Generate cell -> Only use 1 for now, until it's working better.
        if self.cell is None:
            self.cell = HopfRNNCellTheta(
                units = self.units,
                activation = self.activation,
                dtype = self.dtype,
                name = self.name + '_hopf_cell'
            )
            
        # From tf.keras.layers.basernn
        def get_step_input_shape( shape ):
            if isinstance( shape , tf.TensorShape ):
                shape = tuple( shape.as_list() )
            return ( shape[0] , ) + shape[2:]
        step_input_shape = tf.nest.map_structure( get_step_input_shape , input_shape )
        
        # Build cell(s)
        if not self.cell.built:
            with tf.name_scope( self.cell.name ):
                self.cell.build( step_input_shape )         # Last build shape -> self.cells.build( input_shape[1::] )
                assert self.cell.built , 'HopfBifurCpxRNNLayer failed to build.'
                
        if isinstance( self.activation , str ) and 'hopf' in self.activation:
            state_init = tf.ones( shape = ( step_input_shape[0] , self.units ) , dtype = tf.float32 )
        else:
            state_init = tf.zeros( shape = ( step_input_shape[0] , self.units ) , dtype = tf.float32 )
        state_init = tf.cast( state_init , dtype = self.dtype )
        self.state = tf.Variable(
            initial_value = state_init,
            shape = state_init.shape,
            dtype = self.dtype,
            trainable = False,
            name = 'state_'+self.name
        )
#        _print( self.name + ' state' , self.state )
#        exit(0)
        
        super( HopfRNNLayerTheta , self ).build( input_shape )
        
        self.step = tf.Variable( [ 0 ] , trainable = False , dtype = tf.int32 )
        
        self.built = True
        
        
    def call( self , inputs , training = False ):
        
#        tf.print( '\n'*8+'=='*25+' Layer '+'=='*25+'\nCount:',self.step[0],'\n' )
#        _print( 'inputs' , inputs )
        
#        _input = tf.reshape( inputs , [ self.bsz , self.vsz , 1 ] )
        _input = inputs.transpose(( 0 , 2 , 1 ))
        _input = tf.signal.rfft( _input , fft_length = [ self.seqlen ] ) / self.seqlen
        _input = _input[::,::,0:self.units]
        _input = _input.transpose(( 0 , 2 , 1 ))
        
        rnn_return = self.rnn_call(
            self.cell,
            _input,
            self.state,
            time_major = False,
            return_all_outputs = True,    # If True -> seq_out has batch sized output; else, single output
            training = training
        )
        _clast , _cout , _cstate = rnn_return
        
        self.state.assign( _cstate )
        
#        tf.print( '\n'+' *'*15 + ' RNN Returned ' + '* '*15 + '\n' )
#        _print( '_clast' , _clast )
#        _print( '_cout' , _cout )
#        _print( '_cstate' , _cstate )
        
        _output = _cout.transpose(( 0 , 2 , 1 ))
        _output = tf.signal.irfft( _output * self.seqlen , fft_length = [ self.seqlen ] )
        output = _output.transpose(( 0 , 2 , 1 ))
#        _print( 'output' , output )
        
        if not self.return_sequences: output = output[::,-1,::]
        
#        max_img_val = tf.math.reduce_max( tf.math.imag( _output ) )
#        tf.debugging.assert_less( max_img_val , 1.e-5, 'imaginary values went above 1.e-5 issue with irfft() input map.' )
#        tf.print('maximg:', max_img_val )
        
#        output = tf.cast( _output , inputs.dtype )
#        _print( 'output' , output )
        
        self.step.assign_add( [ 1 ] )
        
        return output
        
        
    def get_config(self):
        config = super( HopfRNNLayerTheta , self ).get_config()
        config.update( { 'state_size' : self.state_size } )
        return config
        
        
    @classmethod
    def from_config( self , config ):
        return self( **config )



    @tf.function
    def rnn_call( self , cell , inputs , states , time_major=False , return_all_outputs=False , training=False):
        
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
        
        return ( _lastout , _outputs , _states )


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

