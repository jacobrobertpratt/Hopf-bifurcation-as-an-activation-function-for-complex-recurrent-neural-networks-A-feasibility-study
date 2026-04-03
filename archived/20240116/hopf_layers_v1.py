import os

import numpy as np

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow.python.keras import backend
np_config.enable_numpy_behavior()

''' Local Imports '''
from proj_utils import _print
from activations import CpxReLU , HopfActivCpx
from initializers import Eye , Orthogonal , GlorotNorm , GlorotUnif , Hermitian , Unitary

@tf.custom_gradient
def _pgrad( inpt ):
    out = inpt
    def grad( gL ):
        _print( 'gL' , gL )
        return gL
    return out , grad


@tf.custom_gradient
def herm_map_V2( H , x ):
    
#    HT = tf.linalg.matrix_transpose( H )
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
        
#        tf.print( '- '*25 )
        
        lmda , W = tf.linalg.eigh( H )
        adjW= tf.linalg.adjoint( W )
        adjx = tf.linalg.adjoint( x )
#        _print( 'adjx' , adjx )
#        _print( 'gL' , gL )
        
        J = backend.dot( gL , adjx )
#        _print( 'J' , J )
        
        func = tf.math.exp( lmda )
        _func = tf.expand_dims( func , 0 )
        _eigs = tf.expand_dims( lmda , 0 )
#        _print( '_func' , _func )
#        _print( '_eigs' , _eigs )
        
        D = tf.linalg.diag( func )
#        _print( 'D' , D )
        
        G = tf.math.divide_no_nan(
            _func - tf.linalg.matrix_transpose( _func ),
            _eigs - tf.linalg.matrix_transpose( _eigs )
        ) + D
        
        Vbar = tf.linalg.matmul( tf.linalg.matmul( adjW , J ) , W )
        GV = tf.math.multiply( G , Vbar )
        
        Dv_H = tf.linalg.matmul( tf.linalg.matmul( W , GV ) , adjW )
#        _print( 'Dv_H' , Dv_H )
        
#        tst = tf.linalg.diag( tf.linalg.diag_part( Dv_H ) )
#        _print( 'tst' , tst )
        
        # !!!! BEST w/ Base1 Mapping !!!!
        gH = Dv_H - tf.linalg.diag( tf.linalg.diag_part( Dv_H ) ) / 2.
        gx = backend.dot( H , gL ) + backend.dot( tf.linalg.matrix_transpose( H ) , gL )
        
#        evls , evcs = tf.linalg.eigh( gH )
#        _print( 'gH) evls' , evls )
#        _print( 'gH) evcs' , evcs )
        
        
        return gH , gx
        
    return y , grad
    
    
@tf.custom_gradient
def herm_map_V1( H , x ):
    
    y = tf.linalg.matmul( H , x )
    
    def grad( gL ):
        ''' SOURCE:
        The following directional derivative calculation is based on the paper by Najfeld and Havel.
        Cite: I. Najfeld and T. F. Havel, “Derivatives of the Matrix Exponential and Their Computation,”
        Advances in Applied Mathematics, vol. 16, no. 3, pp. 321–375, Sep. 1995, doi: 10.1006/aama.1995.1017.
        '''
        
        lmda , W = tf.linalg.eigh( H )
        adjW= tf.linalg.adjoint( W )
        adjx = tf.linalg.adjoint( x )
        
        J = backend.dot( gL , adjx )
        
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
        
        # !!!! BEST w/ Base1 Mapping !!!!
        gH = Dv_H - tf.linalg.diag( tf.linalg.diag_part( Dv_H ) ) / 2.
        gx = backend.dot( H , gL ) + backend.dot( tf.linalg.matrix_transpose( H ) , gL )
        
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
            method = 1,
            activation = 'M1',
            recurrent_weight = 'O',
            input_weight = 'GU',
            **kwargs
        ):
        
        super( HopfRNNCellTheta , self ).__init__( **kwargs )
        
        self.units = units
        
        self.method = method
        self.actnme = activation
        self.rwgtnme = recurrent_weight
        self.iwgtnme = input_weight
        
        self.activation = None
        self.recwgt = None
        self.inpwgt = None
        
        self.state_size = units
        self.output_size = units
        self.activation = activation
        self.hopfact = None
        
        
        self.method = method
        
        if isinstance( activation , str ):
            tstact = activation.lower()
            if 'm1' == tstact: self.hopfact = HopfActivCpx()
#            if 'm2' == tstact: self.hopfact = HopfActivCpx2()
            elif 't' == tstact: self.activation = tf.keras.activations.tanh
            elif 'cr' == tstact: self.activation = CpxReLU()
#            if 'mr' == tstact: self.activation = ModReLU()
            else: self.activation = None
        
        if isinstance( recurrent_weight , str ):
            tstrwgt = recurrent_weight.lower()
            if 'h' == tstrwgt: self.recwgt = Hermitian
            elif 'u' == tstrwgt: self.recwgt = Unitary
            elif 'o' == tstrwgt: self.recwgt = Orthogonal
            elif 'gn' == tstrwgt: self.recwgt = GlorotNorm
            elif 'gu' == tstrwgt: self.recwgt = GlorotUnif
            elif 'i' == tstrwgt: self.recwgt = Eye
            
        if isinstance( input_weight , str ):
            tstiwgt = input_weight.lower()
            if 'h' == tstiwgt: self.inpwgt = Hermitian
            elif 'u' == tstiwgt: self.inpwgt = Unitary
            elif 'o' == tstiwgt: self.inpwgt = Orthogonal
            elif 'gn' == tstiwgt: self.inpwgt = GlorotNorm
            elif 'gu' == tstiwgt: self.inpwgt = GlorotUnif
            elif 'i' == tstiwgt: self.inpwgt = Eye
            
    def build( self , input_shape ):
        
        '''
        print( self.name + ' units:' , self.units )
        print( self.name + ' state_size:' , self.state_size )
        print( self.name + ' output_size:' , self.output_size )
        print( self.name + ' input_shape:' , input_shape )
#        exit(0)
#       '''
        
        self.mxlen = max( input_shape[-1] , self.units )
#        print( self.name + ' mxlen' , self.mxlen )
        
        do_save = False
        do_train = True
        
        mtd = str( self.method ) + '_' + str( input_shape[0] )
        self.A = self.add_weight(
            shape = ( input_shape[0] , input_shape[0] ),
            initializer = self.recwgt( name = self.rwgtnme + mtd , save = do_save ),
            trainable = do_train,
            dtype = tf.complex64,
            name = 'A' + mtd + '_' + self.name
        )
        
        self.B = self.add_weight(
            shape = ( input_shape[0] , input_shape[0] ),
            initializer = self.inpwgt( name = self.iwgtnme + mtd , save = do_save ),
            trainable = do_train,
            dtype = tf.complex64,
            name = 'B' + mtd + '_' + self.name
        )
        
        self.onslke = tf.constant( 
            tf.ones( shape = ( input_shape[0] , 1 ) , dtype = tf.complex64 ),
            dtype = tf.complex64
        )
        
        self.step = tf.Variable( [ 0 ] , trainable = False , dtype = tf.int32 )
        
        super( HopfRNNCellTheta , self ).build( input_shape )
        
        self.built = True
        
        
    @tf.function
    def split_input( self , x ):
        x_stk = tf.unstack( tf.expand_dims( tf.linalg.matrix_transpose( x ) , -1 ) )
        return x_stk[0] , x_stk[1::]
        
    @tf.function
    def concat_output( self , v_0 , v_n ):
        return tf.linalg.matrix_transpose(
            tf.squeeze( tf.stack( [ v_0 ] + v_n ) , -1 )
        )
    
    @tf.function
    def concat_roll_output( self , v_i , v_n ):
        return tf.linalg.matrix_transpose(
            tf.roll( tf.squeeze( tf.stack( [ v_i ] + v_n ) , -1 ) , -1 , 0 )
        )
        
    @tf.function
    def method_one( self , A , B , x , z ):
        
        z_0 , z_n = self.split_input( z )
        x_0 , x_n = self.split_input( x )
        
        Az = tf.linalg.matmul( A , z_0 )
        Bx = tf.linalg.matmul( B , x_0 )
#        Az = herm_map_V1( self.A , z_n )
#        Bx = herm_map_V1( self.B , x )
#        Az = herm_map_V2( z_0 , self.A )
#        Bx = herm_map_V2( x_0 , self.B )
#        _print( self.name + ' Az' , Az , summarize = -1 )
#        _print( self.name + ' Bx' , Bx , summarize = -1 )
        
        z_i = ( Az + Bx ) / 2.
#        _print( 'z_i' , z_i )
        
        if self.hopfact is not None:
            zros = tf.stop_gradient( tf.ones_like( z_i ) )
            y_t = self.hopfact( z_0 , z_i , zros , self.step , self.units )
#            _print( self.name + ' hopf1 -> y_t' , y_t )
        elif self.activation is not None:
            y_t = self.activation( y_i )
#            _print( self.name + ' act -> y_t' , y_t )
        else:
            y_t = z_i
#            _print( self.name + ' None -> y_t' , y_t )
        
        z_j = self.concat_roll_output( z_i , z_n )
        y_t = y_t + z_j
        
        return y_t , z_j
        
        
    @tf.function
    def method_two( self , A , B , x , z ):
        
        Az = tf.linalg.matmul( A , z )
        Bx = tf.linalg.matmul( B , x )
#        Az = herm_map_V1( self.A , z_n )
#        Bx = herm_map_V1( self.B , x )
#        Az = herm_map_V2( z_0 , self.A )
#        Bx = herm_map_V2( x_0 , self.B )
#        _print( self.name + ' Az' , Az , summarize = -1 )
#        _print( self.name + ' Bx' , Bx , summarize = -1 )
        
        z_i = ( Az + Bx ) / 2.
#        _print( 'z_i' , z_i )
        
        if self.hopfact is not None:
            
            @tf.function
            def _hopf( args ):
                a , b = args
                zros = tf.stop_gradient( tf.ones_like( b ) )
                return ( self.hopfact( a , b , zros , self.step , self.units ) )
            
            y_t = tf.vectorized_map( _hopf , ( tf.expand_dims( z , 1 ) , tf.expand_dims( z_i , 1 ) ) )
            y_t = tf.squeeze( tf.reshape( y_t , [ z.shape[0] , 1 , z.shape[-1] ] ) , 1 )
#            _print( self.name + ' hopf1 -> y_t' , y_t )
            
        elif self.activation is not None:
            y_t = self.activation( y_i )
#            _print( self.name + ' act -> y_t' , y_t )
        else:
            y_t = z_i
#            _print( self.name + ' None -> y_t' , y_t )
        
        z_j = z_i
        
#        z_j = self.concat_roll_output( z_i , z_n )
#        y_t = y_t + z_j
        
        return y_t , z_j
        
        
    def call( self , inputs , states , training = False ):
        
#        tf.print( '\n'+'- '*20+' Cell '+'- '*20+'\nCount:' , self.step[0] , '\n' )
        
        x = inputs[0] if tf.nest.is_nested( inputs ) else inputs
        z = states[0] if tf.nest.is_nested( states ) else states
#        _print( self.name + ' x' , x )
#        _print( self.name + ' z' , z )
        
        if self.method == 1:
            y_t , z_j = self.method_one( self.A , self.B , x , z )
        else:
            y_t , z_j = x , z
        
        self.step.assign( tf.math.floormod( self.step + 1 ,  ( self.units + 1 ) ) )
        
        z_t = [ z_j ] if tf.nest.is_nested( states ) else z_j
        
        return y_t , z_t
        
        # Extra Stuff ... #
#        self.step.assign_add( [ 1 ] )
#        tf.debugging.check_numerics( tf.math.abs( z_t ) , 'var: z_t  ->  hopf cell has nan value' )
        
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
        method = 1,
        activation = 'M1',
        recurrent_weight = 'O',
        input_weight = 'GU',
        return_sequences = False,
        time_major = False,
        stateful = False,
        **kwargs
    ):
        
        super( HopfRNNLayerTheta , self ).__init__( **kwargs )
        
        # To hold the cell(s) for the model.
        self.cell = None
        
        self.units = units if units > 3 else 4
        self.osz = units
        
        self.method = method
        self.activation = activation
        self.recurrent_weight = recurrent_weight
        self.input_weight = input_weight
        
        self.return_sequences = return_sequences
        self.stateful = stateful
        self.time_major = time_major
        
        
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
        self.isz = inshp[1]
        self.vsz = inshp[2] if len( inshp ) == 3 else -1
        
        self.dofft2 = False if self.vsz == 1 else True
#        print( 'dofft2' , self.dofft2 )
        
        maxseq = max( self.units , self.isz // 2 + 1 , self.vsz // 2 + 1 )
#        print( 'maxseq' , maxseq )
        
        self.seqlen = ( maxseq - 1 ) * 2
#        print( 'seqlen' , self.seqlen )
#        exit(0)
        
        self.fftnorm = tf.constant(
            tf.cast( self.seqlen , dtype = tf.complex64 ),
#            tf.cast( tf.math.sqrt( 0. + self.seqlen ) , dtype = tf.complex64 ),
            dtype = tf.complex64
        )
        
        # Generate cell -> Only use 1 for now, until it's working better.
        if self.cell is None:
            self.cell = HopfRNNCellTheta(
                units = self.units,
                method = self.method,
                activation = self.activation,
                recurrent_weight = self.recurrent_weight,
                input_weight = self.input_weight,
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
        
        super( HopfRNNLayerTheta , self ).build( input_shape )
        
        self.step = tf.Variable( [ 0 ] , trainable = False , dtype = tf.int32 )
        
        self.built = True
        
        
    def call( self , inputs , training = False ):
        
#        tf.print( '\n'*8+'=='*25+' Layer '+'=='*25+'\nCount:',self.step[0],'\n' )
#        _print( self.name + ' inputs' , inputs )
        
        _input = inputs.transpose(( 0 , 2 , 1 ))
#        _print( self.name + ' A) _input' , _input )
        
        if self.dofft2:
            _input = tf.signal.rfft2d( _input , fft_length = [ self.seqlen , self.seqlen ] )
            _input = _input / self.fftnorm
            _input = _input[::,0:self.units,::]
#            _print( self.name + ' B) fft2 _input' , _input )
#            _print( self.name + ' C) fft2 _input' , _input )
        else:
            _input = tf.signal.rfft( _input , fft_length = [ self.seqlen ] )
#            _input = _input / self.fftnorm
#            _input = _input[::,::,0:self.units]
#            _print( self.name + ' B) fft1 _input' , _input )
#            _print( self.name + ' C) fft1 _input' , _input )
        
        _input = _input / self.fftnorm
        _input = _input[::,::,0:self.units]
        _input = _input.transpose(( 0 , 2 , 1 ))
#        _print( 'D) _input' , _input )
        
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
        
#        _output = _output * tf.math.sqrt( 0. + self.seqlen )
        _output = _cout.transpose(( 0 , 2 , 1 )) * self.fftnorm
#        _print( 'A) _output' , _output )
        
        if self.dofft2:
            _output = _output * self.fftnorm
            _output = tf.signal.irfft2d( _output , fft_length = [ self.seqlen , self.seqlen ] )
#            _output = _output[::,0:self.osz,0:self.isz]
#            _print( 'B) fft2 _output' , _output )
#            _print( 'C) fft2 _output' , _output )
        else:
            _output = tf.signal.irfft( _output , fft_length = [ self.seqlen ] )
#            _output = _output[::,0:self.osz,0:self.isz]
#            _print( 'B) fft1 _output' , _output )
#            _print( 'C) fft1 _output' , _output )
        
        _output = _output[::,0:self.osz,0:self.isz]
        _output = _output.transpose(( 0 , 2 , 1 ))
#        _print( 'D) _output' , _output )
        
        if not self.return_sequences: _output = _output[::,-1,::]
#        _print( 'E) _output' , _output )
        
#        max_img_val = tf.math.reduce_max( tf.math.imag( _output ) )
#        tf.debugging.assert_less( max_img_val , 1.e-5, 'imaginary values went above 1.e-5 issue with irfft() input map.' )
#        tf.print('maximg:', max_img_val )
        
        outputs = tf.cast( _output , inputs.dtype )
#        _print( 'output' , output )
        
        self.step.assign_add( [ 1 ] )
        
        return outputs
        
        
    def get_config(self):
        config = super( HopfRNNLayerTheta , self ).get_config()
        config.update( { 'state_size' : self.state_size } )
        return config
        
        
    @classmethod
    def from_config( self , config ):
        return self( **config )



    @tf.function
    def rnn_call( self , cell , inputs , states , time_major=False , return_all_outputs=False , training=False):
        
        flat_states = tf.nest.flatten( states )
        
        # Get cell's calling function
        cell_call_fn = ( cell.__call__ if callable(cell) else cell.call )
        
        # Callback for backend.rnn() operations.
        def _step(step_inputs, step_states):
            step_states = step_states[0]
            output, new_states = cell_call_fn(step_inputs, step_states, training=training)
            return output, new_states
        
        new_lastouts, new_outputs, new_states = tf.keras.backend.rnn(
            _step,
            inputs,
            flat_states,
            time_major = time_major,
            return_all_outputs=return_all_outputs
        )
        
        _lastout = new_lastouts[0] if tf.nest.is_nested( new_lastouts ) else new_lastouts
#        _print('_lastout',_lastout)
        
        _outputs = new_outputs[0] if tf.nest.is_nested( new_outputs ) else new_outputs
#        _print('_outputs',_outputs)
        
        _states = new_states[0] if tf.nest.is_nested( new_states ) else new_states
#        _print('_states',_states)
        
        return ( _lastout , _outputs , _states )

