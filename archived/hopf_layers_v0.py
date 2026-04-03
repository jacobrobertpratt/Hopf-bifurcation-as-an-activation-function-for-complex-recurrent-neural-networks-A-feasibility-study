import os

import numpy as np

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow.python.keras import backend
np_config.enable_numpy_behavior()

''' Local Imports '''
from proj_utils import _print
from activations import HopfActCpx , CpxReLU , ModReLU , CpxCard
from initializers import Eye , Orthogonal , GlorotNorm , GlorotUnif , Hermitian , Unitary
    

@tf.custom_gradient
def _pgrad( inpt ):
    out = inpt
    def grad( gL ):
        _print( 'gL' , gL )
        return gL
    return out , grad
    
    
@tf.custom_gradient
def herm_map( v , H ):
    
    y = tf.linalg.matmul( v , H )
    
    def grad( gL ):
        ''' SOURCE:
        The following directional derivative calculation is based on the paper by Najfeld and Havel.
        Cite: I. Najfeld and T. F. Havel, “Derivatives of the Matrix Exponential and Their Computation,”
        Advances in Applied Mathematics, vol. 16, no. 3, pp. 321–375, Sep. 1995, doi: 10.1006/aama.1995.1017.
        '''
        
        lmda , W = tf.linalg.eigh( H )
        adjW= tf.linalg.adjoint( W )
        adjv = tf.linalg.adjoint( v )
        
        J = backend.dot( adjv , gL )
        
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
        gv = backend.dot( gL , H ) + backend.dot( gL , tf.linalg.matrix_transpose( H ) )
        
        return gv , gH
        
    return y , grad
    
    
@tf.custom_gradient
def unit_map( self , U , x ):
    
    y = tf.linalg.matmul( U , x )
    
    def grad( gl ):
        
        _print( 'gL' , gL )
        
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
            activation = 'M1',
            recurrent_weight = 'O',
            input_weight = 'GU',
            train_weights = True,
            save_weights = True,
            **kwargs
        ):
        
        super( HopfRNNCellTheta , self ).__init__( **kwargs )
        
        self.units = units
        self.train_weights = train_weights
        self.save_weights = save_weights
        
        self.actnme = activation
        self.rwgtnme = recurrent_weight
        self.iwgtnme = input_weight
        
        self.activation = None
        self.recwgt = None
        self.inpwgt = None
        
        self.state_size = units
        self.output_size = units
        self.activation = activation
        self.hopfact1 = None
        self.hopfact2 = None
        
        if isinstance( activation , str ):
            tstact = activation.lower()
            if 'm1' == tstact: self.hopfact1 = HopfActCpx( units = self.units )
            if 'm2' == tstact: self.hopfact2 = HopfActCpx(units = self.units )
            elif 't' == tstact: self.activation = tf.keras.activations.tanh
            elif 'cr' == tstact: self.activation = CpxReLU()
            elif 'cc' == tstact: self.activation = CpxCard()
            elif 'mr' == tstact: self.activation = ModReLU()
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
        exit(0)
        #'''
        
        inshp = list( input_shape )
        self.bsz = inshp[0]
        self.isz = inshp[-1]
        
        do_save = self.save_weights
        do_train = self.train_weights
        
        wgtshp = ( self.units , self.units )
        
        wgtnme = 'u' + str( self.units ) + '_' + self.name.split('_')[0]
        self.A = self.add_weight(
            shape = wgtshp,
            initializer = self.recwgt( name = self.rwgtnme + '_' + wgtnme , save = do_save ),
            trainable = do_train,
            dtype = tf.complex64,
            name = 'A_' + wgtnme
        )
        
        self.B = self.add_weight(
            shape = wgtshp,
            initializer = self.inpwgt( name = self.iwgtnme + '_' + wgtnme , save = do_save ),
            trainable = do_train,
            dtype = tf.complex64,
            name = 'B_' + wgtnme
        )
        
        '''
        self.bias = self.add_weight(
            shape = ( self.units , ),
            initializer = tf.zeros,
            trainable = do_train,
            dtype = tf.complex64,
            name = 'bias_' + wgtnme
        )
        #'''
        
        self.step = tf.Variable( [ 0 ] , trainable = False , dtype = tf.int32 )
        
        super( HopfRNNCellTheta , self ).build( input_shape )
        
        self.built = True
        
        
    @tf.function
    def split_input( self , v ):
        vstk = tf.unstack( v , axis = -1 )
        return tf.expand_dims( vstk[0] , -1 ) , tf.stack( vstk[1::] , axis = -1 )
        
    @tf.function
    def combine_output( self , v0 , v_ ):
        vlst = [ tf.squeeze( v0 , -1 ) ] + tf.unstack( v_ , axis = -1 )
        return tf.stack( vlst , axis = -1 )
        
    @tf.function
    def std_map( self , A , B , v1 , v2 ):
        Av1 = tf.linalg.matmul( v1 , A )
        Bv2 = tf.linalg.matmul( v2 , B )
        return Av1 + Bv2 , Av1 , Bv2
    
    @tf.function
    def get_2plx( self , vals ):
        re , im = tf.math.real( vals ) , tf.math.imag( vals )
        _2plx = tf.cast( [[ re, -im ],[ im , re ]] , dtype=re.dtype ).transpose((2,3,0,1))
        return _2plx
        
    @tf.function
    def get_det( self , vals ):
        _2plx = self.get_2plx( vals )
        return tf.linalg.det( _2plx ) 
    
    @tf.function
    def get_trc( self , vals ):
        _2plx = self.get_2plx( vals )
        return tf.linalg.trace( _2plx ) 
        
    @tf.function
    def eig_vals( self , vals ):
        _2plx = self.get_2plx( vals )
        det = tf.linalg.det( _2plx )
        trc = tf.linalg.trace( _2plx )
        dif = trc**2 - 4.*det
        sqr = tf.math.sqrt( tf.cast( dif , dtype=vals.dtype ) )
        evls = ( trc + sqr ) / 2.
        return evls
        
    @tf.function
    def mkeunt( self , v ):
        return v / tf.math.abs( v )
        
    def call( self , inputs , states , training = False ):
        
#        tf.print( '\n'+'- '*20+' Cell '+'- '*20+'\nCount:' , self.step[0] , '\n' )
        
        z = states[0] if tf.nest.is_nested( states ) else states
        x = inputs[0] if tf.nest.is_nested( inputs ) else inputs
#        _print( self.name + ' z' , z )
#        _print( self.name + ' x' , x )
        
        z0 , z_ = self.split_input( z )
        x0 , x_ = self.split_input( x )
#        _print( self.name + ' z_' , z_ , summarize=-1 )
        
        z_i , Az , Bx = self.std_map( self.A , self.B , z_ , x_ )
#        _print( 'z_i' , z_i )
        
        if self.hopfact1 is not None:
            b_i = tf.stop_gradient( tf.ones_like( z_i ) )
            y_k = self.hopfact1( z_ , z_i , b_i )
        elif self.hopfact2 is not None:
            b_i = tf.math.pow( -1.j*z_i , 1. / self.units )
            y_k = self.hopfact2( z_ , z_i , b_i )
        elif self.activation is not None:
            y_k = self.activation( z_i )
        else:
            y_k = z_i
#        _print( self.name + ' y_i' , y_i )
#        _print( self.name + ' z_i' , z_i )
        
        re_j = ( z0 + x0 ) / 2.
        y_t = self.combine_output( re_j , y_k )
        
        z_k , _ = tf.linalg.normalize( z_ + z_i , ord = 2 ) # , axis=[-2,-1] )
        z_t = self.combine_output( re_j , z_k )
        
        z_t = [ z_t ] if tf.nest.is_nested( states ) else z_t
#        _print( 'y_t' , y_t )
#        _print( 'z_t' , z_t )
        
#        tf.debugging.check_numerics( tf.math.abs( z_t ) , 'var: z_t  ->  hopf cell has nan value' )

#        self.step.assign( tf.math.floormod( self.step + 1 ,  ( self.units + 1 ) ) )
        
        return y_t , z_t
        
        # Extra Stuff ... #
#        tf.debugging.check_numerics( tf.math.abs( z_t ) , 'var: z_t  ->  hopf cell has nan value' )
        
        
@tf.keras.utils.register_keras_serializable( 'hopf_theta_layer' )
class HopfRNNLayerTheta( tf.keras.layers.Layer ):
    
    def __init__(
        self,
        units,
        activation = 'm1',
        recurrent_weight = 'O',
        input_weight = 'GU',
        return_sequences = False,
        time_major = False,
        stateful = False,
        train_weights = True,
        save_weights = True,
        **kwargs
    ):
        
        super( HopfRNNLayerTheta , self ).__init__( **kwargs )
        
        # To hold the cell(s) for the model.
        self.cell = None
        
        self.units = units if units > 3 else 4
        self.osz = units
        
        self.activation = activation
        self.recurrent_weight = recurrent_weight
        self.input_weight = input_weight
        
        self.time_major = time_major
        self.return_sequences = return_sequences
        self.stateful = stateful
        self.train_weights = train_weights
        self.save_weights = save_weights
        
        
    def build( self , input_shape ):
        
        '''
        print( self.name + ' input_shape:' , input_shape )
        print( self.name + ' units:', self.units )
        print( self.name + ' return_sequences:', self.return_sequences )
        print( self.name + ' stateful:', self.stateful )
        print( self.name + ' time_major:', self.time_major )
        print( self.name + ' activation:', self.activation )
        #exit(0)
        #'''
        
        self.usz = self.units + 1
        
        inshp = list( input_shape )
        self.bsz = inshp[0]
        self.isz = inshp[1]
        self.vsz = inshp[2] if len( inshp ) == 3 else -1
        
        # Calc. max rfft mapping length -> depends on units and window size
        rfft_vsz = self.vsz // 2 + 1
        maxseq = max( self.units , rfft_vsz )
        
        # Add 1 if same as cell units -> we want to be just a bit larger to avoid last column imaginary zeros
        if maxseq == self.units: maxseq = maxseq + 1
        
        self.seqlen = maxseq * 2 # ( maxseq - 1 ) * 2 + 2   # x4 instead of x2 to extend input rfft
        
        '''
        print( 'units:' , self.units )
        print( 'usz:' , self.usz )
        print( 'isz rfft size:' , self.isz // 2 + 1 )
        print( 'vsz rfft size:' , self.vsz // 2 + 1 )
        print( 'maxseq' , maxseq )
        exit(0)
        #'''
#        print( 'seqlen' , self.seqlen )
        
        ## Used to normalize the input rfft mapping ##
        fftnrm = tf.cast( tf.math.sqrt( 0. + self.seqlen ) , dtype = self.dtype )  # Fails when using the Hopf-Activation
#        fftnrm = tf.cast( self.seqlen / 2. , dtype = self.dtype )
        self.fftnorm = tf.constant( fftnrm , dtype = self.dtype )
#        _print( 'fftnrm' , fftnrm )
#        exit(0)
        
        # Create internal hopf cell (represents a single timestep )
        if self.cell is None:
            self.cell = HopfRNNCellTheta(
                units = self.units,
                activation = self.activation,
                recurrent_weight = self.recurrent_weight,
                input_weight = self.input_weight,
                train_weights = self.train_weights,
                save_weights = self.save_weights,
                dtype = self.dtype,
                name = self.name + '_cell'
            )
            
        # From tf.keras.layers.basernn -> calculate cell input shape
        def get_step_input_shape( shape ):
            if isinstance( shape , tf.TensorShape ):
                shape = tuple( shape.as_list() )
            return ( shape[0] , ) + shape[2:]
        step_input_shape = tf.nest.map_structure( get_step_input_shape , input_shape )
        
        # Build cell
        if not self.cell.built:
            with tf.name_scope( self.cell.name ):
                self.cell.build( step_input_shape )         # Last build shape -> self.cells.build( input_shape[1::] )
                assert self.cell.built , 'HopfBifurCpxRNNLayer failed to build.'
                
        state_init = tf.ones( shape = ( self.bsz , self.usz ) , dtype = self.dtype )
        self.state = tf.Variable(
            initial_value = state_init,
            shape = state_init.shape,
            dtype = self.dtype,
            trainable = False,
            name = 'state_'+self.name
        )
#        _print( self.name + ' state' , self.state )
        
        super( HopfRNNLayerTheta , self ).build( input_shape )
        
#        self.step = tf.Variable( [ 0 ] , trainable = False , dtype = tf.int32 )
        
        self.built = True
        
        
    def call( self , inputs , training = False ):
        
#        tf.print( '\n'*8+'=='*25+' Layer '+'=='*25, '\nCount:' , self.step[0] , '   seqlen:' , self.seqlen , '\n' )
#        _print( self.name + ' A) inputs' , inputs )
        
        _input = tf.signal.rfft( inputs , fft_length = [ self.seqlen ] )
#        _print( self.name + ' B) _input' , _input[0] )
        
        _input = _input[::,::,0:self.usz] / self.fftnorm
#        _print( self.name + ' C) _input' , _input[0] )
        
        # Reset the cell's counter
#        self.cell.step.assign( [ 0 ] )
        
        rnn_return = self.rnn_call(
            self.cell,
            _input,
            self.state,
            return_all_outputs = True,    # If True -> seq_out has batch sized output; else, single output
            training = training
        )
        _clast , _cout , _cstate = rnn_return
        
        if self.stateful: self.state.assign( _cstate )
        
#        tf.print( '\n'+' *'*15 + ' RNN Returned ' + '* '*15 + '\nCount:' , self.step[0] , '   seqlen:' , self.seqlen , '\n' )
#        _print( self.name + ' _clast' , _clast )
#        _print( self.name + ' _cout' , _cout )
#        _print( self.name + ' _cstate' , _cstate )
        
        _output = _cout * self.fftnorm
        _output = tf.signal.irfft( _output , fft_length = [ self.seqlen ] )
#        _print( self.name + ' A) _output' , _output )
        
        _output = _output[::,0:self.isz,0:self.osz]
#        _print( self.name + ' B) _output' , _output )
        
        if not self.return_sequences: _output = _output[::,-1,::]
        
        # Used for Testing #
#        max_img_val = tf.math.reduce_max( tf.math.imag( _output ) )
#        tf.debugging.assert_less( max_img_val , 1.e-5, 'imaginary values went above 1.e-5 issue with irfft() input map.' )
#        tf.print('maximg:', max_img_val )
        
#        self.step.assign_add( [ 1 ] )
        
        return tf.cast( _output , inputs.dtype )
        
        
    def get_config(self):
        config = super( HopfRNNLayerTheta , self ).get_config()
        config.update( { 'state_size' : self.state_size } )
        return config
        
        
    @classmethod
    def from_config( self , config ):
        return self( **config )



    @tf.function
    def rnn_call( self , cell , inputs , states , time_major=False , input_length = None , return_all_outputs=False , training=False):
        
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

