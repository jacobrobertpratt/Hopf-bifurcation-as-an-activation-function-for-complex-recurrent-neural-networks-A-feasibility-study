''' Standard Imports '''
import os

''' Special Imports '''
import numpy as np
from scipy.integrate import odeint

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

'''
Also get Citation for NeuralODE paper. Chen et al., 2018
Citation for tensorflow Probability:
J. V. Dillon et al., “TensorFlow Distributions.”
arXiv, Nov. 28, 2017. doi: 10.48550/arXiv.1711.10604. '''
import tensorflow_probability as tfp

''' Local Imports '''
from proj_utils import _print


@tf.keras.utils.register_keras_serializable( 'activations' )
class HopfActCpx( tf.keras.layers.Layer ):
    
    def __init__( self , units = 8 , **kwargs ):
        
        self.units = units
        
        self.dopri = tfp.math.ode.DormandPrince(
#            rtol=1e-3,
#            atol=1e-6,
            safety_factor = 0.9,
            first_step_size = 1.e-6,
            max_num_steps = int( 5e3 )
        )
        
        super( HopfActCpx , self ).__init__( **kwargs )
        
    def build( self , input_shape ):
        
        super( HopfActCpx , self ).__init__()
        
        self.built = True
        

        
    @tf.function
    def cpx_hopf_DiffEQ( self , t , z_stk , re_a , im_a , re_b , im_b ):
        
        re_z , im_z = tf.unstack( z_stk )
        z = tf.complex( re_z , im_z )
        a = tf.complex( re_a , im_a )
        b = tf.complex( re_b , im_b )
        
        @tf.function
        def hopf_map( z , a , b ):
            zz = tf.cast( tf.math.pow( tf.math.abs( z ) , 2 ) , dtype = z.dtype )
            bzz = tf.math.multiply( (-1.+0.j)*b , zz )
            w = a + bzz
            dz = tf.math.multiply( w , z )
            return dz
        
        dz = hopf_map( z , a , b )
        
        return tf.stack( [ tf.math.real( dz ) , tf.math.imag( dz ) ] )
        
#        tf.debugging.check_numerics( tf.math.abs( z ) , 'var: z  ->  hopf activation has nan value' )
#        dr = tf.math.multiply_no_nan( ( a - r * r ) , r )
#        dr = ( a - r*r ) * r
    
    @tf.function
    def cpx_hopf_ODE( self , z , a , b ):
        
        t_0 = 0.
        t_n = 2*np.pi
        t_n = tf.linspace( t_0 , t_n , num = 8 )
        
        re_a , im_a = tf.math.real( a ) , tf.math.imag( a )
        re_b , im_b = tf.math.real( b ) , tf.math.imag( b )
        
        state = tf.stack( [ tf.math.real( z ) , tf.math.imag( z ) ] )
        const = { 're_a' : re_a , 'im_a' : im_a , 're_b' : re_b , 'im_b' : im_b }
        ode = self.dopri.solve(
            self.cpx_hopf_DiffEQ,
            t_0,
            state,
            solution_times = t_n,
            constants = const
        )
        re_zt , im_zt = tf.unstack( ode.states[-1] )
        return tf.complex( re_zt , im_zt )
        
    def call( self , z , a , b ):
        z_0 , a_0 , b_0 = tf.cast( z , dtype = tf.complex128 ) , tf.cast( a , dtype = tf.complex128 ) , tf.cast( b , dtype = tf.complex128 )
        z_t = self.cpx_hopf_ODE( z_0 , a_0 , b_0 )
        return tf.cast( z_t , dtype = z.dtype )
        
        
        
@tf.keras.utils.register_keras_serializable( 'activations' )
class CpxReLU( tf.keras.layers.Layer ):
    
    def __init__( self , **kwargs ):
        if 'name' not in kwargs: kwargs['name'] = 'cpxrelu'
        self.act = tf.keras.activations.relu
        super( CpxReLU , self ).__init__( **kwargs )
        
        
    def build( self , input_shape ):
        super( CpxReLU , self ).__init__()
        self.built = True
        
    def call( self , z ):
        re , im = self.act( tf.math.real( z ) ) , self.act( tf.math.imag( z ) )
        return tf.complex( re , im )
        
        
        
@tf.keras.utils.register_keras_serializable( 'activations' )
class ModReLU( tf.keras.layers.Layer ):
    
    ''' SOURCE:
    [1] M. Arjovsky, A. Shah, and Y. Bengio, “Unitary Evolution Recurrent Neural Networks,” p. 9, 2016.
    Quote: It is a pointwise nonlinearity, σmodReLU(z) : C → C, which affects only the absolute value of a complex number.
    '''
    
    def __init__( self , **kwargs ):
        
        if 'name' not in kwargs: kwargs['name'] = 'modrelu'
        if 'dtype' not in kwargs: kwargs['dtype'] = tf.float32
        
        super( ModReLU , self ).__init__( **kwargs )
        
        
    def build( self , input_shape ):
        
#        print( 'input_shape:' , input_shape )
        
        inshp = list( input_shape )
        self.bsz = inshp[0]
        self.isz = inshp[-1]
        
        def get_init_wgt( shape , dtype ):
            return tf.zeros( shape = shape , dtype = dtype )
        
        bshp = ( self.bsz , self.isz )
        self.bias = self.add_weight(
            shape = bshp,
            initializer = get_init_wgt,
            trainable = True,
            dtype = self.dtype,
            name = self.name + 'bias'
        )
        
        self.zro = tf.cast( [0.] , dtype = self.dtype )
        
        super( ModReLU , self ).__init__()
        
        self.built = True
        
    def call( self , z ):
        abs = tf.math.abs( z )
        unt = z / tf.cast( abs , dtype = z.dtype )
        max = tf.math.maximum( abs + self.bias , self.zro )
        max = tf.cast( max , dtype = z.dtype )
        return tf.math.multiply( max , unt )
        
        
@tf.keras.utils.register_keras_serializable( 'activations' )
class CpxCard( tf.keras.layers.Layer ):
    
    ''' SOURCE:
    [1] P. Virtue, S. X. Yu, and M. Lustig, “Better than real: Complex-valued neural nets for MRI fingerprinting,”
    2017 IEEE International Conference on Image Processing (ICIP), Beijing: IEEE, Sep. 2017, pp. 3953–3957.
    doi: 10.1109/ICIP.2017.8297024.
    Quote:  With this activation, input values that lie on the positive real axis are scaled by one, input values on the
            negative real axis are scaled by zero, and input values with nonzero imaginary components are gradually scaled
            from one to zero as the complex number rotates in phase from positive real axis towards the negative real axis.
            When the input values are restricted to real values, the complex cardioid function is simply the ReLU activation
            function.
    '''
    
    def __init__( self , **kwargs ):
        if 'name' not in kwargs: kwargs['name'] = 'complex_cardioid'
        super( CpxCard , self ).__init__( **kwargs )
        
    def build( self , input_shape ):
        super( CpxCard , self ).__init__()
        self.built = True
        
    def call( self , z ):
        ang = tf.math.angle( z )
        crd = tf.cast( ( tf.math.cos( ang ) + 1. ) / 2. , dtype = z.dtype )
        return tf.math.multiply( crd ,  z )
        
        
        