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
class HopfActivCpx( tf.keras.layers.Layer ):
    
    def __init__( self , **kwargs ):
        
        self.dopri = tfp.math.ode.DormandPrince(
            rtol = 1.49012e-8,
            atol = 1.49012e-8,
            first_step_size = 1.e-6,
            max_num_steps = int( 1e4 )
        )
        
        self.eps = 1.e-3
        
        super( HopfActivCpx , self ).__init__( **kwargs )
        
        
    def build( self , input_shape ):
        
        super( HopfActivCpx , self ).__init__()
        
        self.built = True
        
        
    @tf.function
    def cpx_hopf_DiffEQ( self , t , z , re_a , im_a , re_b , im_b ):
    
#        tf.print( '-'*10 , 'time:' , t , '-'*10 )
    
        re_z , im_z = tf.unstack( z )
        z = tf.complex( re_z , im_z )
        a = tf.complex( re_a , im_a )
        b = tf.complex( re_b , im_b )
#        _print( 'z' , z )
#        _print( 'a' , a )
#        _print( 'b' , b )
        
#        tf.debugging.check_numerics( tf.math.abs( z ) , 'var: z  ->  hopf activation has nan value' )
        
#        zz = tf.math.conj( z ) * z
        zz = tf.math.multiply( tf.math.conj( z ) , z )
#        _print( 'zz' , zz )
        
#        bzz = b * zz
        bzz = tf.math.multiply( b , zz )
#        _print( 'bzz' , bzz )
        
        w = a - bzz
#        _print( 'w' , w )
        
#        dz = w * z
        dz = tf.math.multiply( w , z )
#        _print( 'dz' , dz )
        
#        tf.debugging.check_numerics( tf.math.abs( dz ) , 'var: dz  ->  hopf activation has nan value' )
        
        return tf.stack( [ tf.math.real( dz ) , tf.math.imag( dz ) ] )
        
#        dr = tf.math.multiply_no_nan( ( a - r * r ) , r )
#        dr = ( a - r*r ) * r
        
    @tf.function
    def cpx_hopf_ODE( self , z , a , b , stp , unts ):
        
        '''
        t_0 = (stp[0]+0.)*np.pi*2 / unts
        t_1 = t_0 + 2*np.pi / unts
        t_n = [ t_0 , t_1 ]
#        tf.print( 't_0:' , t_0 )
#        tf.print( 't_1:' , t_1 )
        '''
        t_0 = 0.
        t_1 = 2*np.pi - 2*np.pi / unts
        t_n = tf.linspace( t_0 , t_1 , num = unts )
#        '''
        
        re_a = tf.math.real( a )
        im_a = tf.math.imag( a )
        re_b = tf.math.real( b )
        im_b = tf.math.imag( b )
        
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
        
        
    def call( self , z , a , b , stp , unts ):
        z_t = self.cpx_hopf_ODE( z , a , b , stp , unts )
        return z_t
    


@tf.keras.utils.register_keras_serializable( 'activations' )
class HopfActivTheta( tf.keras.layers.Layer ):
    
    def __init__( self , **kwargs ):
        
        self.dopri = tfp.math.ode.DormandPrince(
            rtol = 1.49012e-8,
            atol = 1.49012e-8,
#            first_step_size = 1.e-6,
            max_num_steps = int( 5e3 )
        )
        
        self.eps = 1.e-3
        
        super( HopfActivTheta , self ).__init__( **kwargs )
        
        
    def build( self , input_shape ):
        self.built = True
        
        
    @tf.function
    def polar_hopf_DiffEQ( self , t , z , a , b ):
        
#        tf.print( '-'*10,'time:',t,'-'*10)
        
        r , w = tf.unstack( z )
        
#        _print( 'r' , r )
        mxr = tf.math.reduce_max( r )
        tf.debugging.assert_less( mxr , tf.cast( 1.e6 , dtype=r.dtype ) , 'var: r  ->  above 1e6 in hopf activation' )
        
        r2 = tf.math.multiply_no_nan( r , r )
        dr = tf.math.multiply_no_nan( a - r2 , r )
#        dr = tf.math.multiply( a , r ) - tf.math.pow( r , 3 )
#        dr = ( a - r**2 ) * r
        dw = b
#        _print( 'dr' , dr )
#        _print( 'dw' , dw )
#        tf.debugging.check_numerics( dr , 'var: dr  ->  hopf activation has nan value' )
#        tf.debugging.check_numerics( dw , 'var: dw  ->  hopf activation has nan value' )
        return tf.stack( [ dr , dw ] )
        
#        dr = tf.math.multiply_no_nan( ( a - r * r ) , r )
#        dr = ( a - r*r ) * r
        
    @tf.function
    def polar_hopf_ODE( self , stp , r , w , a , b ):
        
        t_0 = 0. # stp[0] + 0.
        t_1 = 2*np.pi # stp[0] + 1.
        t_n = [ t_0 , t_1 ]
#        t_2 = ( t_0 + t_1 ) / 2.
#        t_n = [ t_0 , t_2 , t_1 ]
#        t_n = tf.linspace( t_0 , t_1 , num = 31 )
        
        state = tf.stack( [ r , w ] )
        const = { 'a' : a , 'b' : b }
        ode = self.dopri.solve(
            self.polar_hopf_DiffEQ,
            t_0,
            state,
            solution_times = t_n,
            constants = const
        )
        return tf.unstack( ode.states[-1] )
        
    def call( self , stp , r , w , a , b ):
#    def call( self , stp , r_0 , w_0 , a_0 , b_0 ):
        
        r_0 , w_0 , a_0 , b_0 = tf.cast( r , dtype = tf.float64 ) , tf.cast( w , dtype = tf.float64 ) , tf.cast( a , dtype = tf.float64 ) , tf.cast( b , dtype = tf.float64 )
#        r_0 , w_0 , a_0 , b_0 = tf.expand_dims( r_0 , 1 ) , tf.expand_dims( w_0 , 1 ) , tf.expand_dims( a_0 , 1 ) , tf.expand_dims( b_0 , 1 )
        
        r_t , w_t = self.polar_hopf_ODE( stp , r_0 , w_0 , a_0 , b_0 )
        
#        r_t , w_t = tf.squeeze( r_t , 1 ) , tf.squeeze( w_t , 1 )
        r_t , w_t = tf.cast( r_t , dtype = r.dtype ) , tf.cast( w_t , dtype = w.dtype )
        
        return r_t , w_t
    
    def get_config( self ):
        config = {}
        base_config = super( HopfActivTheta , self ).get_config()
        if 'input_spec' not in base_config:
            config['input_spec'] = self.input_spec[0].get_config()
        return dict( list( base_config.items() ) + list( config.items() ) )

@tf.keras.utils.register_keras_serializable( 'activations' )
class HopfActivRadius( tf.keras.layers.Layer ):
    
    def __init__( self , **kwargs ):
        
        self.last_state = None
        
        self.dopri =  tfp.math.ode.DormandPrince(
            rtol = 1.49012e-8,
            atol = 1.49012e-8
        )
        
        super( HopfActivRadius , self ).__init__( **kwargs )
        
        
    def build( self , input_shape ):
        self.built=True
        
        
    @tf.function
    def polar_hopf_DiffEQ( self , t , r , mu ):
        dr = tf.math.multiply( mu , r ) - tf.math.pow( r , 3 )
#        _print( 'dr' , dr )
#        tf.debugging.check_numerics( dr , 'dr for polar hopf activation is nan value' )
        return dr
        
        
    @tf.function
    def polar_hopf_ODE( self , r_0 , mu ):
        
        t_0 = 0.
        t_1 = 2*np.pi/32.
        t_n = [ t_1 ]
#        t_n = tf.linspace( t_0 , 2*np.pi , 12 )
        
        ode_state = r_0
        ode_const = { 'mu' : mu }
        
        ode = self.dopri.solve(
            self.polar_hopf_DiffEQ,
            t_0,
            ode_state,
            solution_times = t_n,
            constants = ode_const
        )
        r_t = ode.states[-1]
        return r_t
        
        
    def call( self , z_0 , alph ):
        r_t = self.polar_hopf_ODE( z_0 , alph )
#        _print( 'r_t' , r_t )
        return r_t
    
    def get_config( self ):
        config = {}
        base_config = super( HopfActivRadius , self ).get_config()
        if 'input_spec' not in base_config:
            config['input_spec'] = self.input_spec[0].get_config()
        return dict( list( base_config.items() ) + list( config.items() ) )



@tf.keras.utils.register_keras_serializable( 'activations' )
class CpxReLU( tf.keras.layers.Layer ):
    
    def __init__( self , **kwargs ):
        
        self.act = tf.keras.activations.relu
        
        super( CpxReLU , self ).__init__( **kwargs )
        
        
    def build( self , input_shape ):
        super( CpxReLU , self ).__init__()
        self.built = True
        
    def call( self , z ):
        re , im = self.act( tf.math.real( z ) ) , self.act( tf.math.imag( z ) )
        return tf.complex( re , im )


@tf.keras.utils.register_keras_serializable( 'activations' )
class modReLU( tf.keras.layers.Layer ):
    
    def __init__( self , **kwargs ):
        
        self.act = tf.keras.activations.relu
        
        super( modReLU , self ).__init__( **kwargs )
        
        
    def build( self , input_shape ):
        super( modReLU , self ).__init__()
        self.built = True
        
    def call( self , z ):
        re , im = self.act( tf.math.real( z ) ) , self.act( tf.math.imag( z ) )
        return tf.complex( re , im )