import os
import math

import numpy as np
import scipy
from scipy.stats import unitary_group

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from proj_utils import _cpx_ptrace, _cpx_pdet, _cpx_eigvals


def generate_unitary_matrix( dim , trace_zero=True):
    unt = unitary_group.rvs( dim )
#    unt = scipy.linalg.logm(unt)
#    unt = (unt + np.conjugate(unt)[::-1,::-1])/2.
#    unt = unt / np.sqrt(np.trace(np.conjugate(unt).T @ unt))
#    unt = unt / np.power(np.linalg.det(unt)/(2*np.pi),(1./(n)))
#    _cpx_eigvals( 'unt' , unt )
#    _cpx_pdet( 'unt' , unt )
#    _cpx_ptrace( 'unt' , unt )
#    exit(0)
#    unt = scipy.linalg.expm(unt)
    return unt


def weight_saver( gen_func , name = 'A' , save = False , **params ):
    _dim = params['dim'] if 'dim' in params else []
    if save is True:
        fn = name + '_wgt.npy'
        if os.path.exists( fn ):
#            print('Loading Weight ... ' + name )
            wgt = np.load( fn , allow_pickle=True )
        else:
            wgt = gen_func( **params )
            np.save( fn , wgt , allow_pickle=True )
#            print('Creating & Saving Weight ... ' + name )
    else:
        wgt = gen_func( **params )
#        print('Creating Weight ... ' + name )
    
    return wgt


class GlorotUnif( tf.keras.initializers.Initializer ):
    
    def __init__(self , name = 'GU' , save = False , **kwargs ):
        
        self.save = save
        
        self.name = name
        
        super( GlorotUnif , self).__init__( **kwargs )
        
    def __call__( self , shape , dtype ):
        
        def _gen( shape , dtype ):
            return tf.keras.initializers.GlorotUniform()( shape , tf.float32 ).numpy()
        
        wgt_name = self.name + '_glorot_unif'
        
        wgt = weight_saver( _gen , name = wgt_name , save = self.save , **{ 'shape' : shape , 'dtype' : dtype } )
        
        # Convert to ndarray
        if not isinstance( wgt , np.ndarray ): wgt = np.asarray( wgt )
        
        try:
            return tf.reshape( tf.cast( wgt.copy() , dtype = dtype ) , shape )
        except:
            wgt2 = weight_saver( _gen , name = wgt_name + '2' , save = False , **{ 'shape' : shape , 'dtype' : dtype } )
            return tf.reshape( tf.cast( wgt2.copy() , dtype = dtype ) , shape )


class GlorotNorm( tf.keras.initializers.Initializer ):
    
    def __init__(self , name = 'GN' , save = False , **kwargs ):
        
        self.save = save
        
        self.name = name
        
        super( GlorotNorm , self).__init__( **kwargs )
        
    def __call__( self , shape , dtype ):
        
        def _gen( shape , dtype ):
            return tf.keras.initializers.GlorotNormal()( shape , tf.float32 ).numpy()
        
        wgt_name = self.name + '_glorot_norm'
        
        wgt = weight_saver( _gen , name = wgt_name , save = self.save , **{ 'shape' : shape , 'dtype' : dtype } )
        
        # Convert to ndarray
        if not isinstance( wgt , np.ndarray ): wgt = np.asarray( wgt )
        
        try:
            return tf.reshape( tf.cast( wgt.copy() , dtype = dtype ) , shape )
        except:
            wgt2 = weight_saver( _gen , name = wgt_name + '2' , save = False , **{ 'shape' : shape , 'dtype' : dtype } )
            return tf.reshape( tf.cast( wgt2.copy() , dtype = dtype ) , shape )

class Orthogonal( tf.keras.initializers.Initializer ):
    
    def __init__(self , name = 'A' , save = False , **kwargs ):
        
        self.save = save
        
        self.name = name
        
        super( Orthogonal , self).__init__( **kwargs )
        
    def __call__( self , shape , dtype ):
        
        def _gen( shape , dtype ):
            return tf.keras.initializers.Orthogonal()( shape , tf.float32 ).numpy()
        
        wgt_name = self.name + '_orth'
        
        wgt = weight_saver( _gen , name = wgt_name , save = self.save , **{ 'shape' : shape , 'dtype' : dtype } )
        
        # Convert to ndarray
        if not isinstance( wgt , np.ndarray ): wgt = np.asarray( wgt )
        
        try:
            return tf.reshape( tf.cast( wgt.copy() , dtype = dtype ) , shape )
        except:
            wgt2 = weight_saver( _gen , name = wgt_name + '2' , save = False , **{ 'shape' : shape , 'dtype' : dtype } )
            return tf.reshape( tf.cast( wgt2.copy() , dtype = dtype ) , shape )
            
            
            
class Hermitian(tf.keras.initializers.Initializer):
    
    def __init__(self , name = 'A' , save = False , **kwargs ):
        
        self.save = save
        
        self.name = name
        
        super( Hermitian , self).__init__( **kwargs )
        
    def __call__( self , shape , dtype ):
        
        def _gen( shape , dtype ):
            R = np.random.rand( shape[0] , shape[1] ) + 1.j*np.random.rand( shape[0] , shape[1] )
            H = ( R @ np.conjugate( R ).T )
            H = H / np.trace( np.conjugate( R ).T @ R )
            return H
        
        wgt_name = self.name + '_herm'
        
        wgt = weight_saver( _gen , name = wgt_name , save=self.save , **{ 'shape' : shape , 'dtype' : dtype } )
        
        if not isinstance( wgt , np.ndarray ): wgt = np.asarray( wgt )
        
        return tf.reshape( tf.cast( wgt.copy() , dtype=dtype ) , shape )
        
        
        
class Unitary( tf.keras.initializers.Initializer ):
    
    def __init__( self , name = 'U' , save = False , **kwargs ):
        
        self.save = save
        
        self.name = name
        
        super( Unitary , self ).__init__( **kwargs )
        
    def __call__( self , shape , dtype = None ):
        
        size = shape[-1]
        
        def _gen( sz = 4 ):
            return unitary_group.rvs( sz )
        
        wgt_name = self.name + '_unit'
        
        wgt = weight_saver( _gen , name = wgt_name , save = self.save , **{ 'sz' : size } )
        
        if not isinstance( wgt , np.ndarray ): wgt = np.asarray( wgt )
        
        return tf.reshape( tf.cast( wgt.copy() , dtype = dtype ) , shape )
        
        
        
        
class Eye( tf.keras.initializers.Initializer ):
    
    def __init__( self , name = 'I' , save = False , **kwargs ):
        
        self.save = save
        
        self.name = name
        
        super( Eye , self ).__init__( **kwargs )
        
    def __call__( self , shape , dtype = None ):
        
        size = shape[-1]
        
        def _gen( sz = 4 ):
            return np.eye( sz , dtype = np.float32 )
        
        wgt_name = self.name + '_eye'
        
        wgt = weight_saver( _gen , name = wgt_name , save = self.save , **{ 'sz' : size } )
        
        if not isinstance( wgt , np.ndarray ): wgt = np.asarray( wgt )
        
        return tf.reshape( tf.cast( wgt.copy() , dtype = dtype ) , shape )
        
        
# Creats a (size x size) unitary matrix ignores input shapes except to check batch size.
class SkewHermitian( tf.keras.initializers.Initializer ):
    
    def __init__( self , name = 'S' , save = False , conjsym = False , bkwrd = False , **kwargs ):
        
        self.save = save
        self.bkwrd = bkwrd
        self.name = name
        self.conjsym = conjsym
        
        super( SkewHermitian , self ).__init__( **kwargs )
        
    def __call__( self , shape , dtype = None ):
        
        size = shape[-1]
        
        def gen_wgt( dim = 4 , conjsym = False ):
            unt = unitary_group.rvs( dim )
            eigs = np.random.rand( dim )
            if conjsym:
                unt = scipy.linalg.logm( unt )
                unt = (unt + np.conjugate( unt )[::-1,::-1]) / 2.
                unt = scipy.linalg.expm( unt )
                eigs = ( eigs + eigs[::-1] ) / 2.
            return [ unt @ np.diag( eigs ) @ np.conjugate( unt ).T ]
        
        wgt_name = self.name + '_skew'
        
        wgt = weight_saver( gen_wgt , name = wgt_name , save=self.save , **{ 'dim' : size , 'conjsym' : self.conjsym } )
        
        if not isinstance( wgt , np.ndarray ): wgt = np.asarray( wgt )
        
        wgt = -1.j*wgt if self.bkwrd is True else 1.j*wgt
        
        return tf.reshape( tf.cast( wgt.copy() , dtype=dtype ) , shape )
        
        
class RandConjSymmVects(tf.keras.initializers.Initializer):
        
    def __call__(self, shape, dtype=None):
        
        batch_dim = shape[0]
        
        input_dim = shape[-1]
        
        def gen_random_conjsymm_vector(sz):
            vec = np.random.rand(sz)
            vec = vec - np.mean(vec)
            vec /= np.mean(vec)
            vec = np.exp(-2.j*np.pi*vec)
            vec = vec * np.conjugate(vec[::-1])
            return vec
        wgt = gen_random_conjsymm_vector( input_dim )
        return tf.cast(wgt.copy(), dtype=dtype)



class RandStandardNormal( tf.keras.initializers.Initializer ):
    
    def __init__( self , **kwargs ):
        
        self.rndnorm_init = tf.keras.initializers.RandomNormal( mean=0. , stddev=1. )
        
        super( RandStandardNormal , self ).__init__( **kwargs )
        
    
    def __call__( self , shape , dtype = None ):
        
        shape = list( shape )[-2::]
        
        wgt = self.rndnorm_init( ( shape[0] , shape[1] ) , dtype = dtype )
        
        return tf.reshape( tf.cast( wgt.copy() , dtype = dtype ) , [ shape[0] , shape[1] ] )



# Initializer that you can set a weight, then the add_weight function in 
#   Keras will actually use it. (kind of a work around because add_weight doesn't let you just add a weight)
class SetterGetter(tf.keras.initializers.Initializer):
    
    def __init__(self, wgt):
        self.wgt = wgt
        
    def __call__(self, shape, dtype=None):
        tf.ensure_shape(self.wgt, shape)
        self.wgt = tf.cast(self.wgt, dtype=dtype)
        return self.wgt




