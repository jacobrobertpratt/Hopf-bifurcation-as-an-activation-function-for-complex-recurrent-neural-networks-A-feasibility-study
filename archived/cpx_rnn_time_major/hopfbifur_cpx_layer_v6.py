import os
import math
import collections
import time 

import numpy as np
import scipy
from scipy.stats import unitary_group

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

''' Local Imports '''
import proj_utils as utils
from activations import HopfBifur
from initializers import Hermitian , HermitianV2 , Unitary , UnitaryV2 , RandomUnitComplex

'''
TODO:
- Add multiple cells
- Copy this code over to the use_RNN_class version to see if performance is different.
- Try and see what the output of the ODESover looks like and if we can use some regularizer
    on that data for the output state. 
- Make stuff into complex next.
'''


def make_unitary_wgt( sz , nme='U' , do_save=False ):
    
    def gen_wgt_matrix(t_sz):
        return unitary_group.rvs(t_sz)
    
    if do_save is True:
        if os.path.exists('rnn_mat_'+nme+'.npy'):
            wgt = np.load('rnn_mat_'+nme+'.npy',allow_pickle=True)
            print('Unitary Weight: Loading ... ' + nme )
            # Check saved shape and replace if not the same.
            if wgt.shape[-1] != sz:
                wgt = gen_wgt_matrix(sz)
                np.save('rnn_mat_'+nme+'.npy',wgt,allow_pickle=True)
                print('Unitary Weight: Creating ... ' + nme )
        else:
            wgt = gen_wgt_matrix(sz)
            np.save('rnn_mat_'+nme+'.npy',wgt,allow_pickle=True)
            print('Unitary Weight: Creating ... ' + nme )
    else:
        wgt = gen_wgt_matrix(sz)
        print('Unitary Weight: Creating ... ' + nme )
    
    wgt = scipy.linalg.logm( wgt )
    wgt = ( wgt + np.conjugate( wgt )[::-1,::-1] ) / 2.
    wgt = wgt * np.power( np.linalg.det( 2 * wgt ) , (1./(sz)) )
    wgt = scipy.linalg.expm( wgt )
    
    return wgt.copy()




def _print( msg , arr , **kwargs):
    
    def p_func( msg , arr , **kwargs):
        if arr.dtype.is_complex:
            tf.print('\n'+msg+':\nreal:\n', tf.math.real(arr),'\nimag:\n',tf.math.imag(arr),'\nshape:',arr.shape,'  dtype:',arr.dtype,'  type:',type(arr),'\n',**kwargs)
        else:
            tf.print('\n'+msg+':\n', arr,'\nshape:', arr.shape,'  dtype:', arr.dtype,'  type:',type(arr),'\n',**kwargs)
    
    if tf.nest.is_nested(arr):
        tf.nest.map_structure(p_func,[msg]*len(arr) , arr)
    else:
        p_func( msg , arr , **kwargs)





@tf.keras.utils.register_keras_serializable('hopfbifur_cpx_layer')
class HopfBifurCpxRNNCell(tf.keras.layers.Layer):
    
    def __init__(
                    self,
                    state_size,
                    activation=None,
                    **kwargs
                  ):
        
        super( HopfBifurCpxRNNCell , self ).__init__( **kwargs )
        
        # Size of input cells
        self.state_size = state_size
        
        # Set default activation or input activation.
        self.activation = HopfBifur( dtype=self.dtype ) if activation is None else activation
        
        
        
    def build( self, input_shape ):
        
        batch_size = input_shape[0]
        input_size = input_shape[-1]
        
        V = make_unitary_wgt( self.state_size , nme='V' )
        eigs = np.random.rand( self.state_size )
        eigs = ( eigs + eigs[::-1] ) / 2.
        
        def get_cos( shape , dtype ):
            C = V @ np.diag( np.cos( eigs ) ) @ np.conjugate( V ).T
            return tf.reshape( tf.cast( C , dtype=self.dtype ) , shape )
            
        self.C = self.add_weight(
            name = 'C_herm_'+self.name,
            shape = ( self.state_size , self.state_size ),
            dtype = self.dtype,
            initializer = get_cos,
            trainable = True
        )
#        evls = tf.linalg.eigvalsh( self.C )
#        evls = tf.expand_dims( evls , 0 )
#        _print( 'Cos evls' , evls.T , summarize=-1 )
        
        def get_sin( shape , dtype ):
            S = V @ np.diag( np.sin( eigs ) ) @ np.conjugate( V ).T
            return tf.reshape( tf.cast( S , dtype=self.dtype ) , shape )
            
        self.S = self.add_weight(
            name = 'S_herm_'+self.name,
            shape = ( self.state_size , self.state_size ),
            dtype = self.dtype,
            initializer = get_sin,
            trainable = True
        )
#        evls = tf.linalg.eigvalsh( self.S )
#        evls = tf.expand_dims( evls , 0 )
#        _print( 'Sin evls' , evls.T , summarize=-1 )
#        exit(0)
        
#        '''
        self.H = self.add_weight(
            name = 'H_herm_'+self.name,
            shape = ( self.state_size , self.state_size ),
            dtype = self.dtype,
            initializer = HermitianV2(),
            trainable = True
        )
        '''
        self.H = self.add_weight(
            name='H_herm_'+self.name,
            shape=( self.state_size , self.state_size ),
            dtype=self.dtype,
            initializer=Hermitian( name='H' , save=True ),
            trainable=True
        )
#        evls = tf.linalg.eigvalsh( self.H )
#        _print('evls',evls,summarize=-1)
#        exit(0)
#       '''
        
        super( HopfBifurCpxRNNCell , self ).build( input_shape )
        
        self.cnt = tf.Variable( [0] , trainable=False , dtype=tf.int32 )
        
        self.built=True
        
        
        
    def call( self , input , state , training=False ):
        
#        tf.print( '\n'+'- '*20+' Cell '+'- '*20+'\nCount:' , self.cnt[0] , '\n' )
        x_in = input[0] if tf.nest.is_nested( input ) else input
        z_in = state[0] if tf.nest.is_nested( state ) else state
#        _print( 'z_in' , z_in )
#        _print( 'x_in' , x_in )
        
        ## z and x are a part of the same inner product space.
        z = tf.linalg.matrix_transpose( z_in )
        x = tf.linalg.matrix_transpose( x_in )
#        _print( 'z' , z )
#        _print( 'x' , x )
        
#        C = self.C
#        S = self.S
        
#        U = C + 1.j*S
#        _print( 'UU' , tf.linalg.matmul( U , U , adjoint_b=True ) )
        
#        x_t = self.herm_unit_map( C , S , x )
        
        x_t = self.hermitian_map( self.H , x )
        z_t = z
#        _print( 'x_t' , x_t )
#        _print( 'z_t' , z_t )
        
        # check for shape compatibility & re-nest if the previous one was nested.
        x_map = tf.reshape( x_t , x_in.shape )
        z_map = tf.reshape( z_t , z_in.shape )
        
        # Make nested if it was origionally
        y_out = [ x_map ] if tf.nest.is_nested( input ) else x_map
        z_out = [ z_map ] if tf.nest.is_nested( state ) else z_map
        
        self.cnt.assign_add([1])
        
        return y_out , z_out
        
        
    @tf.custom_gradient
    def herm_unit_map( self , C , S , x ):
        
        y = tf.linalg.matmul( H , x ) + tf.linalg.matmul( 1.j*S , x )
        y = ( y + tf.math.conj( y )[::-1] ) / 2.
#        _print( 'y' , y , summarize=-1 )
        
        def grad( gL ):
            
#            _print( 'gL' , gL , summarize=-1 )
            
            ## The following directional derivative calculation is based on the paper by Najfeld and Havel.
            ## Cite: I. Najfeld and T. F. Havel, “Derivatives of the Matrix Exponential and Their Computation,”
            ## Advances in Applied Mathematics, vol. 16, no. 3, pp. 321–375, Sep. 1995, doi: 10.1006/aama.1995.1017.
            
            gL2 = tf.cast( gL , dtype=tf.complex128 )
            x2 = tf.cast( x , dtype=tf.complex128 )
            y2 = tf.cast( y , dtype=tf.complex128 )
            H2 = tf.cast( H , dtype=tf.complex128 )
            
            lmda2 , W2 = tf.linalg.eig( H2 )
            adjW2 = tf.linalg.adjoint( W2 )
#            _print( 'lmda2' , lmda2 )
            
#            V = ( tf.linalg.matmul( x2 , gL2 , adjoint_b=True ) + tf.linalg.matmul( gL2 , x2 , adjoint_b=True ) ) / 2. 
            V = tf.linalg.matmul( gL2 , x2 , adjoint_b=True )
            Vinr = tf.math.sqrt( tf.linalg.trace( tf.linalg.matmul( V , V , adjoint_a=True ) ) )
            Vinr = tf.reshape( Vinr , list( Vinr.shape ) + [ 1 , 1 ] )
            V = tf.math.divide_no_nan( V, Vinr )
            
            G = self.gen_exp_eigen_G( lmda2 )
            
            
            
            Vbar = tf.linalg.matmul( tf.linalg.matmul( adjW2 , V ) , W2 )
            GV = tf.math.multiply( Vbar , G )

            Dv_H2 = tf.linalg.matmul( tf.linalg.matmul( W2 , GV ) , adjW2 )
#            Dv_H2 = tf.math.reduce_mean( Dv_H2 , 0 )
#            Dv_H2 = tf.linalg.adjoint( Dv_H2 )
            
            gx2 = tf.linalg.matmul( H2 , gL2 , adjoint_a=True )
            
            gH = tf.reshape( tf.cast( Dv_H2 , dtype=H.dtype ) , H.shape )
            gx = tf.reshape( tf.cast( gx2 , dtype=x.dtype) , x.shape )
            
            return gH , gx
            
        return y , grad
        
        
    @tf.function
    def gen_exp_eigen_G( self , eigs ):
        eigs = tf.reshape( eigs , [eigs.shape[-1]] )
        exps = tf.math.exp( eigs )
        D = tf.linalg.diag( exps )
        eigs = tf.expand_dims( eigs , 0 )
        exps = tf.expand_dims( exps , 0 )
        eigs_diff = ( tf.linalg.adjoint( eigs ) - eigs )
        exps_diff = ( tf.linalg.adjoint( exps ) - exps )
        diff = tf.math.divide_no_nan( exps_diff , eigs_diff )
        G = diff + D
        return tf.reshape( G , [eigs.shape[-1]]*2 )
        
    @tf.custom_gradient
    def hermitian_map( self , H , x ):
        
        y = tf.linalg.matmul( H , x )
        y = ( y + tf.math.conj( y )[::-1] ) / 2.
#        _print( 'y' , y , summarize=-1 )
        
        def grad( gL ):
            
#            _print( 'gL' , gL , summarize=-1 )
            
            ## The following directional derivative calculation is based on the paper by Najfeld and Havel.
            ## Cite: I. Najfeld and T. F. Havel, “Derivatives of the Matrix Exponential and Their Computation,”
            ## Advances in Applied Mathematics, vol. 16, no. 3, pp. 321–375, Sep. 1995, doi: 10.1006/aama.1995.1017.
            
            gL2 = tf.cast( gL , dtype=tf.complex128 )
            x2 = tf.cast( x , dtype=tf.complex128 )
            y2 = tf.cast( y , dtype=tf.complex128 )
            H2 = tf.cast( H , dtype=tf.complex128 )
            
            lmda2 , W2 = tf.linalg.eig( H2 )
            adjW2 = tf.linalg.adjoint( W2 )
#            _print( 'lmda2' , lmda2 )
            
#            V = ( tf.linalg.matmul( x2 , gL2 , adjoint_b=True ) + tf.linalg.matmul( gL2 , x2 , adjoint_b=True ) ) / 2. 
            V = tf.linalg.matmul( gL2 , x2 , adjoint_b=True )
            Vinr = tf.math.sqrt( tf.linalg.trace( tf.linalg.matmul( V , V , adjoint_a=True ) ) )
            Vinr = tf.reshape( Vinr , list( Vinr.shape ) + [ 1 , 1 ] )
            V = tf.math.divide_no_nan( V, Vinr )
            
            G = self.gen_exp_eigen_G( lmda2 )
            
            Vbar = tf.linalg.matmul( tf.linalg.matmul( adjW2 , V ) , W2 )
            GV = tf.math.multiply( Vbar , G )

            Dv_H2 = tf.linalg.matmul( tf.linalg.matmul( W2 , GV ) , adjW2 )
#            Dv_H2 = tf.math.reduce_mean( Dv_H2 , 0 )
#            Dv_H2 = tf.linalg.adjoint( Dv_H2 )
            
            gx2 = tf.linalg.matmul( H2 , gL2 , adjoint_a=True )
            
            gH = tf.reshape( tf.cast( Dv_H2 , dtype=H.dtype ) , H.shape )
            gx = tf.reshape( tf.cast( gx2 , dtype=x.dtype) , x.shape )
            
            return gH , gx
            
        return y , grad
        
        
    @tf.function
    def gen_cos_eigen_G( self , eigs ):
        eigs = tf.reshape( eigs , [eigs.shape[-1]] )
        feigs = tf.math.cos( eigs )
        dfeigs = -tf.math.sin( eigs)
        D = tf.linalg.diag( dfeigs )
        eigs = tf.expand_dims( eigs , 0 )
        feigs = tf.expand_dims( feigs , 0 )
        feigs_diff = ( tf.linalg.adjoint( feigs ) - feigs )
        eigs_diff = ( tf.linalg.adjoint( eigs ) - eigs )
        diff = tf.math.divide_no_nan( feigs_diff , eigs_diff )
        G = diff + D
        return tf.reshape( G , [eigs.shape[-1]]*2 )

    @tf.custom_gradient
    def herm_cos_map( self , H , x ):
        
        y = tf.linalg.matmul( H , x )
#        _print( 'A) y' , y , summarize=-1 )
        
        y = ( y + tf.math.conj( y )[::-1] ) / 2.
#        _print( 'B) y' , y , summarize=-1 )
        
        def grad( gL ):
            
            ## The following directional derivative calculation is based on the paper by Najfeld and Havel.
            ## Cite: I. Najfeld and T. F. Havel, “Derivatives of the Matrix Exponential and Their Computation,”
            ## Advances in Applied Mathematics, vol. 16, no. 3, pp. 321–375, Sep. 1995, doi: 10.1006/aama.1995.1017.
            
            gL2 = tf.cast( gL , dtype=tf.complex128 )
            x2 = tf.cast( x , dtype=tf.complex128 )
            y2 = tf.cast( y , dtype=tf.complex128 )
            H2 = tf.cast( H , dtype=tf.complex128 )
            
            lmda2 , W2 = tf.linalg.eigh( H2 )
            
#            plmda2 = tf.linalg.matrix_transpose( tf.expand_dims( lmda2 , 0 ) )
#            _print( 'Cos lmda2' , plmda2 , summarize=-1 )
            
            D2 = tf.linalg.diag( lmda2 )
            adjW2 = tf.linalg.adjoint( W2 )

#            V = ( tf.linalg.matmul( x2 , gL2 , adjoint_b=True ) + tf.linalg.matmul( gL2 , x2 , adjoint_b=True ) ) / 2. 
            V = tf.linalg.matmul( gL2 , x2 , adjoint_b=True )
            Vinr = tf.math.sqrt( tf.linalg.trace( tf.linalg.matmul( V , V , adjoint_a=True ) ) )
            Vinr = tf.reshape( Vinr , list( Vinr.shape ) + [ 1 , 1 ] )
            V = tf.math.divide_no_nan( V , Vinr )
            
            G = self.gen_cos_eigen_G( lmda2 )
            
            Vbar = tf.linalg.matmul( tf.linalg.matmul( adjW2 , V ) , W2 )
            GV = tf.math.multiply( Vbar , G )

            Dv_H2 = tf.linalg.matmul( tf.linalg.matmul( W2 , GV ) , adjW2 )
#            Dv_H2 = tf.math.reduce_mean( Dv_H2 , 0 )
#            Dv_H2 = tf.linalg.adjoint( Dv_H2 )
            
            gx2 = tf.linalg.matmul( H2 , gL2 , adjoint_a=True )
            
            gH = tf.reshape( tf.cast( Dv_H2 , dtype=H.dtype ) , H.shape )
            gx = tf.reshape( tf.cast( gx2 , dtype=x.dtype) , x.shape )
            
            return gH , gx
            
        return y , grad
        
        
    @tf.function
    def gen_sin_eigen_G( self , eigs ):
        eigs = tf.reshape( eigs , [eigs.shape[-1]] )
        feigs = tf.math.sin( eigs )
        dfeigs = tf.math.cos( eigs)
        D = tf.linalg.diag( dfeigs )
        eigs = tf.expand_dims( eigs , 0 )
        feigs = tf.expand_dims( feigs , 0 )
        feigs_diff = ( tf.linalg.adjoint( feigs ) - feigs )
        eigs_diff = ( tf.linalg.adjoint( eigs ) - eigs )
        diff = tf.math.divide_no_nan( feigs_diff , eigs_diff )
        G = diff + D
        return tf.reshape( G , [eigs.shape[-1]]*2 )
        
    @tf.custom_gradient
    def herm_sin_map( self , H , x ):
        
        y = tf.linalg.matmul( 1.j*H , x )
#        _print( 'A) y' , y , summarize=-1 )
        
        y = ( y + tf.math.conj( y )[::-1] ) / 2.
#        _print( 'B) y' , y , summarize=-1 )
        
        def grad( gL ):
            
            ## The following directional derivative calculation is based on the paper by Najfeld and Havel.
            ## Cite: I. Najfeld and T. F. Havel, “Derivatives of the Matrix Exponential and Their Computation,”
            ## Advances in Applied Mathematics, vol. 16, no. 3, pp. 321–375, Sep. 1995, doi: 10.1006/aama.1995.1017.
            
            gL2 = tf.cast( gL , dtype=tf.complex128 )
            x2 = tf.cast( x , dtype=tf.complex128 )
            y2 = tf.cast( y , dtype=tf.complex128 )
            H2 = tf.cast( H , dtype=tf.complex128 )
            
            lmda2 , W2 = tf.linalg.eigh( H2 )
#            plmda2 = tf.linalg.matrix_transpose( tf.expand_dims( lmda2 , 0 ) )
#            _print( 'Sin lmda2' , plmda2 , summarize=-1 )
            
            D2 = tf.linalg.diag( lmda2 )
            adjW2 = tf.linalg.adjoint( W2 )
            
#            V = ( tf.linalg.matmul( x2 , gL2 , adjoint_b=True ) + tf.linalg.matmul( gL2 , x2 , adjoint_b=True ) ) / 2. 
            V = tf.linalg.matmul( gL2 , x2 , adjoint_b=True )
            Vinr = tf.math.sqrt( tf.linalg.trace( tf.linalg.matmul( V , V , adjoint_a=True ) ) )
            Vinr = tf.reshape( Vinr , list( Vinr.shape ) + [ 1 , 1 ] )
            V = tf.math.divide_no_nan( V, Vinr )
            V = -1.j*V
            
            G = self.gen_sin_eigen_G( lmda2 )
            
            Vbar = tf.linalg.matmul( tf.linalg.matmul( adjW2 , V ) , W2 )
            GV = tf.math.multiply( Vbar , G )

            Dv_H2 = tf.linalg.matmul( tf.linalg.matmul( W2 , GV ) , adjW2 )
#            Dv_H2 = tf.math.reduce_mean( Dv_H2 , 0 )
#            Dv_H2 = tf.linalg.adjoint( Dv_H2 )
            
            gx2 = tf.linalg.matmul( H2 , gL2 , adjoint_a=True )
            
            gH = tf.reshape( tf.cast( Dv_H2 , dtype=H.dtype ) , H.shape )
            gx = tf.reshape( tf.cast( gx2 , dtype=x.dtype) , x.shape )
            
            return gH , gx
            
        return y , grad
        
        
    @tf.custom_gradient
    def map_unitary_kernel( self , U , x ):
        
        y = tf.linalg.matmul( U , x )
        
        def grad( gL ):
            
            ## From projUNN  Paper.
            ## Following weight updates by projecting a 'closest' unitary matrix and implementing a different optimization function.
            ## B. Kiani, R. Balestriero, Y. LeCun, and S. Lloyd,
            ## “projUNN: efficient method for training deep networks with unitary matrices.”
            ## arXiv, Oct. 13, 2022. Accessed: Jun. 27, 2023. [Online]. Available: http://arxiv.org/abs/2203.05483
            
            _print( 'gL' , gL )
            
            gL2 = tf.cast( gL , dtype=tf.complex128 )
            U2 = tf.cast( U , dtype=tf.complex128 )
            x2 = tf.cast( x , dtype=tf.complex128 )
            y2 = tf.cast( y , dtype=tf.complex128 )
            
#            _print( 'gL2' , gL2 )
#            _print( 'U2' , U2 )
#            _print( 'x2' , x2 )
            J = -tf.linalg.matmul( gL2 , x2 , adjoint_b=True )
            inrJ = tf.math.sqrt( tf.linalg.trace( tf.linalg.matmul( J , J , adjoint_a=True ) ) )
            inrJ = tf.reshape( inrJ , list(inrJ.shape)+[1,1] )
            J = tf.math.divide_no_nan( J , inrJ )
            
            gU2 = tf.linalg.matmul(
                U2,
                ( J - tf.linalg.matmul( tf.linalg.matmul( U2 , J , adjoint_b=True ) , U2 ) ) / 2.,
                adjoint_a=True
            )
            gU2 = tf.math.reduce_mean( gU2 , 0 )
            
            gx2 = tf.zeros_like( x2 )
            
            gU = tf.cast( gU2 , dtype = U.dtype )
            gx = tf.cast( gx2 , dtype = x.dtype )
            
            return gU , gx
            
        return y , grad
        
        
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
                   output_size=None,
                   activation=None,
                   **kwargs
                  ):
        
        super(HopfBifurCpxRNNLayer, self).__init__(**kwargs)
        
        # The state size of each cells, if None then defaults to square-root of input dimensions and cast as an integer.
        self.state_size = state_size
        
        self.output_size = output_size
        
        # To hold the cell(s) for the model.
        self.cell = None
        
        # Set the activation function
        self.activation = activation
        
        
        
    def build(self, input_shape):
        
        self.batch_size = input_shape[0]
        self.time_size = input_shape[-2]
        self.input_size = input_shape[-1]
        
        
        if self.output_size is None: self.output_size = self.input_size
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
                
#        assert self.output_size > 2 , 'output size must be greater than 2 due to initialization schemea.'
        
        state_init = np.random.rand( self.state_size )
#        state_init = ( np.arange( self.state_size ) + 1. ) / ( self.state_size + 2 )
        state_init = ( state_init - state_init[::-1] ) / 2.
        state_init = np.roll( state_init , self.state_size // 2 , 0 )
        state_init = np.exp( -2.j*np.pi * state_init )
        state_init = tf.reshape( tf.cast( state_init , dtype=self.dtype ) , [ 1 , self.state_size ] )
        self.state = tf.Variable(
            initial_value = state_init,
            shape = ( 1 , self.state_size ),
            dtype = self.dtype,
            trainable = False,
            name = 'state_'+self.name
        )
#        state = tf.math.angle( tf.linalg.matrix_transpose( self.state ) )
#        _print( 'state' , state , summarize=-1 )
#        exit(0)
        
        rnginit = tf.linspace( 0. , 2*np.pi + 0. , self.input_size )
#        rnginit = np.arange( self.input_size )
        rnginit = tf.cast( tf.reshape( [ rnginit ] , [ 1 , self.input_size ] ) , dtype = self.dtype )
        self.inrng = tf.constant( rnginit , dtype = self.dtype , name = 'in_rng_' + self.name )
        
        rnginit = tf.linspace( 0. , 2*np.pi + 0. , self.output_size )
#        rnginit = np.arange( self.output_size )
        rnginit = tf.cast( tf.reshape( [ rnginit ] , [ 1 , self.output_size ] ), dtype = self.dtype )
        self.outrng = tf.constant( rnginit , dtype = self.dtype , name = 'out_rng_' + self.name )
        
        self.sqrt_state_size = tf.constant(
            tf.math.sqrt( tf.cast( self.state_size , dtype = self.dtype ) ),
            dtype = self.dtype
        )
        
        super( HopfBifurCpxRNNLayer , self ).build( input_shape )
        
        self.cnt = tf.Variable( [0] , trainable=False , dtype=tf.int32 )
        
        self.built = True
        
        
    def make_base( self , z , range ):
        ang = tf.linalg.matrix_transpose( tf.math.angle( z ) )
        rng = range + tf.math.reduce_mean( tf.math.abs( ang ) )
        base = tf.math.pow( tf.math.exp( 1.j * ang ) , rng )
        base , _ = tf.linalg.normalize( base , ord=2 , axis=-1)
        return base
        
        
    def call(self, input, training=False ):
        
#        tf.print('\n'*8+'=='*25+' Layer '+'=='*25+'\nCount:',self.cnt[0],'\n')
#        _print( 'L input' , input )
        
        _state = self.state
        _input = tf.cast( input , dtype=self.dtype )
#        _print( 'L _input' , _input )
        
        # Map the input to the current state
        _ibase = self.make_base( _state , self.inrng )
        _input = tf.linalg.matmul( _ibase , _input , transpose_b=True )
        _input = tf.linalg.matrix_transpose( _input )
#        _print( '_input' , _input )
        
#        '''
        rnn_return = self.rnn_call(
            self.cell,
            _input,
            _state,
            time_major=True,            # Provides our 'cell' layer with single input tensors instead of entire batch.
            return_all_outputs=True,    # If True -> seq_out has batch sized output; else, single output
            training = training
        )
        last_out , cell_out , new_state = rnn_return
        '''
        new_state = _state
        cell_out = _input
#        '''
        
#        tf.print('\n'+' *'*15+' RNN Returned '+'* '*15+'\n')
#        _print( 'L last_out', last_out)
#        _print( 'L cell_out', cell_out)
#        _print( 'L new_state', new_state )
#        _print( 'L self.state', self.states )
#        _print( 'L self.manif', self.manif )
        
        tf.debugging.check_numerics( tf.math.real( last_out ) , 'a last_out element is nan' )
        tf.debugging.check_numerics( tf.math.real( cell_out ) , 'a cell_out element is nan' )
        tf.debugging.check_numerics( tf.math.real( new_state ) , 'a new_state element is nan' )
        
        new_state = tf.reshape( new_state , _state.shape )
        cell_out = tf.reshape( cell_out , _input.shape )
#        _print( 'L cell_out', cell_out)
#        _print( 'L new_state', new_state )
        
        self.state.assign( new_state )
        
#        new_state = tf.math.conj( new_state )
        state_base = self.make_base( new_state , self.outrng )
#        _print( 'state_base' , state_base )
        
        output_map = tf.linalg.matmul( cell_out , state_base )
#        _print( 'output_map' , output_map )
        
        max_img_val = tf.math.reduce_max( tf.math.imag( output_map ) )
#        tf.debugging.assert_less( max_img_val , 1.e-5, 'imaginary values went above 1.e-5 issue with recurrance.' )
#        tf.print('L img:', max_img_val )
        
        output = tf.cast( tf.math.real( output_map ) , input.dtype )
#        _print( 'output' , output )
        
        self.cnt.assign_add([1])
        
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




class HopfCell(tf.keras.layers.Layer):
    
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