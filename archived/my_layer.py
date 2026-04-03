import os
import sys
import math
from datetime import datetime
from inspect import currentframe, getframeinfo

import numpy as np
import scipy
from scipy.stats import unitary_group


import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


''' 

Also get Citation for NeuralODE paper. Chen et al., 2018
Citation for tensorflow Probability:
J. V. Dillon et al., “TensorFlow Distributions.” arXiv, Nov. 28, 2017. doi: 10.48550/arXiv.1711.10604.      '''
import tensorflow_probability as tfp

''' Local Imports'''
from utils import _print, _print_matrix

def _cpx_ptrace(msg='',mat=None):
    if mat is None:
        return
    trc = tf.linalg.trace(mat)
    tf.print('\n'+str(msg) + ' Trace:',tf.math.real(trc),tf.math.imag(trc),'\n')

def _cpx_pdet(msg='',mat=None):
    if mat is None:
        return
    det = tf.linalg.det(mat)
    tf.print('\n'+str(msg) + ' Determinant:    ',tf.math.real(det),tf.math.imag(det),'\n')

def _cpx_eigvals(msg='',mat=None):
    if mat is None:
        return
    tf.print(msg + ' Eigen Values:')
    evls = tf.linalg.eigvals(mat)
    for j in range(int(mat.shape[-1])):
        tf.print(str(j+1)+')', tf.math.real(evls[j]), tf.math.imag(evls[j]))
    tf.print()

''' ---- LOSS FUNCTION ---- '''
class MyLoss(tf.keras.losses.Loss):

    ''' '''
    def __init_(self, reduction=tf.keras.losses.Reduction.NONE, name=None, **kwargs):
        super().__init_(reduction=reduction, name=name, **kwargs)

    ''' '''
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


''' RNN CELL '''
class MyRNNCell(tf.keras.layers.Layer):
    
    ''' MyRNNCell ''' 
    def __init__(self,**kwargs):
        
        super(MyRNNCell, self).__init__(**kwargs)
    
    
    ''' MyRNNCell '''
    def build(self, input_shape):
        
        super(MyRNNCell,self).build(input_shape=input_shape)

        in_sz = input_shape[-1]
        self.in_sz = in_sz
        
        root = np.exp(-2j*np.pi*np.arange(in_sz)/in_sz).reshape(-1,1)
        self.root = tf.constant(root,dtype=tf.complex128)
        
        # Normal dft
        init_dft = root**np.arange(in_sz)
        self.dft = tf.constant(init_dft, dtype=tf.complex128)
        
        # Unitary dft
        udft = init_dft / np.sqrt(in_sz)
        self.udft = tf.constant(udft,dtype=tf.complex128)
        
        self.zeros  = tf.constant(np.zeros(shape=(in_sz,1),dtype=np.complex128),dtype=tf.complex128)
        self.ones  = tf.constant(np.ones(shape=(in_sz,1),dtype=np.complex128),dtype=tf.complex128)
        
        self.eye = tf.constant(np.eye(in_sz),dtype=tf.complex128)
        
        self.pow = tf.constant([np.arange(in_sz)],dtype=tf.float64)
        
        adj = np.ones(shape=(in_sz,1),dtype=np.complex128)
        adj[0] = (0.+0.j)
        self.adj = tf.constant(adj,dtype=tf.complex128)
        
        '''
        if os.path.exists('rnn_mat_u.npy'):
            U = np.load('rnn_mat_u.npy',allow_pickle=True)
            # Check saved shape and replace if not the same.
            if U.shape[-1] != mini_sz:
                U = unitary_group.rvs(mini_sz)
                U = scipy.linalg.logm(U)
                np.save('rnn_mat_u.npy',U,allow_pickle=True)
        else:
            U = unitary_group.rvs(mini_sz)
            U = scipy.linalg.logm(U)
            np.save('rnn_mat_u.npy',U,allow_pickle=True)
        '''
        
        def make_unitary_wgt(sz,nme='U',do_save=True):
            
            def gen_wgt_matrix(t_sz):
                wgt = unitary_group.rvs(t_sz)
                wgt = tf.linalg.logm(wgt)
                wgt = (wgt + tf.math.conj(wgt)[::-1,::-1])/2.
#                wgt = wgt / tf.math.pow(tf.linalg.det(wgt),(1/in_sz))
                wgt = tf.linalg.expm(wgt)
#                _print('wgt',wgt)
#                Ichk = wgt @ tf.math.conj(wgt).T
#                _print('Ichk',Ichk)
#                _cpx_eigvals('wgt',wgt)
#                _cpx_pdet('wgt',wgt)
#                _cpx_ptrace('wgt',wgt)
#                exit(0)
                return wgt
            
            if do_save is True:
                if os.path.exists('rnn_mat_'+nme+'.npy'):
                    wgt = np.load('rnn_mat_'+nme+'.npy',allow_pickle=True)
                    # Check saved shape and replace if not the same.
                    if wgt.shape[-1] != sz:
                        wgt = gen_wgt_matrix(sz)
                        np.save('rnn_mat_'+nme+'.npy',wgt,allow_pickle=True)
                        tf.print('Generating new weight matrix for '+nme)
                else:
                    wgt = gen_wgt_matrix(sz)
                    np.save('rnn_mat_'+nme+'.npy',wgt,allow_pickle=True)
                    tf.print('Generating new weight matrix for '+nme)
            else:
                wgt = gen_wgt_matrix(sz)
            
            return wgt
        
        # Set to True if we want to reuse the initialization matrices for common training
        _save = False
        _train = False
        _sz = in_sz-1
        
        U = make_unitary_wgt(_sz,nme='U',do_save=_save)
        self.U = tf.Variable(   U,
                                name='U',
                                dtype=tf.complex128,
                                trainable=_train
                                )
        
        W = make_unitary_wgt(_sz,nme='W',do_save=_save)
        self.W = tf.Variable(   W,
                                name='W',
                                dtype=tf.complex128,
                                trainable=_train
                                )
        
        if _save is True:
            if os.path.exists('rnn_vec_z.npy'):
                z = np.load('rnn_vec_z.npy',allow_pickle=True)
                # Check saved shape and replace if not the same.
                if z.shape[0] != _sz:
#                    z = np.random.rand(_sz).reshape(-1,1)
                    z = np.arange(_sz).reshape(-1,1)/_sz
                    np.save('rnn_vec_z.npy',z,allow_pickle=True)
                    tf.print('Generating new state vector for z')
            else:
#                z = np.random.rand(_sz).reshape(-1,1)
                z = np.arange(_sz).reshape(-1,1)/_sz
                np.save('rnn_vec_z.npy',z,allow_pickle=True)
                tf.print('Generating new state vector for z')
        else:
#            z = np.random.rand(_sz).reshape(-1,1)
            z = np.arange(_sz).reshape(-1,1)
        
        r = np.random.rand(_sz).reshape(-1,1)
        r = (r + r[::-1])/2.
        z = (z - z[::-1])/64.
        z = r*np.exp(1.j*z)
        self.z = tf.Variable(
                              initial_value=z,
                              name='z',
                              dtype=tf.complex128,
                              trainable=False
                             )
        
        self.counter = tf.Variable([1],dtype=tf.float64,trainable=False)
        
        self.built = True


    ''' MyRNN '''
    @tf.function
    def calc_cpx_Hopf(self,z,a,b):
        
#        tf.print('\n-----------------------------------------------------------------------------\n')
        
        re_z, im_z = tf.unstack(z)
        re_a, im_a = tf.unstack(a)
        re_b, im_b = tf.unstack(b)
        
        z = tf.complex(re_z,im_z)
        a = tf.complex(re_a,im_a)
        b = tf.complex(re_b,im_b)
        
        zz = tf.math.conj(z)*z
        
        bzz = b*zz
        
        w = a - bzz
        
        dz = w*z
        
        return tf.stack([tf.math.real(dz),tf.math.imag(dz)])
        
        
    ''' MyRNN '''
    @tf.function
    def hopf_Diff_EQ(self,t,state,re_a,im_a,re_b,im_b):
        re_z, im_z = tf.unstack(state)
        z_stk = tf.stack([re_z,im_z])
        a_stk = tf.stack([re_a,im_a])
        b_stk = tf.stack([re_b,im_b])
        re_dz, im_dz = tf.unstack( self.calc_cpx_Hopf( z_stk , a_stk , b_stk ) )
        return tf.stack([re_dz,im_dz])
        
        
    ''' MyRNN '''
    @tf.function
    def hopf_ODE(self,z,a,b):
        
        # Setup time intervals #
        t_0 = 0.
        t_1 = tf.math.sqrt(0.+self.in_sz)
        t_r = tf.linspace(t_0,t_1,num=int(t_1)*4)
        
        re_z, im_z = tf.math.real(z), tf.math.imag(z)
        re_a, im_a = tf.math.real(a), tf.math.imag(a)
        re_b, im_b = tf.math.real(b), tf.math.imag(b)
        
        ode_state = tf.stack([re_z,im_z])
        ode_const = {'re_a':re_a,'im_a':im_a,'re_b':re_b,'im_b':im_b}
        DormPrnc = tfp.math.ode.DormandPrince()
        ode = DormPrnc.solve(
                              self.hopf_Diff_EQ,
                              t_0,
                              ode_state,
                              solution_times=t_r,
                              constants=ode_const
                             )
        ode_t = ode.states[-1]
        re_zt, im_zt = tf.unstack(ode_t)
        
        zt = tf.complex(re_zt,im_zt)
        
        return zt


    ''' MyRNNCell '''
    @tf.custom_gradient
    def map_unitary(self,M,v):
        
        Mv = M @ v
        
        def grad(gL):
            
#            tf.print('\n --------------------------- back prop --------------------------- \n',self.counter[0],')\n')
            
#            gL = (gL + tf.math.conj(gL)[::-1])/2.   # Cleans up the gradient so that it's conj-symm
            
            # Combined with a modified Optimizer, successfully learns unitary.
            J = gL*tf.math.conj(v).T
            TM = tf.math.conj(M).T @ ((J - M @ tf.math.conj(J).T @ M)/2.)
#            _print('TM',TM)
            
            gM = TM
            gv = tf.zeros_like(v)
            
            return gM, gv
        
#        _print('Mv',Mv)
        
        # Cleanup remainders for stability.
        Mv = (Mv + tf.math.conj(Mv)[::-1])/2.
        
        return Mv, grad
    
    @tf.custom_gradient
    def move_unitary_mat(self,U,z,x):
        
        def grad(gL):
            
            return gU, gz, gx
        
        return newU
        
    ''' MyRNNCell '''
    def call(self,input,training=False):
        
#        tf.print('\n-----------------------------------------------------------------------------\n',self.counter[0],')\n')
        
        x = input
        z = self.z
        
        U = self.U
        W = self.W
        
        _print('x',x)
        _print('z',z)
#        _print('U',U)
        
        x0 = x[0]
#        _print('r_0',r_0)
        
        # TODO TEST:
        #   - Set test data to a different size.
        #   - Add bias parameter
        #   - Test on different time-series data
        #   - Put input map in the RNN Model and add more layers.
        #   - Think about projection from input to state need a +/- index on re_a and strictly + on re_b
        #   - Find the orthogonal phse given the non-linear theta value.
        
        Uz = self.map_unitary(U,z)
        Wx = self.map_unitary(W,x[1::])
#        _print('Uz',Uz)
#        _print('Wx',Wx)
        
#        x_ = x[1::]
#        _print('z',z)
#        _print('x_',x_)
        
#        h_t = self.eye[1::,1::] - (Uz @ tf.math.conj(Wx).T)/(tf.math.conj(Uz).T @ Uz)
#        h_t = h_t @ z
#        _print('h_t',h_t)
        
#        _print('mag',mag)
#        _print('ang',ang)
        
        re_a = tf.ones_like(tf.math.real(z))
        im_a = tf.zeros_like(tf.math.imag(z))
        
        re_b = tf.ones_like(tf.math.real(z))
        im_b = tf.zeros_like(tf.math.imag(z))
        
        a = tf.complex(re_a,im_a)
        b = tf.complex(re_b,im_b)
#        _print('a',a)
#        _print('b',b)
        
        # Map to the limit cycle #
        z_t = self.hopf_ODE( z , a , b )
#        _print('z_t',z_t)
        
#        z_mag = tf.math.abs(z_t)
#        z_ang = tf.math.angle(z_t)
#        _print('z_mag',z_mag)
#        _print('z_ang',z_ang)
        
        # Pulling the real-value and re-mapping it to the DFT fixes floating point erros that explode the model.
#        self.fftmean.assign(_mean)
        self.z.assign(z_t)
        
        self.counter.assign_add([1])
        
        y_t = tf.concat([[x0],z_t],0)
        
        return tf.reshape(y_t,input.shape)
    
    
''' '''
class MyRNN(tf.keras.layers.Layer):
    
    ''' MyRNN '''
    def __init__(self,**kwargs):
        
        super(MyRNN, self).__init__(**kwargs)
        
        # Creates state_size number of RNNCells with each cell being state_size large.
#        self.cells = [MyRNNCell(self.state_size,cell_id=j,name='rnncell_'+str(j)) for j in range(self.state_size)]
        self.cell = MyRNNCell(name='rnncell')
        
        self.counter = tf.Variable([0],dtype=tf.float64,trainable=False)


    ''' MyRNN '''
    def build(self,input_shape):
        
        super(MyRNN,self).build(input_shape=input_shape)
        
        in_sz = input_shape[-1]
        self.in_sz = in_sz
        
        self.ones = tf.constant(np.ones(shape=(in_sz,1)),dtype=tf.float64)
        
        self.pow = tf.constant([np.arange(in_sz)],dtype=tf.float64)
        
        self.cell.build(input_shape[1::])
        
        udft = np.exp(-2.j*np.pi*np.arange(in_sz)/in_sz).reshape(-1,1)**np.arange(in_sz)/np.sqrt(in_sz)
        self.udft = tf.constant(udft,dtype=tf.complex128)
        
        self.built = True

    ''' MyRNN '''
    @tf.function
    def inmap(self,v):
        _v = self.udft @ v
        return _v

    ''' MyRNN '''
    @tf.function
    def outmap(self,v):
        _v = (v + tf.math.conj(v)[::-1])/2.
        _v = tf.math.conj(_v).T @ self.udft
        _print('_v',_v)
        return tf.math.real(_v)


    ''' MyRNN '''
    def call(self, input, training=False):

#        tf.print('\n-----------------------------------------------------------------------------------------------------------------------\nCount',self.counter[0],'\n')
        
        _input = tf.reshape(input,[self.in_sz,1])
#        _print('_input',_input)
        
        ''' INPUT TO COMPLEX FUNCTION '''
        _x = self.inmap(_input)
#        _print('_x',_x)
        
        ''' MAP HISTORY TO PROJECTION '''
        # ... #
        
        ''' MAP INPUT TO CELLS '''
        # Make multiple calls ... maybe.
        y_t = self.cell.call(_x,training=training)
        _print('y_t',y_t)
        
        ''' MAP CELL RESULTS TO OUTPUT VALUE(S) '''
        y_out = self.outmap(y_t)
#        _print('y_out',y_out)
        
        ''' UPDATE AUXILIARY VARIABLES '''
        self.counter.assign_add([self.in_sz])
        
        return tf.reshape(y_out,_input.shape)


''' '''
class MyModel(tf.keras.Model):

    ''' '''
    def __init__(self, input, output):
        
        super(MyModel, self).__init__(input, output)
        
        self.my_loss = MyLoss()
        
#        self.my_accu = tf.keras.metrics.KLDivergence(name='kullback_leibler_divergence', dtype=None)
        self.my_accu = tf.keras.metrics.CosineSimilarity(name='cosine_similarity', dtype=None, axis=-1)
        
        self.my_prec = None # tf.keras.metrics.Precision(thresholds=None, top_k=None, class_id=None, name='precision', dtype=None)
        
        self.results = {}
        
        if self.my_loss is not None:
            self.results['loss'] = tf.constant([0.],dtype=tf.float64)
        
#        if self.my_accu is not None:
#            self.results['accuracy'] = tf.constant([0.],dtype=tf.float64)

#        if self.my_prec is not None:
#            self.results['precision'] = tf.constant([0.],dtype=tf.float64)

    ''' '''
    def train_step(self, data):
        
#        tf.print('\n___________________________________________________START__________________________________________________\n')

        input, image = data
        input = tf.cast(input,dtype=tf.float64)
        image = tf.cast(image,dtype=tf.float64)
        
        with tf.GradientTape() as tape:
            
#            tf.print('\n-----------------------------------------------------------------------------------------------------\nPrediction:\n')
            
#            _print('MyModel -> input',input)
            
            pred = tf.cast(self(input, training=True),dtype=input.dtype)  # Forward pass
            pred = tf.reshape(pred,image.shape)
#            _print('MyModel -> pred',pred)
#            _print('MyModel -> image',image)
            
            # My Local Loss for multi-dim loss stuffs
            loss = tf.math.pow(image - pred,2)/(0.+pred.shape[-1])
#            _print('loss',loss)
            
#            loss = self.compiled_loss(image,pred)
#            loss = tf.math.pow(pred - image,2)/(0.+pred.shape[-1])
#            _print('MyModel -> loss',loss)
            
            if self.my_loss is not None:
                self.results['loss'] = tf.math.reduce_mean(loss)
    #            _print('MyModel -> accuracy',self.my_accu.result())
            
#            if self.my_accu is not None:
#                self.my_accu.update_state(image,pred)
#                self.results['accuracy'] = self.my_accu.result()
    #            _print('MyModel -> accuracy',self.my_accu.result())
            
#            if self.my_prec is not None:
#                self.my_prec.update_state(image,pred)
#                self.results['precision'] = self.my_prec.result()
    #            _print('MyModel -> precision',self.my_prec.result())
            
#            tf.print('\n-----------------------------------------------------------------------------------------------------\nBackward Gradient:\n')
            
            # Compute gradients 
            vars = self.trainable_variables
#            for v in vars:
#                _print('MyModel -> train_vars',v)
#                unt = tf.linalg.expm(v)
#                untI = tf.matmul(tf.transpose(unt),unt)
#                _print('V untI',untI)
                
            grad = tape.gradient(loss, vars)
            #tf.print('MyModel:train_step -> Gradients:\n')
#            for g in _grad:
#                _print('MyModel -> grads',g)
#                unt = tf.linalg.expm(g)
#                untI = tf.matmul(tf.transpose(unt),unt)
#                _print('G untI',untI)
            
        wgts = self.trainable_weights
        #tf.print('Wgts trainable_weights:\n',_wgts,'\n')
        
        self.optimizer.apply_gradients(zip(grad,wgts))
        
        # Updated trainable weights
        #tf.print('Wgts POST-apply_gradients:')
        #for _w in _wgts:
        #    printComplex('MyModel:train_step -> _w', _w, print_shape=True)
        
        #tf.print('\n___________________________________________________END__________________________________________________\n\n')
        
        return self.results