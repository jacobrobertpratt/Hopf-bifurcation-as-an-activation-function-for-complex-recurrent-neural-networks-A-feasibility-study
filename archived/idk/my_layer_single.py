import os
import sys
import math
from datetime import datetime
from inspect import currentframe, getframeinfo

import numpy as np
import scipy
from scipy.stats import unitary_group
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import matplotlib.pyplot as plt

import tensorflow_probability as tfp

#from keras.utils import losses_utils

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
    return evls
    
def _cpx_eigvalsh(msg='',mat=None):
    if mat is None:
        return
    tf.print(msg + ' Eigen Values:')
    evls = tf.linalg.eigvalsh(mat)
    for j in range(int(mat.shape[-1])):
        tf.print(str(j+1)+')', tf.math.real(evls[j]), tf.math.imag(evls[j]))
    tf.print()
    return evls

def _print(_str,arr,make_full=False):
    
    tf.print(_str+':\n')
    if isinstance(arr,list):
        tf.print(_str+':\n',arr,'\nshape:',len(arr),'type: (list)')
    elif tf.is_tensor(arr):
        if arr.dtype != tf.complex128:
            arr = tf.cast(arr,dtype=tf.complex128)
        if make_full is False:
            tf.print(' real:\n',tf.math.real(arr))
            tf.print(' imag:\n',tf.math.imag(arr))
        else:
            for j in range(int(arr.shape[0])):
                tf.print(str(j+1)+')', tf.math.real(arr[j]), tf.math.imag(arr[j]))
        tf.print('shape:',arr.shape,'    dtype:',arr.dtype)
        tf.print('\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n')


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
    
    ''' MyRNN '''
    @tf.function
    def calc_cpx_Hopf(self,t,z,a,b):
        
#        tf.print('\n------------------------- HOPF -------------------------\ntime:',t)
        
        re_z, im_z = tf.unstack(z)
        re_a, im_a = tf.unstack(a)
        re_b, im_b = tf.unstack(b)
        
        z = tf.complex(re_z,im_z)
        a = tf.complex(re_a,im_a)
        b = tf.complex(re_b,im_b)
        
        zz = tf.math.conj(z)*z
#        _print('zz',zz)
        
        bzz = b*zz
#        _print('b',b)
#        _print('bzz',bzz)
        
        w = a + bzz
#        _print('a',a)
#        _print('w=a-bzz',w)
        
        dz = w*z
#        _print('z',z,True)
#        _print('max(z)',tf.math.reduce_max(tf.math.abs(z)))
        
#        _print('dz',dz,True)
#        _print('max(dz)',tf.math.reduce_max(tf.math.abs(dz)))
        
        return tf.stack([tf.math.real(dz),tf.math.imag(dz)])
    
    ''' MyRNN '''
    @tf.function
    def hopf_Diff_EQ(self,t,state,re_a,im_a,re_b,im_b):
        re_z, im_z = tf.unstack(state)
        z_stk = tf.stack([re_z,im_z])
        a_stk = tf.stack([re_a,im_a])
        b_stk = tf.stack([re_b,im_b])
        re_dz, im_dz = tf.unstack(self.calc_cpx_Hopf(t,z_stk,a_stk,b_stk))
        return tf.stack([re_dz,im_dz])
    
    ''' MyRNN '''
    @tf.function
    def hopf_ODE(self,z,a,b):
        
        # Setup time intervals #
        t_0 = 0.
        t_1 = 2*np.pi
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
    def build(self, input_shape):
        
        super(MyRNNCell,self).build(input_shape=input_shape)

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
                        tf.print('Generating new matrix for '+nme)
                else:
                    wgt = gen_wgt_matrix(sz)
                    np.save('rnn_mat_'+nme+'.npy',wgt,allow_pickle=True)
                    tf.print('Generating new matrix for '+nme)
            else:
                wgt = gen_wgt_matrix(sz)
                tf.print('Generating new matrix for '+nme)
            return wgt
        
        def make_random_normal_vec(sz,nme='z',do_save=True):
            if do_save is True:
                if os.path.exists('rnn_vec_'+nme+'.npy'):
                    vec = np.load('rnn_vec_'+nme+'.npy',allow_pickle=True)
                    # Check saved shape and replace if not the same.
                    if vec.shape[0] != sz:
                        vec = np.random.rand(sz).reshape(-1,1)
                        np.save('rnn_vec_'+nme+'.npy',vec,allow_pickle=True)
                        tf.print('Generating new vector for '+nme)
                else:
                    vec = np.random.rand(sz).reshape(-1,1)
                    np.save('rnn_vec_'+nme+'.npy',vec,allow_pickle=True)
                    tf.print('Generating new vector for '+nme)
            else:
                vec = np.random.rand(sz).reshape(-1,1)
                tf.print('Generating new vector for '+nme)
            return vec
        
        def make_range_vec(sz,nme='z',do_save=True):
            if do_save is True:
                if os.path.exists('rnn_vec_'+nme+'.npy'):
                    vec = np.load('rnn_vec_'+nme+'.npy',allow_pickle=True)
                    # Check saved shape and replace if not the same.
                    if vec.shape[0] != sz:
                        vec = np.arange(1,sz+1).reshape(-1,1)
                        np.save('rnn_vec_'+nme+'.npy',vec,allow_pickle=True)
                        tf.print('Generating new vector for '+nme)
                else:
                    vec = np.arange(1,sz+1).reshape(-1,1)
                    np.save('rnn_vec_'+nme+'.npy',vec,allow_pickle=True)
                    tf.print('Generating new vector for '+nme)
            else:
                vec = np.arange(1,sz+1).reshape(-1,1)
                tf.print('Generating new vector for '+nme)
            return vec
        
        # Set to True if we want to reuse the initialization matrices for common training
        _save = False
        _train = True
        _sz = in_sz
        
        A = make_unitary_zero_det_weight(_sz,nme='A',do_save=_save)
        A = (A + tf.math.conj(A).T)/2.
        self.A = tf.Variable(   A,
                                name='A_hrm',
                                dtype=tf.complex128,
                                trainable=_train
                                )
        
        B = make_unitary_zero_det_weight(_sz,nme='B',do_save=_save)
        B = (B + tf.math.conj(B).T)/2.
        self.B = tf.Variable(   B,
                                name='B_hrm',
                                dtype=tf.complex128,
                                trainable=_train
                                )
        
        G = make_unitary_zero_det_weight(_sz,nme='G',do_save=_save)
        self.G = tf.Variable(   G,
                                name='G_unt',
                                dtype=tf.complex128,
                                trainable=_train
                                )
        
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


    ''' MyRNNCell '''
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

    ''' MyRNNCell '''
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


    ''' MyRNNCell '''
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


    ''' MyRNNCell '''
    @tf.custom_gradient
    def exp_map(self,a,b):
        N = tf.math.exp(tf.math.conj(a)*b)*(-1.+0.j)
        def grad(gL):
            ga = gL*N*b
            gb = gL*N*tf.math.conj(a)
            return ga, gb
        return N, grad


    ''' MyRNNCell '''
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
#        '''
        z_t = self.hopf_ODE(z,a,b)
        '''
        z_t = tf.math.tanh(a)
        #'''
#        _print('z_t',z_t,True)
        
        # Pulling the real-value and re-mapping it to the DFT fixes floating point erros that explode the model.
        self.z.assign(z_t)
        self.counter.assign_add([1.])
        
        return tf.reshape(z_t,input.shape)
    
    
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
        
        self.cell.build(input_shape[1::])
        
        self.built = True

    ''' MyRNN '''
    def call(self, input, training=False):

#        tf.print('\n-----------------------------------------------------------------------------------------------------------------------\nCount',self.counter[0],'\n')
        
        _input = tf.reshape(input,[self.in_sz,1])
#        _print('_input',_input)
        
        ''' INPUT TO COMPLEX FUNCTION '''
        # ... #
        
        ''' MAP HISTORY TO PROJECTION '''
        # ... #
        
        ''' MAP INPUT TO CELLS '''
        # Make multiple calls ... maybe.
        y_t = self.cell.call(_input,training=training)
#        _print('y_t',y_t)
        
        exp = tf.math.exp(1.j*tf.math.angle( y_t ) )
#        _print('exp',exp)
        
        ''' MAP CELL RESULTS TO OUTPUT VALUE(S) '''
#        y_hat = tf.math.conj(y_t).T @ exp
        y_hat = tf.math.conj( y_t ).T @ y_t
#        _print('y_hat',y_hat)
        
        ''' UPDATE AUXILIARY VARIABLES '''
        self.counter.assign_add([self.in_sz])
        
        return tf.math.real(y_hat) # tf.reshape(y_hat,_input.shape)


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
#            pred = tf.reshape(pred,image.shape)
#            _print('MyModel -> pred',pred)
#            _print('MyModel -> image',image)
            
            # My Local Loss for multi-dim loss stuffs
#            loss = tf.math.pow(image - pred,2)/(0.+pred.shape[-1])
#            _print('loss',loss)
            
            loss = self.compiled_loss(image,pred)
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