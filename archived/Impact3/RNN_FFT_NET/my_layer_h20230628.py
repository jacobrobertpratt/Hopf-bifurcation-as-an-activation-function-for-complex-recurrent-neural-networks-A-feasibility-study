
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

from keras.utils import losses_utils

''' Local Imports'''
from utils import _print, _print_matrix


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
    def __init__(self, state_size, sample_skip=1,cell_id=0, **kwargs):
        
        super(MyRNNCell, self).__init__(**kwargs)
        
        self.cell_id = cell_id
        self.state_size = int(state_size/2)*2
        self.freq = self.cell_id / self.state_size
        self.sample_skip = sample_skip
        
    
    ''' MyRNNCell '''
    def build(self, input_shape):
        
        super(MyRNNCell,self).build(input_shape=input_shape)

        in_sz = input_shape[-1]
        self.in_sz = in_sz
        
        root = np.exp(-2j*np.pi*np.arange(in_sz)/in_sz).reshape(-1,1)
        self.root = tf.constant(root,dtype=tf.complex128)
        
        init_dft = root**np.arange(in_sz)
        self.dft = tf.constant(init_dft,dtype=tf.complex128)
        
        self.zros  = tf.constant(np.zeros(shape=in_sz,dtype=np.float64),dtype=tf.float64)
        self.ones  = tf.constant(np.ones(shape=(in_sz,1),dtype=np.complex128),dtype=tf.complex128)
        
        self.eye = tf.constant(np.eye(in_sz),dtype=tf.complex128)
        
        self.pow = tf.constant([np.arange(in_sz)],dtype=tf.float64)
        
        H1 = tf.cast(unitary_group.rvs(in_sz),dtype=tf.complex128)
        H1_diag = tf.cast(np.diag(np.random.rand(in_sz)),dtype=tf.complex128)
        H1 = H1 @ (H1_diag @ tf.math.conj(H1).T)
        H1 = (H1 + H1.T[::-1,::-1])/2.
        self.H1 = tf.Variable( H1,
                               name='H1_'+str(self.cell_id),
                               dtype=tf.complex128,
                               trainable=False
                              )
        
        H2 = tf.cast(unitary_group.rvs(in_sz),dtype=tf.complex128)
        H2_diag = tf.cast(np.diag(np.random.rand(in_sz)),dtype=tf.complex128)
        H2 = H2 @ (H2_diag @ tf.math.conj(H2).T)
        H2 = (H2 + H2.T[::-1,::-1])/2.
        self.H2 = tf.Variable( H2,
                               name='H2_'+str(self.cell_id),
                               dtype=tf.complex128,
                               trainable=False
                              )
        
        U = tf.cast(unitary_group.rvs(in_sz),dtype=tf.complex128)
        U = tf.linalg.logm(U)
        U = (U + tf.math.conj(U[::-1,::-1]))/2.
        U = tf.linalg.expm(U)
        self.U = tf.Variable( U,
                              name='U_'+str(self.cell_id),
                              dtype=tf.complex128,
                              trainable=False
                             )
        
        V = tf.cast(unitary_group.rvs(in_sz),dtype=tf.complex128)
        V = tf.linalg.logm(V)
        V = (V + tf.math.conj(V[::-1,::-1]))/2.
        V = tf.linalg.expm(V)
        self.V = tf.Variable( V,
                              name='V_'+str(self.cell_id),
                              dtype=tf.complex128,
                              trainable=False
                             )
        
        W = tf.linalg.logm(tf.cast(unitary_group.rvs(in_sz),dtype=tf.complex128))
        W = tf.linalg.expm((W + tf.math.conj(W[::-1,::-1]))/2.)
        self.W = tf.Variable( W,
                              name='W_'+str(self.cell_id),
                              dtype=tf.complex128,
                              trainable=False
                             )
        
        init_z = np.random.rand(in_sz).reshape(-1,1)
        init_z = np.exp(1.j*init_z) * np.random.rand(in_sz).reshape(-1,1)
        init_z = (init_z + tf.math.conj(tf.expand_dims(init_z.T[0][::-1],1)))       # Make a conj-symmetric vector #
        self.z = tf.Variable(
                              initial_value=init_z,
                              name='z_'+str(self.cell_id),
                              dtype=tf.complex128,
                              trainable=False
                             )
        
        self.counter = tf.Variable([0],dtype=tf.float64,trainable=False)
        
        self.built = True

    @tf.function
    def calc_cpx_Hopf(self, z, A, B, C):
        
#        tf.print('\n-----------------------------------------------------------------------------\n')
        
        re_z, im_z = tf.unstack(z)
        re_A, im_A = tf.unstack(A)
        re_B, im_B = tf.unstack(B)
        re_C, im_C = tf.unstack(C)
        
        z = tf.complex(re_z,im_z)
        A = tf.complex(re_A,im_A)
        B = tf.complex(re_B,im_B)
        C = tf.complex(re_C,im_C)
        
        Izz = B @ (self.eye * (tf.math.conj(z).T @ C @ z))
#        _print('Izz',Izz)

        W = A - Izz
#        _print('W',W)

        dz = W @ z
#        _print('dz',dz)

        return tf.stack([tf.math.real(dz),tf.math.imag(dz)])
    
    @tf.function
    def hopf_Diff_EQ(self,t,state,re_A,im_A,re_B,im_B,re_C,im_C):
        re_z, im_z = tf.unstack(state)
        z_stk = tf.stack([re_z,im_z])
        A_stk = tf.stack([re_A,im_A])
        B_stk = tf.stack([re_B,im_B])
        C_stk = tf.stack([re_C,im_C])
        re_dz, im_dz = tf.unstack(self.calc_cpx_Hopf(z_stk,A_stk,B_stk,C_stk))
        return tf.stack([re_dz,im_dz])
    
    @tf.function
    def hopf_ODE(self,z,A,B,C):
        
        # Setup time intervals #
        t_0 = 0.
        t_1 = 2*np.pi # self.in_sz # t_0 + self.in_sz  # Should be changed to output size.
        t_r = tf.linspace(t_0,t_1,num=int(t_1)*4)
        
        re_z, im_z = tf.math.real(z), tf.math.imag(z)
        re_A, im_A = tf.math.real(A), tf.math.imag(A)
        re_B, im_B = tf.math.real(B), tf.math.imag(B)
        re_C, im_C = tf.math.real(C), tf.math.imag(C)
        
        ode_state = tf.stack([re_z,im_z])
        ode_const = {'re_A':re_A,'im_A':im_A,'re_B':re_B,'im_B':im_B,'re_C':re_C,'im_C':im_C}
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
    
    @tf.custom_gradient
    def cpx_matmul(self,W,z):
        Wz = W @ z
        def grad(dL):
            _print('dL',dL)
            dW = tf.zeros_like(W)
            dz = tf.zeros_like(z)
            return dW, dz
        return Wz, grad
    
    ''' MyRNNCell '''
    def call(self, input, training=False):
        
        x_in = input
        z = self.z
        U = self.U
        V = self.V
        W = self.W
        H1 = self.H1
        H2 = self.H2
#        _print_matrix('U',U)
        
        # Map the input 'x' to the phase-space of the state.
        rng = np.arange(2.*np.pi/self.in_sz,2.*np.pi,2.*np.pi/(self.in_sz+1))
        exp_in = (tf.math.exp(-1.j*tf.math.angle(self.z))**rng) @ x_in
        exp_in = tf.linalg.diag(exp_in.T[0])
#        _print('exp_in',exp_in)
        
        A = self.eye
        B = self.eye
        C = self.eye
        
        # Map to the limit cycle
#        _print('z',z)
        z_t = self.hopf_ODE(z,A,B,C)
        self.z.assign(z_t)
        _print('z_t',z_t)
        
        self.counter.assign_add([self.in_sz])
        
        return tf.reshape(z_t,input.shape)
    
    
''' '''
class MyRNN(tf.keras.layers.Layer):
    
    
    ''' MyRNN '''
    def __init__(
                  self,
                  cell_size=2,
                  sample_skip=1,
                  **kwargs
                 ):
        
        super(MyRNN, self).__init__(**kwargs)
        
        if cell_size < 2:
            tf.print('ERROR: cell size < 2.')
            exit(0)
        
        self.cell_size = cell_size
        
        # Creates state_size number of RNNCells with each cell being state_size large.
#        self.cells = [MyRNNCell(self.state_size,cell_id=j,name='rnncell_'+str(j)) for j in range(self.state_size)]
        self.cell = MyRNNCell(cell_size,sample_skip=sample_skip,cell_id=1,name='rnncell_'+str(1))
        
        self.counter = tf.Variable([0],dtype=tf.float64,trainable=False)


    ''' MyRNN '''
    def build(self,input_shape):
        
        super(MyRNN,self).build(input_shape=input_shape)
        
        in_sz = input_shape[-1]
        self.in_sz = in_sz
        
        self.pow = tf.constant([np.arange(in_sz)],dtype=tf.float64)
        
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
        z_t = self.cell.call(_input,training=training)
#        _print('z_t',z_t)
        
        ''' MAP CELL RESULTS TO OUTPUT VALUE(S) '''
        rng = np.arange(2.*np.pi/self.in_sz,2.*np.pi,2.*np.pi/(self.in_sz+1))
        map = tf.math.exp(-1.j*tf.math.angle(z_t))**rng
#        _print('map',map)
        
        cpx_y = tf.math.conj(z_t).T @ map
        _print('cpx_y',cpx_y)
        
        y_out = tf.math.real(cpx_y)
        
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
        
        if self.my_accu is not None:
            self.results['accuracy'] = tf.constant([0.],dtype=tf.float64)

        if self.my_prec is not None:
            self.results['precision'] = tf.constant([0.],dtype=tf.float64)

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
            
            loss = self.compiled_loss(image,pred)
#            _print('MyModel -> _loss',_loss)

            if self.my_loss is not None:
                self.results['loss'] = tf.math.reduce_mean(loss)
    #            _print('MyModel -> accuracy',self.my_accu.result())
            
            if self.my_accu is not None:
                self.my_accu.update_state(image,pred)
                self.results['accuracy'] = self.my_accu.result()
    #            _print('MyModel -> accuracy',self.my_accu.result())
            
            if self.my_prec is not None:
                self.my_prec.update_state(image,pred)
                self.results['precision'] = self.my_prec.result()
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




''' EXTRA STUFF '''

''' 
Orthogonal Matrix with 1-pair of equal and opposite purely complex eigenvectors
'''

'''
def rayleigh_quotent(self,msg,A,U,eval,evec):

tf.print('\n\n\n\n---------------------------------------'+msg+'--------------------------------------\n')

for j in range(25):

inv = tf.linalg.inv((A - (eval * U)))
num = inv @ evec
den = tf.math.sqrt(tf.math.conj(num).T @ num)
evec = num / den
#            _print('evec',evec)

eval = (tf.math.conj(evec).T @ A @ evec)/(tf.math.conj(evec).T @ evec)
#            _print('eval',eval)

#            dif = tf.math.reduce_max(tf.math.abs(A @ evec - eval * evec))
#            _print('dif',dif)

return eval, evec

'''

'''
#    @tf.custom_gradient
def calc_du_linear(self,u,z):

den = tf.math.sqrt((tf.math.conj(u).T @ u)*(tf.math.conj(z).T @ z))

I_uu = self.eye - ((u @ tf.math.conj(u).T)/(tf.math.conj(u).T @ u))
I_zz = self.eye - ((z @ tf.math.conj(z).T)/(tf.math.conj(z).T @ z))

du = (I_uu @ z) / den
dz = (I_zz @ u) / den

re_dL = tf.math.conj(du) * dz + tf.math.conj(dz) * du
im_dL = tf.math.conj(du) * dz - tf.math.conj(dz) * du

dL = re_dL + im_dL

def grad(dG):

#            tst = (u + tf.math.conj(u).T)/2.
tst = dL / tf.math.conj(u).T
_print('tst',tst)

dH = tf.zeros_like(H)
du = tf.zeros_like(u)

return dH, du

return dL #, grad
'''

       
'''
RAYEIGH QUOTENT DERIVATIVE
den = tf.math.sqrt((tf.math.conj(Mx).T @ Mx)*(tf.math.conj(z).T @ z))
I_uu = self.eye - ((Mx @ tf.math.conj(Mx).T)/(tf.math.conj(Mx).T @ Mx))
I_zz = self.eye - ((z @ tf.math.conj(z).T)/(tf.math.conj(z).T @ z))
du = (I_uu @ z) / den
dz = (I_zz @ Mx) / den
re_dL = tf.math.conj(du) * dz + tf.math.conj(dz) * du
im_dL = tf.math.conj(du) * dz - tf.math.conj(dz) * du
L = re_dL + im_dL
_print('L',L)
'''

'''
#        init_ones = tf.keras.initializers.Ones()
#        init_zeros = tf.keras.initializers.Zeros()
#        init_glorot = tf.keras.initializers.GlorotUniform(seed=None)
'''


'''

def cpxCalcEigs(self,mat):
evls = tf.linalg.eigvals(mat)
tf.print(' evls[0]:', tf.math.real(evls[0]), tf.math.imag(evls[0]))
tf.print(' evls[1]:', tf.math.real(evls[1]), tf.math.imag(evls[1]))
tf.print(' evls[2]:', tf.math.real(evls[2]), tf.math.imag(evls[2]))
tf.print(' evls[3]:', tf.math.real(evls[3]), tf.math.imag(evls[3]))
tf.print(' evls[4]:', tf.math.real(evls[4]), tf.math.imag(evls[4]))
tf.print(' evls[5]:', tf.math.real(evls[5]), tf.math.imag(evls[5]))
tf.print(' evls[6]:', tf.math.real(evls[6]), tf.math.imag(evls[6]))
tf.print(' evls[7]:', tf.math.real(evls[7]), tf.math.imag(evls[7]))
tf.print(' evls[8]:', tf.math.real(evls[8]), tf.math.imag(evls[8]))
tf.print(' evls[9]:', tf.math.real(evls[9]), tf.math.imag(evls[9]))
tf.print('evls[10]:',tf.math.real(evls[10]),tf.math.imag(evls[10]))
tf.print('evls[11]:',tf.math.real(evls[11]),tf.math.imag(evls[11]))
tf.print('evls[12]:',tf.math.real(evls[12]),tf.math.imag(evls[12]))
tf.print('evls[13]:',tf.math.real(evls[13]),tf.math.imag(evls[13]))
tf.print('evls[14]:',tf.math.real(evls[14]),tf.math.imag(evls[14]))
tf.print('evls[15]:',tf.math.real(evls[15]),tf.math.imag(evls[15]))

def calcCpxStuff(self,t,L_stk,X_stk,Z_stk,A_stk,U_stk,E_stk):

re_X, im_X = tf.unstack(X_stk)  # constant
re_Z, im_Z = tf.unstack(Z_stk)
re_L, im_L = tf.unstack(L_stk)  # constant
re_A, im_A = tf.unstack(A_stk)  # constant
re_U, im_U = tf.unstack(U_stk)  # constant
re_E, im_E = tf.unstack(E_stk)  # constant

x = tf.complex(re_X,im_X)
z = tf.complex(re_Z,im_Z)
L = tf.complex(re_L,im_L)
A = tf.complex(re_A,im_A)
U = tf.complex(re_U,im_U)
E = tf.complex(re_E,im_E)

dz = A @ z
#        _print('dz',dz)

return tf.stack([tf.math.real(dz),tf.math.imag(dz)])

def real_ode_unit_tst(self,t,ode_state,re_A,im_A):

#        tf.print('\n---------------------------------------------------------\nt:',t,'\n')

# Stacked Input Constants
A_stk = tf.stack([re_A,im_A])

# Input States
re_X, im_X, re_Z, im_Z, re_L, im_L, re_U, im_U, re_E, im_E = tf.unstack(ode_state)

# Convert back to vertical vectors
re_x = tf.expand_dims(tf.linalg.diag_part(re_X),1)
im_x = tf.expand_dims(tf.linalg.diag_part(im_X),1)
re_z = tf.expand_dims(tf.linalg.diag_part(re_Z),1)
im_z = tf.expand_dims(tf.linalg.diag_part(im_Z),1)

# Stacked Input States
X_stk = tf.stack([re_x,im_x])
Z_stk = tf.stack([re_z,im_z])
L_stk = tf.stack([re_L,im_L])
U_stk = tf.stack([re_U,im_U])

E_stk = tf.stack([re_E,im_E])

dZ_stk = self.calcCpxStuff(t,L_stk,X_stk,Z_stk,A_stk,U_stk,E_stk)

# State output
re_dX = tf.zeros_like(re_X)
im_dX = tf.zeros_like(im_X)
re_dz, im_dz = tf.unstack(dZ_stk)
re_dZ = tf.linalg.diag(re_dz.T[0])
im_dZ = tf.linalg.diag(im_dz.T[0])

# No change to the Linear function over time
re_dL = tf.zeros_like(re_L)
im_dL = tf.zeros_like(im_L)

re_dU = tf.zeros_like(re_U)
im_dU = tf.zeros_like(im_U)

re_dE = tf.zeros_like(re_E)
im_dE = tf.zeros_like(im_E)

return tf.stack([re_dX,im_dX,re_dZ,im_dZ,re_dL,im_dL,re_dU,im_dU,re_dE,im_dE])

#    @tf.function
def map_to_limit_cycle(self,z,x,S,D):

# Setup time intervals
t_0 = 0.        # self.counter[0]
t_1 = 2*np.pi   # t_0 + self.sample_skip
t_r = tf.linspace(t_0,t_1,num=self.sample_skip*4) # self.sample_skip*4) # self.sample_skip*8)

# Separate real & imaginary parts
re_x = tf.math.real(x)
im_x = tf.math.imag(x)
re_z = tf.math.real(z)
im_z = tf.math.imag(z)

# state values -> (must have a return change in state i.e., dx/dt output from ODESolver. change can be zeros implying no change.)
re_X = tf.linalg.diag(re_x.T[0])
im_X = tf.linalg.diag(im_x.T[0])
re_Z = tf.linalg.diag(re_z.T[0])
im_Z = tf.linalg.diag(im_z.T[0])

# A is skew-hermitian => U is unitary.
#        U = tf.linalg.expm(S)
re_U = tf.math.real(D)
im_U = tf.math.imag(D)

L = S # U @ tf.linalg.diag(x.T[0]) @ tf.math.conj(U).T
re_L = tf.math.real(L)
im_L = tf.math.imag(L)

re_A = tf.math.real(S)
im_A = tf.math.imag(S)
#        A = tf.complex(re_A,im_A)

E = tf.math.exp(-2j*np.pi/self.inpt_size)*self.eye
re_E = tf.math.real(E)
im_E = tf.math.imag(E)

ode_state = tf.stack([re_X,im_X,re_Z,im_Z,re_L,im_L,re_U,im_U,re_E,im_E])
# Constant Values passed to ODESolver
consts = {'re_A':re_A,'im_A':im_A}
DormPrnc = tfp.math.ode.DormandPrince()
ode = DormPrnc.solve(
self.real_ode_unit_tst,
t_0,
ode_state,
solution_times=t_r,
constants=consts
)
ode_t = ode.states[-1]
re_Xt, im_Xt, re_Zt, im_Zt, re_Lt, im_Lt, re_Ut, im_Ut, re_Et, im_Et = tf.unstack(ode_t)

re_zt = tf.linalg.diag_part(re_Zt)
im_zt = tf.linalg.diag_part(im_Zt)

#        zt = tf.expand_dims(tf.complex(re_zt,im_zt),1)
zt = tf.complex(re_zt,im_zt)
#        _print('zt',zt)

zt = z
return tf.reshape(zt,z.shape) # tf.reshape(zt,z.shape)

'''


'''
CREATE MATRIX WITH eigenvalue 2x = +/-iw and other are less than zero with iw = 0.

UT = U.T

E0 = tf.expand_dims(UT[0],1)
E1 = tf.expand_dims(UT[1],1)
E2 = tf.expand_dims(UT[2],1)
E3 = tf.expand_dims(UT[3],1)
E4 = tf.expand_dims(UT[4],1)
E5 = tf.expand_dims(UT[5],1)
E6 = tf.expand_dims(UT[6],1)
E7 = tf.expand_dims(UT[7],1)
E8 = tf.expand_dims(UT[8],1)
E9 = tf.expand_dims(UT[9],1)
E10 = tf.expand_dims(UT[10],1)
E11 = tf.expand_dims(UT[11],1)
E12 = tf.expand_dims(UT[12],1)
E13 = tf.expand_dims(UT[13],1)
E14 = tf.expand_dims(UT[14],1)
E15 = tf.expand_dims(UT[15],1)

P0 = (E0 @ tf.math.conj(E0).T)/(tf.math.conj(E0).T @ E0)
P1 = (E1 @ tf.math.conj(E1).T)/(tf.math.conj(E1).T @ E1)
P2 = (E2 @ tf.math.conj(E2).T)/(tf.math.conj(E2).T @ E2)
P3 = (E3 @ tf.math.conj(E3).T)/(tf.math.conj(E3).T @ E3)
P4 = (E4 @ tf.math.conj(E4).T)/(tf.math.conj(E4).T @ E4)
P5 = (E5 @ tf.math.conj(E5).T)/(tf.math.conj(E5).T @ E5)
P6 = (E6 @ tf.math.conj(E6).T)/(tf.math.conj(E6).T @ E6)
P7 = (E7 @ tf.math.conj(E7).T)/(tf.math.conj(E7).T @ E7)
P8 = (E8 @ tf.math.conj(E8).T)/(tf.math.conj(E8).T @ E8)
P9 = (E9 @ tf.math.conj(E9).T)/(tf.math.conj(E9).T @ E9)
P10 = (E10 @ tf.math.conj(E10).T)/(tf.math.conj(E10).T @ E10)
P11 = (E11 @ tf.math.conj(E11).T)/(tf.math.conj(E11).T @ E11)
P12 = (E12 @ tf.math.conj(E12).T)/(tf.math.conj(E12).T @ E12)
P13 = (E13 @ tf.math.conj(E13).T)/(tf.math.conj(E13).T @ E13)
P14 = (E14 @ tf.math.conj(E14).T)/(tf.math.conj(E14).T @ E14)
P15 = (E15 @ tf.math.conj(E15).T)/(tf.math.conj(E15).T @ E15)

prjP0x = P0 @ x
prjP1z = P1 @ z
com = prjP1z @ tf.math.conj(prjP0x).T - prjP0x @ tf.math.conj(prjP1z).T
tot = com - (P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9 + P10 + P11 + P12 + P13 + P14 + P15)
A = tot
'''

'''

# MATRIX MULTIPLICAITION EXAMPLE WITH GRADIENT BACKPROP #

def cpxMatVecMult(self,mat,vec):
re_mat, im_mat = tf.unstack(mat)
re_vec, im_vec = tf.unstack(vec)
cpx_mat = tf.complex(re_mat,im_mat)
cpx_vec = tf.complex(re_vec,im_vec)
cpx_mlt = cpx_mat @ cpx_vec
mlt_stk = tf.stack([tf.math.real(cpx_mlt),tf.math.imag(cpx_mlt)])
def grad(stk_dL):

#            tf.print('\n-----------------------------------------------------------------------------\n')

re_dL, im_dL = tf.unstack(stk_dL)

#            g_stk_re = tf.gradients(mlt_stk,re_mat)
#            _print('g_stk_re',g_stk_re)

#            g_stk_im = tf.gradients(mlt_stk,im_mat)
#            _print('g_stk_im',g_stk_im)

re_dm = tf.zeros_like(re_mat)
im_dm = tf.zeros_like(im_mat)
re_dv = tf.zeros_like(re_vec)
im_dv = tf.zeros_like(im_vec)

din_mat = tf.stack([re_dm,im_dm])
din_vec = tf.stack([re_dv,im_dv])

return din_mat, din_vec

return mlt_stk #, grad
'''


'''

# OLD HOPF-BIFURCATION VERSION

#    @tf.function
def real_ode_holo_tst(self,t,ode_i,re_M,im_M):

#        tf.print('\n-------------------------------------------------------------------------------------\nTime:',t,'\n')
#        _print('re_W',re_W)
#        _print('im_W',im_W)

re_z, im_z, = tf.unstack(ode_i) # Comes out vertical vectors
#        _print('re_z',re_z)
#        _print('im_z',im_z)

z_stk = tf.stack([re_z,im_z])
M_stk = tf.stack([re_M,im_M])

# --------------- WORKING HOPF-BIFURCATION --------------- #
# Works to a specific degree, but the phase & beta changes are not 
#        re_inr_x, im_inr_x = tf.unstack(self.cpxWeightedInnr(tf.stack([re_x,im_x]),tf.stack([re_W,im_W])))
#        _print('re_inr_x',re_inr_x)
#        _print('im_inr_x',im_inr_x)

#        re_zHz, im_zHz = tf.unstack(self.cpxWeightedInnr(z_stk,H_stk))
#        _print('re_zHz',re_zHz)

# Multiply Hermitian matrix by inner product of the state.
#        re_zz, im_zz = tf.unstack(self.cpxInnrProd(z_stk,z_stk))
#        _print('re_inr_z',re_inr_z)
#        _print('im_inr_z',im_inr_z)

W_z2 = M_stk * re_zz
#        _print('W_z2',W_z2)

W_diff = M_stk - W_z2
#        _print('W_diff',W_diff)

#        signWstk = tf.math.sign(A_stk[1])
#        signWz2 = tf.math.sign(W_z2[1])
#        _print('sign_diff',signWstk - signWz2)

re_dz, im_dz = tf.unstack(self.cpxMultMatVec(W_diff,z_stk))
#        _print('re_dz',re_dz)
#        _print('im_dz',im_dz)

#        re_dz = tf.zeros_like(re_z)
#        im_dz = tf.zeros_like(im_z)

return tf.stack([re_dz,im_dz],0)



# BAKER-CAMPBELL-HAUSDORFF 

def baker_cambell_hausdorf(self,X,Y,rng=24):
def commutator(A,B):
return A @ B - B @ A
Xcomm = Ycomm = commutator(X,Y)/2.
lst_Xcomm = lst_Ycomm = tf.zeros_like(Xcomm)
for n in range(3,rng):
#            tf.print(str(n)+')')
fact = tf.cast([0.+math.factorial(n)],dtype=tf.float64)
Xcomm = Xcomm + commutator(X,Xcomm)/fact
Ycomm = Ycomm + commutator(Y,Ycomm)/fact
#            _print('Xcomm',Xcomm)
#            _print('Ycomm',Ycomm)
# Set last commutators for comparison
lst_Xcomm = Xcomm
lst_Ycomm = Ycomm
Z = X + Y + Xcomm + Ycomm
return Z / 2.

# EXPLICIT EIGEN VALUE FOR A 16X16 MATRIX

def cpxCalcEigs(self,mat):
evls = tf.linalg.eigvals(mat)
tf.print(' evls[0]:',tf.math.real(evls[0]),tf.math.imag(evls[0]))
tf.print(' evls[1]:',tf.math.real(evls[1]),tf.math.imag(evls[1]))
tf.print(' evls[2]:',tf.math.real(evls[2]),tf.math.imag(evls[2]))
tf.print(' evls[3]:',tf.math.real(evls[3]),tf.math.imag(evls[3]))
tf.print(' evls[4]:',tf.math.real(evls[4]),tf.math.imag(evls[4]))
tf.print(' evls[5]:',tf.math.real(evls[5]),tf.math.imag(evls[5]))
tf.print(' evls[6]:',tf.math.real(evls[6]),tf.math.imag(evls[6]))
tf.print(' evls[7]:',tf.math.real(evls[7]),tf.math.imag(evls[7]))
tf.print(' evls[8]:',tf.math.real(evls[8]),tf.math.imag(evls[8]))
tf.print(' evls[9]:',tf.math.real(evls[9]),tf.math.imag(evls[9]))
tf.print('evls[10]:',tf.math.real(evls[10]),tf.math.imag(evls[10]))
tf.print('evls[11]:',tf.math.real(evls[11]),tf.math.imag(evls[11]))
tf.print('evls[12]:',tf.math.real(evls[12]),tf.math.imag(evls[12]))
tf.print('evls[13]:',tf.math.real(evls[13]),tf.math.imag(evls[13]))
tf.print('evls[14]:',tf.math.real(evls[14]),tf.math.imag(evls[14]))
tf.print('evls[15]:',tf.math.real(evls[15]),tf.math.imag(evls[15]))



# STACKED GRADIENT EXAMPLE FOR BACKPROB IN CPXMATMULTVEC
def grad(stk_dL):

re_dL, im_dL = tf.unstack(stk_dL)
#            _print('re_dL',re_dL)
#            _print('im_dL',im_dL)

re_dm = tf.gradients(mlt,re_mat)
_print('re_dm',re_dm)

re_dm = tf.zeros_like(re_mat)
im_dm = tf.zeros_like(im_mat)
re_dv = tf.zeros_like(re_vec)
im_dv = tf.zeros_like(im_vec)

din_mat = tf.stack([re_dm,im_dm])
din_vec = tf.stack([re_dv,im_dv])

return din_mat, din_vec
'''




'''
zN = (z @ tf.math.conj(z).T) / (tf.math.conj(z).T @ z)
zP = self.eye - zN

xN = (x @ x.T) / (x.T @ x)
xP = tf.math.real(self.eye) - xN

tst = U @ xN @ tf.math.conj(U).T
_print_matrix('tst',tst)

tst = tst - U @ xP @ tf.math.conj(U).T
_print_matrix('tst',tst,print_Ichk=True)

_one = tf.math.conj(tst).T @ zN + tf.math.conj(tst).T @ zP
_print_matrix('_one',_one,print_Ichk=True)

dx = (x - x.T)/2.
adj_x = U @ dx @ tf.math.conj(U).T
_print_matrix('adj_x',adj_x)

dz = (z - tf.math.conj(z).T)/2.
adj_z = U @ dz @ tf.math.conj(U).T
_print_matrix('adj_z',adj_z)

inr = tf.math.conj(adj_x).T @ H @ adj_z
_print_matrix('inr',inr,print_Ichk=True)

inrP = tf.math.conj(adj_x).T @ H @ adj_z
_print_matrix('inrP',inrP)

inr_xz = tf.math.conj(adj_x).T @ H @ adj_z + tf.math.conj(adj_z).T @ tf.math.conj(H).T @ adj_x
_print_matrix('inr_xz',inr_xz)

xN = (x @ x.T) / (x.T @ x)
xP = tf.math.real(self.eye) - xN

zN = (z @ tf.math.conj(z).T) / (tf.math.conj(z).T @ z)
zP = self.eye - zN

UzU = tf.math.conj(U).T @ zN @ U - tf.math.conj(U).T @ zP @ U
#        _print_matrix('UzU',UzU)

#        zX = tf.math.conj(UzU).T @ xN + tf.math.conj(UzU).T @ xP        # Hermitian-Unitary
#        _print_matrix('zX',zX,print_Ichk=True)
'''

'''
#        herm = (self.wgt + tf.math.conj(self.wgt).T) / 2.
#        skew = (self.wgt - tf.math.conj(self.wgt).T) / 2.
#        _print('herm',herm)
#        _print('skew',skew)
        
#        HrSk_commtst = herm @ skew - skew @ herm
#        _print('HrSk_commtst',HrSk_commtst)
#        exit(0)
        
#        exp_wgt = tf.linalg.expm(self.wgt)
#        _print('exp_wgt',exp_wgt)
#        expwgt_evals = tf.linalg.eigvals(exp_wgt)
#        _print('expwgt_evals',expwgt_evals)
        
        # expm(M) = expm(herm) @ expm(skew), since herm and skew commute
#        exp_herm = tf.linalg.expm(herm)     # Commutative
#        exp_skew = tf.linalg.expm(skew)     # Commutative
#        nexp_skew = tf.linalg.expm(-skew)   # Commutative, == conj(exp_skew).T
#        _print('exp_herm',exp_herm)
#        _print('exp_skew',exp_skew)
#        _print('nexp_skew',nexp_skew)
#        _print('eig exp_herm',tf.linalg.eigvals(exp_herm))
#        _print('eig exp_skew',tf.linalg.eigvals(exp_skew))
#        _print('eig nexp_skew',tf.linalg.eigvals(nexp_skew))
        
#        adj_w = exp_skew @ (exp_herm @ nexp_skew)    # Adjoint of manifold (i.e. tangent space at origin)
#        _print('adj_w',adj_w)
#        _print('adj_w',adj_w @ adj_w)
        
#        adj_inv = tf.linalg.inv(adj_w)
#        _print('adj_inv',adj_inv)
        
#        cos_skew = (exp_skew + nexp_skew) / 2.      # hermitian         -> Even parts of Taylor series
#        sin_skew = (exp_skew - nexp_skew) / 2.      # Skew-hermitian    -> Odd parts of Taylor series
#        _print('cos_skew',cos_skew)
#        _print('sin_skew',sin_skew)
        
#        unt_skew = cos_skew + sin_skew
#        _print('unt_skew',unt_skew)
        
        # Is the Euler breakdown of our manifold where unt_wgt == exp_wgt above.
#        unt_wgt = exp_herm @ cos_skew + exp_herm @ sin_skew
#        _print('unt_wgt',unt_wgt)
        
        # Take only real-valued manifold @ even Taylor series
#        exp_cos = exp_herm @ cos_skew
#        _print('exp_cos',exp_cos)
        
        # Take only real-valued manifold @ odd Taylor series
#        exp_sin = exp_herm @ sin_skew
#        _print('exp_sin',exp_sin)
'''



'''
MANIFOLD INNER PRODUCT CALCULATIONS

# U is a point in our tangent space of our observed manifold    U = [x',x]. 
#   We use the exponential map to map to a point on the global manifold.
tan_in = tf.transpose(input) - input
_print('tan_in',tan_in)

orth_in = tf.linalg.expm(tan_in)
_print('orth_in',orth_in)

unt_in = orth_in.T @ orth_in
_print('unt_in',unt_in)

com_in = (tan_in.T @ orth_in - orth_in.T @ tan_in) @ orth_in
_print('com_in',com_in)

adj1_in = orth_in @ tan_in @ orth_in.T      # Same as --> adj = O @ T @ inv(O) (have tested)
_print('adj1_in',adj1_in)

cannon_innr_prod = tf.linalg.trace(tan_in.T @ (self.eye - orth_in @ orth_in.T) @ tan_in)    # Canonical Inner Product.
_print('cannon_innr_prod',cannon_innr_prod)

o = self.o
O = tf.linalg.expm(o)
oT = tf.transpose(o)
OT = tf.transpose(O)
cannon_innr_prod_o = tf.linalg.trace(oT @ (self.eye - O @ OT) @ o)    # Canonical Inner Product.
_print('cannon_innr_prod_o',cannon_innr_prod_o)
'''

'''
init_ones = tf.keras.initializers.Ones()
init_zeros = tf.keras.initializers.Zeros()
init_glorot = tf.keras.initializers.GlorotUniform(seed=None)
init_eye = tf.keras.initializers.Identity()
'''

''' MyRNNCell
#    @tf.function
def cpx_odefunc_limit_cycle(self,t,state,ax,ay,bx,by):
    
    z_i = tf.complex(state[0],state[1])
    
    c_a = tf.complex(ax,ay)
    c_b = tf.complex(bx,by)
    
    zabs = tf.cast(tf.math.abs(z_i)**2,dtype=tf.complex128)
    dz_dt = z_i * (c_b + c_a*(zabs))
    
    return [tf.math.real(dz_dt),tf.math.imag(dz_dt)]
'''

''' MyRNNCell
#    @tf.function
def real_odefunc_limit_cycle(self,t,x_i,dw,db_re,db_im):
    
    r_i, w_i, br_i, bi_i = tf.unstack(x_i)
    
    dr_dt = r_i * (1.0 - (r_i * r_i))
    dw_dt = dw
    
    dbr_dt = db_re
    dbi_dt = db_im
    
    return tf.stack([dr_dt,dw_dt,dbr_dt,dbi_dt])
'''

''' NEURAL ODE
state = tf.concat([in_0,A],0)

ode = tfp.math.ode.DormandPrince().solve(
                                            self.real_odefunc_exp,
                                            t_1,
                                            state,
                                            solution_times=t
                                           )
ode_states = ode.states
x_t = ode_states[-1][0]

'''

'''
def grad(dL):
    
    e_dL = tf.expand_dims(dL,0)
    dL_di = e_dL * tf.gradients(x_t,in_0)[0]
    dL_dA = e_dL * tf.gradients(x_t,A)[0]
    
    return dL_di, dL_dA
'''

''' INPUT to COMPLEX MAP
dy_dx = tf.expand_dims(tf.concat([[0.],input[0:-1]-input[1::]],0),0)
sym = (tf.transpose(_input) + _input) / 2.0
skew = (tf.transpose(_input) - _input) / 2.0
skew_herm = tf.complex(skew,sym)
unit_in = tf.transpose(tf.linalg.expm(skew_herm))
'''

'''
_dw = self.dw

b_0abs = tf.math.abs(self.b)
b_0ang = tf.math.angle(self.b)

db = self.db
db_re = tf.math.real(db)
db_im = tf.math.imag(db)

x_0abs = tf.math.abs(x_0)
x_0ang = tf.math.angle(x_0)

state = tf.stack([x_0abs,x_0ang,b_0abs,b_0ang])

ode = tfp.math.ode.DormandPrince().solve(
                                          self.real_odefunc_limit_cycle,
                                          t_0,
                                          state,
                                          solution_times=t,
                                          constants={'dw':_dw,'db_re':db_re,'db_im':db_im}
                                         )
r_t, w_t, b_tabs, b_tang = tf.unstack(ode.states[-1])
#            _print('self.A',self.A)

'''

'''
MAP TO MULTIPLE CELLS
@tf.function
def map_to_cells(self, input_map, training=False):
    
    state_size = tuple(c.state_size for c in self.cells)
    nest_cells = tf.nest.pack_sequence_as(state_size,tf.nest.flatten(self.cells))
    
    new_cells = []
    for cell, nest in zip(self.cells,nest_cells):
        new_cell = cell.call(input_map,training=training)
        new_cells.append(new_cell)
    _new_map = tf.convert_to_tensor(new_cells)
    
    return _new_map
'''

#### init
'''
self.x = tf.Variable(
                      initial_value=init_ones(
                                               shape=(self.state_size,1),
                                               dtype=tf.float64,
                                              ),
                       name='x_c'+str(self.cell_id),
                       dtype=tf.float64,
                       trainable=False
                      )

init_glorot_A = init_glorot(shape=(self.state_size,self.state_size),dtype=tf.float64)
#        init_glorot_A -= tf.math.reduce_mean(init_glorot_A)
self.A = tf.Variable(
                      initial_value=init_glorot_A,
                       name='wgt_A_c'+str(self.cell_id),
                       dtype=tf.float64,
                       trainable=True
                      )

init_glorot_B = init_glorot(shape=(self.state_size,self.state_size),dtype=tf.float64)
#        init_glorot_B -= tf.math.reduce_mean(init_glorot_B)
self.B = tf.Variable(
                      initial_value=init_glorot_B,
                      name='wgt_B_c'+str(self.cell_id),
                      dtype=tf.float64,
                      trainable=True
                     )
'''
#### CALLABLE

'''
@tf.function
def odefunc_Ax_Bu(self,t,x,A,B,u):
    dx = tf.keras.activations.tanh(tf.matmul(A,x) + tf.matmul(B,u))
    return dx

@tf.custom_gradient
def map_to_state(self,A,x,B,u):
    
    x_t = self.odefunc_Ax_Bu(0, x, A, B, u)
    
    def grad(dL):
        
        
        t_0 = self.counter[0]
        t_1 = self.counter[0] + 1.0
        
        t = tf.linspace(t_0, t_1, num=8)
        _x = tfp.math.ode.DormandPrince().solve(self.odefunc_Ax_Bu,t_0,x,solution_times=t,constants={'A':A,'B':B,'u':u})

        dA = tf.reshape(tf.convert_to_tensor(tf.gradients(_x.states,A)),A.shape) * dL
        dx = tf.reshape(tf.convert_to_tensor(tf.gradients(_x.states,x)),x.shape) * dL
        dB = tf.reshape(tf.convert_to_tensor(tf.gradients(_x.states,B)),B.shape) * dL
        du = tf.reshape(tf.convert_to_tensor(tf.gradients(_x.states,u)),u.shape) * dL
        
        return dA, dx, dB, du
    
    return x_t, grad
'''

#### Gradient

'''
def grad(dL):
    
    t_0 = self.counter[0]
    t_1 = self.counter[0] + self.state_size
    
    t = tf.linspace(t_0, t_1, num=10)
    _dr, _dt = tfp.math.ode.DormandPrince().solve(self.odefunc_limit_cycle,t_0,x,solution_times=t,constants={'mu':mu,'theta':theta})
    
    drho = tf.reshape(tf.convert_to_tensor(tf.gradients(_dr.states,rho)),rho.shape) * dL
    dmu = tf.reshape(tf.convert_to_tensor(tf.gradients(_dr.states,mu)),mu.shape) * dL
    dtheta = tf.reshape(tf.convert_to_tensor(tf.gradients(_dt.states,theta)),theta.shape) * dL
    
    return drho, dmu, dtheta
'''
