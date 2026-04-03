import os

import numpy as np
from scipy.stats import unitary_group

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import tensorflow_probability as tfp

''' Local Imports '''
import proj_utils as utils
from proj_utils import _print
from activations import HopfBifur

'''
TODO:
- Add multiple cells
- Copy this code over to the use_RNN_class version to see if performance is different.
- Try and see what the output of the ODESover looks like and if we can use some regularizer
    on that data for the output state. 
- Make stuff into complex next.
'''


@tf.keras.utils.register_keras_serializable('prattrnn_layer')
class PrattRNNCell(tf.keras.layers.Layer):
    
    '''
    PrattRNNCell processes one timestep of a Hopf-Bifurcation recurrent network.
    That is the input is mapped with the internal weights, and updated 
    '''
    def __init__(
                    self,
                    size=32,
                    activation=tf.keras.activations.tanh,
                    regularizer=tf.keras.regularizers.L2,
                    **kwargs
                  ):
        
        super(PrattRNNCell, self).__init__(**kwargs)
        
        # Size of input cells
        self.state_size = size
        
#        self.activation = activation
        self.activation = HopfBifur()
        
        self.regularizer = regularizer
    
    def build(self, input_shape):
        
        super(PrattRNNCell,self).build(input_shape)
        
        if self.state_size is None: self.state_size = input_shape[-1]
        
#        ''' Creates a unitary weight matrix from
        def gen_unitary_weight_matrix(sz):
            wgt = unitary_group.rvs(sz)
            wgt = tf.linalg.logm(wgt)
            wgt = (wgt + tf.math.conj(wgt)[::-1,::-1])/2.
            wgt = wgt / tf.math.pow(tf.linalg.det(wgt),(1/sz))
            return tf.linalg.expm(wgt)
        
        # Creates a unitary weight matrix from 
        def gen_complex_random_weight_matrix(rows, cols=None):
            if cols is None: cols = rows
            wgt = (np.random.rand(rows, cols) + 1.j*np.random.rand(rows, cols))
            wgt = wgt / np.sqrt(rows * cols)
            return tf.cast(wgt, dtype=self.dtype)

        unitary_wgt_ini = gen_unitary_weight_matrix(input_shape[-1]) #,self.state_size)
        log_wgt_ini = tf.linalg.logm(unitary_wgt_ini)
        
        # Input Kernel #
        kern_wgt_ini = tf.cast(tf.math.real(unitary_wgt_ini), dtype=self.dtype)
        self.kernel = tf.Variable(
            initial_value=kern_wgt_ini,
            dtype=self.dtype,
            trainable=True,
            name='kernel'
        )
        
        # Recurrent Kernel
        rec_kern_wgt_ini = tf.cast(tf.math.imag(unitary_wgt_ini), dtype=self.dtype)
        self.recurrent_kernel = tf.Variable(
            initial_value=rec_kern_wgt_ini,
            dtype=self.dtype,
            trainable=True,
            name='recurrent_kernel'
        )
#       '''

        # Hidden theta state
        hidden_theta_init = tf.math.sin(2*np.pi*np.arange(self.state_size)/self.state_size)
        hidden_theta_init = tf.cast(hidden_theta_init, dtype=self.dtype)
        self.hidden_theta = tf.constant(
            hidden_theta_init,
            dtype=self.dtype
        )
        
        # theta_kernel
        theta_kernel_init = tf.math.sin(2*np.pi*np.arange(self.state_size)/self.state_size)
        theta_kernel_init = tf.cast(theta_kernel_init, dtype=self.dtype)
        self.theta_kernel = tf.Variable(
            initial_value=theta_kernel_init,
            dtype=self.dtype,
            trainable=True,
            name='theta_kernel'
        )
        
        self.ones = tf.constant(
            tf.ones(shape=input_shape),
            dtype=self.dtype
        )
        
        '''  Print Function for above varibles.
        _print('self.kernel',self.kernel)
        _print('self.recurrent_kernel',self.recurrent_kernel)
        _print('self.hidden_theta',self.hidden_theta)
        exit(0)
#        '''
        
        self.built=True


    def call(self, input_tuple, training=False):
        
#        _print('self.kernel',self.kernel)
#        _print('self.recurrent_kernel',self.recurrent_kernel)
#        _print('self.hidden_state',self.hidden_state)
        
        if tf.nest.is_nested(input_tuple):
            input, state = input_tuple[0], input_tuple[1]
#        _print('input',input)
#        _print('state',state)
        
        input = tf.cast(input,dtype=self.dtype)
        
        # Check for further nesting & un-nest if needed.
        prev_state = state[0] if tf.nest.is_nested(state) else state
#        _print('prev_state',prev_state)
        
        kern_map = tf.linalg.matmul(input, self.kernel)
#        _print('kern_map',kern_map)
        
        rec_kern_map = tf.linalg.matmul(prev_state, self.recurrent_kernel)
#        _print('rec_kern_map',rec_kern_map)
        
        new_state = tf.math.sqrt(
              tf.math.pow(tf.math.cos(kern_map),2)
            + tf.math.pow(tf.math.sin(rec_kern_map),2)
        )
#        _print('new_state',new_state)
        
        output = self.activation(new_state, tf.math.pow(new_state,2))
#        output = tf.math.cos(kern_map) + tf.math.sin(rec_kern_map)
#        output = tf.math.cos(kern_map) + tf.math.sin(rec_kern_map)
#        _print('output',output)
        
        tf.debugging.check_numerics(new_state,'') # Fails on nan values.
        
        # check for shape compatibility & re-nest if the previous one was nested.
        new_state = [new_state] if tf.nest.is_nested(state) else new_state
        
        return (output, new_state)
    
    
    def get_config(self):
        config = {
                    "units": self.units,
                    "activation": tf.keras.activations.serialize(self.activation)
                    }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))








@tf.keras.utils.register_keras_serializable('prattrnn_layer')
class PrattRNNLayer(tf.keras.layers.Layer):
    
    def __init__(
                   self,
                   size=None,
                   **kwargs
                  ):
        
        # Right now just an integer.
        self.size = size
        
        # To hold the cells
        self.cells = None
        
        super(PrattRNNLayer, self).__init__(**kwargs)
    
    
    def build(self, input_shape):
        
        super(PrattRNNLayer, self).build(input_shape)
        
        # Just 1 cell at the moment.
        if self.cells is None:
            self.cells = PrattRNNCell(size=self.size, dtype=self.dtype)
            self.cells.build(input_shape[1::])
        
        state_shape = [input_shape[0],1,self.size]
        ini_zeros = tf.zeros(shape=state_shape, dtype=self.dtype)
        self.states = tf.Variable(
            ini_zeros,
            trainable=False,
            name='layer_states'
        )
        
        '''  Print Function for above varibles.
        tf.print('input_shape',input_shape)
        _print('self.states',self.states)
        _print('self.states.dtype',self.states)
        tf.print('self.dtype',self.dtype)
        exit(0)
#        '''
        
        self.batch_size = input_shape[0]
        self.input_size = input_shape[-1]
        
        self.built = True
        
    
    def call(self, inputs, training=False):
        
#        tf.print('\n-----------------------------------------------------------------------------\n')
        inputs = tf.cast(inputs, dtype=self.dtype)
        
        # TODO: Apply forward mapping with state stuff here
        #       return_state must be True for this.
        #       need to ensure RNN, and PrattRNN cell are complexified.
        
#        _print('states',self.states)
        
#        ''' Perform the RNN operation on batch
        lastout, outputs, states = self.rnn_call(self.cells, inputs, self.states)

        '''
        outputs = self.cells.call((inputs,[self.states]),training=training)
        outputs = outputs[0] if tf.nest.is_nested(outputs) else outputs
#        '''
        
#        '''
#        _print('lastout',lastout)
#        _print('outputs',outputs)
#        _print('states',states)
#       '''
        
        self.states.assign(tf.reshape(states,self.states.shape))
        
        # TODO: outputs mapping
        outputs = outputs[0] if tf.nest.is_nested(outputs) else outputs
        
        return outputs
    
    
    
    def get_config(self):
        base_config = super(PrattRNNLayer,self).get_config()
        if 'cell' in base_config: del base_config['cell']
        return base_config



    @classmethod
    def from_config(self, config):
        return self(**config)


    #    @tf.function
    def rnn_call(self, cell, inputs, states):
        
        states = tf.squeeze(states,axis=1)
        nested_states = [states] if not tf.nest.is_nested(states) else states
        
        # Get cell's calling function
        cell_call_fn = ( cell.__call__ if callable(cell) else cell.call )
        
        # Callback for backend.rnn() operations.
        def step(step_inputs, step_states):
            step_states = step_states[0]
            output, new_states = cell_call_fn((step_inputs, step_states), training=True)
            if not tf.nest.is_nested(new_states): new_states = [new_states]
            return output, new_states
        
        lastout, outputs, states = tf.keras.backend.rnn(step, inputs, nested_states)

        _lastout = lastout[0] if tf.nest.is_nested(lastout) else lastout
#        _print('_lastout',_lastout)
        
        _outputs = outputs[0] if tf.nest.is_nested(outputs) else outputs
#        _print('_outputs',_outputs)
        
        _states = states[0] if tf.nest.is_nested(states) else states
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