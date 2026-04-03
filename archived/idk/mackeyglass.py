''' Standard Imports '''
import os
import sys
import math
import random
from datetime import datetime


''' Special Imports '''
import numpy as np
import scipy
from scipy import signal
from scipy.stats import unitary_group

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import rgb2hex

import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

''' Local Imports'''
from utils import _print, _print_matrix

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

''' Custom Imports '''
#from lmu_layer import LMU
from my_layer import MyRNN, MyLoss, MyModel
#from my_layer_dev_cpx import MyRNN, MyLoss, MyModel
#from my_layer_feedforward_tested import MyRNN, MyLoss, MyModel
#from fourier import dft_clocks

# Print line number
from inspect import currentframe, getframeinfo  # USE: getframeinfo(currentframe()).lineno

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

''' Pints the import version numbers for reference '''
def print_import_versions():
    print("version Used:")
    print("\t    Python:",sys.version)
    print("\t     Numpy:",np.__version__)
    print("\t    OpenCV:",cv2.__version__)
    print("\ttensorflow:",tf.__version__)
    print("\t     Keras:",keras.__version__)
    print("\n\n")

''' '''
def printComplex(name,arr,print_shape=True):
    tf.print('\n',name+':')
    tf.print('\treal:',tf.math.real(arr))
    tf.print('\timag:',tf.math.imag(arr))
    if print_shape:
        tf.print('shape:',arr.shape,'    dtype:',arr.dtype)
    tf.print('\n')


''' Calculate the spectral radius for a numpy array '''
def spectral_radius(nparr):
    comp_nparr = la.eigvals(nparr)
    cm_max = -1.0
    for i in range(len(comp_nparr)):
        cm = math.sqrt(pow(comp_nparr[i].real,2)+pow(comp_nparr[i].imag,2))
        if cm > cm_max:
            cm_max = cm
    return cm_max


''' '''
def set_plt_size(w,h,ax=None):
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


''' Set random edges don't repeat with specific number of connections '''
def random_weight_initializer(shape=(),connect_percent=0.20,value_range=(-0.4,0.4),dtype="float32"):
    cnt = math.ceil(shape[0] * shape[1] * connect_percent)
    arr = []
    k = 0
    for y in range(shape[1]):
        for x in range(shape[0]):
            arr.append((y,x))
            k+=1
    narr = np.asarray(arr)
    np.random.shuffle(narr)
    outarr = np.zeros((shape[0],shape[1]),dtype=dtype)
    i = 0
    while i < cnt:
        idx = random.randint(0,narr.shape[0] - 1)
        outarr[narr[idx][0],narr[idx][1]] = random.uniform(value_range[0],value_range[1])
        narr = np.delete(narr,idx,0)
        i+=1
    return tf.convert_to_tensor(outarr, dtype=dtype)
    
 
''' Set random edges don't repeat with specific number of connections '''
def random_orthogonal_initializer(shape=(),connect_percent=0.20,value_range=(-0.4,0.4),dtype="float32"):
    cnt = math.ceil(shape[0] * shape[1] * connect_percent)
    arr = []
    k = 0
    for y in range(shape[1]):
        for x in range(shape[0]):
            arr.append((y,x))
            k+=1
    narr = np.asarray(arr)
    np.random.shuffle(narr)
    
    outarr = np.eye(shape[1],dtype=dtype)
    print(outarr,"\n")
    outarr = np.roll(outarr,1,axis=0)
    # TODO: ... 
    return outarr


''' Produces the MackeyGlass dataset for input parameters '''
def get_mackey_glass(tao=30,delta_x=10,steps=600,plot=False,save=False,dir='./results/images/'):
    y = [0.2]
    x = [0]
    delta = 1/delta_x
    for t in range(steps):
        y_ = 0.0
        if t < tao:
            y_ = y[t] + delta * ((0.2 * y[t])/(1 + pow(y[t],10)) - 0.1 * y[t])
        else:
            y_ = y[t] + ((0.2 * y[t-tao])/(1 + pow(y[t-tao],10)) - 0.1 * y[t])
        y.append(y_)
        x.append(t)
    if plot is True:
        plt.plot(y)
        if save is True:
            tmp_dir = dir
            tmp_ax = plt.gca()
            if tmp_dir[len(tmp_dir)-1] != '/':
                tmp_dir += '/'
            title = 'Mackey Glass' + ' tao:' + str(tao) + ' delta:1/' + str(delta_x) + ' steps:' + str(steps)
            tmp_ax.set_title(title)
            title = title.replace(' ','_').replace(':','_').replace('/',':')
            tmp_dir += title
            plt.savefig(tmp_dir)
        plt.show()
    return x, y
   
    

''' Returns the specified model '''
def getModel(model_name="myrnn",batch_size=32,n_samples=1,n_features=32):

        # Build the model
        model = None #keras.models.Sequential()
        
        loss='mse'
        optimizer='adam'
        #metrics=["accuracy"]
        
        from my_layer import MyLoss
        
        if model_name == "myrnn":
            # My RNN Model
            
            #inpts = tf.keras.Input(shape=(n_samples,n_features))
            #myrrn_0 = MyRNN(n_features,name='myrnn_0')(inpts)
            #dens_0 = keras.layers.Dense(units=n_features,activation='tanh',name='dense_1')(myrrn_0)
            #model = tf.keras.Model(inputs=inpts, outputs=dens_0)
            
            model = keras.models.Sequential()
            # Important to add this Input & cast to tf.complex64 -> pipeline requires it.
            model.add(tf.keras.layers.InputLayer(input_shape=(n_samples, n_features),dtype=tf.complex64,name='input'))
            model.add(MyRNN(n_features,input_shape=(n_samples,n_features),name='myrnn'))
            model.add(tf.keras.layers.Dense(units=n_features,activation='tanh',name='dense'))
            
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            optimizer=tf.keras.optimizers.RMSprop()
            
        elif model_name == "lstm":
            # LSTM Model
            
            model = keras.models.Sequential()
            l2_size = (n_samples * 2)
            l3_size = (l2_size * 2)
            model.add(keras.layers.LSTM(units = n_samples,input_shape=(None,n_features), return_sequences = True, activation='tanh',name='lstm_1'))
            model.add(keras.layers.LSTM(units = l2_size,input_shape=(n_samples,n_features), return_sequences = True, activation='tanh',name='lstm_2'))
            model.add(keras.layers.LSTM(units = l3_size,input_shape=(l2_size,n_features), activation='tanh',name='lstm_3'))
            model.add(keras.layers.Dense(units=n_features,activation='sigmoid',name='dense_1'))
            
        elif model_name == "lmu":
            # LMU
            
            inpts = keras.Input(shape=(n_samples, n_features))
            layer_1 = LMU(  
                            memory_d=n_samples,
                            order=32,
                            theta=n_samples,
                            hidden_cell=keras.layers.SimpleRNNCell(units=n_samples),   # The larger units -> faster training drops
                            trainable_theta=True,
                            hidden_to_memory=True,
                            memory_to_memory=True,
                            input_to_hidden=False,
                            discretizer="zoh",
                            kernel_initializer="glorot_uniform",
                            recurrent_initializer="orthogonal",
                            kernel_regularizer=None,
                            recurrent_regularizer=None,
                            use_bias=False,
                            bias_initializer="zeros",
                            bias_regularizer=None,
                            dropout=0,
                            recurrent_dropout=0,
                          )(inpts)
            dens_1 = keras.layers.Dense(units=16,activation='tanh',name='dense_1')(layer_1)
            
            # TODO: Attempt to add layers.
            model = tf.keras.Model(inputs=inpts, outputs=dens_1)
            
            optimizer=tf.keras.optimizers.RMSprop()
            
        else:
            print("[mackeyglass|getModel] error: model name not defined.")
            return None

        model.compile(loss = loss) #, metrics = metrics)
        print("model.summary:\n",model.summary(),"\n\n")
		
        return model


def MackeyGlassTrainer():

    do_train = True
    self_test = False
    model_name = 'lmu'
    save_dir = 'results/models/' + model_name
    checkpoint_dir = save_dir + '/checkpoints/'

    batch_size = 32          # Batch size for training.
    
    if model_name == 'myrnn':
        n_epochs = 100
    else:
        n_epochs = 32             # Number of epochs to train for.
    
    # Size of training set on the total MackeyGlass output set
    train_sz = 1440
    
    # number of steps for input
    n_samples = 32
    n_features = 1  # Just the y axis prediction

    # Get MackeyGlass data
    x_glass, y_glass = np.array(get_mackey_glass(tao=22,delta_x=10,steps=2400,plot=False))
    y_glass = y_glass / np.max(y_glass)
    
    y_glass -= np.mean(y_glass)
    
    # Split MackeyGlass data into training and tests sets
    x_train = x_glass[1:train_sz]
    y_train = y_glass[1:train_sz]
    x_test = x_glass[train_sz:len(x_glass)]
    y_test = y_glass[train_sz:len(y_glass)]
    
    if do_train is True:
    
        # Split the test sequence into examples
        X, y = list(), list()
        y_trn_len = len(y_test)
        for i in range(y_trn_len):
            end_ix = i + n_samples
            if end_ix > (y_trn_len-2):
                break
            seq_x, seq_y = y_train[i:end_ix], y_train[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        X = np.array(X)
        y = np.array(y)
        
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        
        # Get the model
        model = getModel(model_name=model_name,n_samples=n_samples,n_features=n_features)
        
        # Keep the best model
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_dir,
                save_weights_only=True,
                monitor='loss',
                mode='min',
                save_best_only=True
            )
        
        # Fit th model by feeding it the set of inputs and the predicted output
        # fit_mod = model.fit(X, y, batch_size=batch_size, epochs=n_epochs, validation_split=0.4, callbacks=[model_checkpoint_callback])
        fit_mod = model.fit(X, y, batch_size=10, epochs=10, validation_split=0.4, callbacks=[model_checkpoint_callback])
        model.load_weights(checkpoint_dir)
        
        # Loss History
        plt.plot(fit_mod.history['loss'])
        plt.plot(fit_mod.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        
        # Save model
        model.save(save_dir)
        
        # Show prediction on test data
        y_tst_len = len(y_test)
        pred_y = []
        for i in range(n_samples):
            pred_y.append(0)
        for i in range(n_samples,y_tst_len):
            end_ix = i + n_samples
            if end_ix > (y_tst_len-1):
                break
            y_test_np = np.array(y_test)[i:end_ix]
            y_test_np = y_test_np.reshape((1, n_samples, n_features))
            yhat = model.predict(y_test_np)
            pred_y.append(yhat[0][0])

        x_glass_tst = x_glass[train_sz:train_sz+len(pred_y)]
        y_glass_tst = y_glass[train_sz:train_sz+len(pred_y)]

        adj = n_samples
        plt.plot(x_glass_tst[adj:len(x_glass_tst)],y_glass_tst[adj:len(x_glass_tst)],'b-')
        plt.plot(x_glass_tst[adj:len(x_glass_tst)],pred_y[0:len(x_glass_tst)-adj],'r--')
        plt.show()
        
    else:
    
        # Restore the model and construct the encoder and decoder.
        # TODO: load custom models keras.models.load_model("results/models/mg_uni_lstm",custom_objects=getModel())
        model = getModel(model_name=model_name,n_samples=n_samples,n_features=n_features)
        model.load_weights(checkpoint_dir)
        
        if self_test is True:
        
            # Show prediction on test data
            y_tst_len = len(y_test)
            pred_y = []
            j = 0
            for i in range(n_samples):
                pred_y.append(0)
            for i in range(n_samples,y_tst_len):
                end_ix = i + n_samples                
                if end_ix > (y_tst_len-1):
                    break
                if i < (2 * n_samples):
                    y_test_np = np.array(y_test)[i:end_ix]
                else:
                    y_test_np = np.array(pred_y)[j:n_samples+j]
                    j += 1
                y_test_np = y_test_np.reshape((1, n_samples, n_features))
                yhat = model.predict(y_test_np)
                pred_y.append(yhat[0][0])
        
        else:
            
            # Show prediction on test data
            y_tst_len = len(y_test)
            pred_y = []
            for i in range(n_samples):
                pred_y.append(0)
            for i in range(n_samples,y_tst_len):
                end_ix = i + n_samples
                if end_ix > (y_tst_len-1):
                    break
                y_test_np = np.array(y_test)[i:end_ix]
                y_test_np = y_test_np.reshape((1, n_samples, n_features))
                yhat = model.predict(y_test_np)
                pred_y.append(yhat[0][0])
        
        x_glass_tst = x_glass[train_sz:train_sz+len(pred_y)]
        y_glass_tst = y_glass[train_sz:train_sz+len(pred_y)]

        x_glass_tst = x_glass[train_sz:train_sz+len(pred_y)]
        y_glass_tst = y_glass[train_sz:train_sz+len(pred_y)]
        
        adj = n_samples
        plt.plot(x_glass_tst[adj:len(x_glass_tst)],y_glass_tst[adj:len(x_glass_tst)],'b-')
        plt.plot(x_glass_tst[adj:len(x_glass_tst)],pred_y[0:len(x_glass_tst)-adj],'r--')
        plt.show()

''' '''
@tf.function
def check_unitary_mat(_str='',p_mat=None):
    
    if len(_str) > 0:
        tf.print('Check Unitary '+_str+':\n -> shape:',p_mat.shape,'\n -> rank:',tf.rank(p_mat))
    else:
        tf.print('Check Unitary Mat:\n -> shape:',p_mat.shape,'\n -> rank:',tf.rank(p_mat))
    
    if p_mat.shape[-1] == 1:
        _mat = np.eye(p_mat.shape[-2],dtype=complex)
        _mat = tf.convert_to_tensor(_mat)
        _mat = tf.cast(_mat,dtype=p_mat.dtype)
        _mat *= p_mat
    else:
        _mat = p_mat
    
    _det = tf.abs(tf.linalg.det(_mat))
    tf.print(' ----> det:',_det,'\n')
    
    return _det

''' '''
def printComplex(name,arr,print_shape=True):
    tf.print('\n',name+':')
    tf.print('\treal:',tf.math.real(arr))
    tf.print('\timag:',tf.math.imag(arr))
    if print_shape:
        tf.print('shape:',arr.shape,'    dtype:',arr.dtype)
    tf.print('\n')


''' '''
def build_weight_dict(model,name_list=[]):
    out_dict = {}
    for wgt in model.get_layer(index=1).weights:
        w = wgt.read_value().numpy()
        for nme in name_list:
            if nme in wgt.name:
                out_dict[nme] = w.copy()
    return out_dict
    
class MyRMSProp(tf.keras.optimizers.RMSprop):

    def __init__(self,state_size, **kwargs):
        super(MyRMSProp,self).__init__(name='MyRMSProp',**kwargs)
        self.eye = tf.constant(np.eye(state_size),dtype=tf.complex128)
        self.in_sz = state_size
        self.count = tf.Variable([0],dtype=tf.int32)
    
    def update_step(self, gradient, variable):
        
        '''Update step given gradient and the associated model variable.'''
        lr = tf.cast(self.learning_rate, variable.dtype)
        
        var_key = self._var_key(variable)
        velocity = self._velocities[self._index_dict[var_key]]
        momentum = None
        if self.momentum > 0:
            momentum = self._momentums[self._index_dict[var_key]]
        average_grad = None
        if self.centered:
            average_grad = self._average_gradients[self._index_dict[var_key]]

        rho = self.rho

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            velocity.assign(rho * velocity)
            velocity.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - rho), gradient.indices
                )
            )
            if self.centered:
                average_grad.assign(rho * average_grad)
                average_grad.scatter_add(
                    tf.IndexedSlices(
                        gradient.values * (1 - rho), gradient.indices
                    )
                )
                denominator = velocity - tf.square(average_grad) + self.epsilon
            else:
                denominator = velocity + self.epsilon
            denominator_slices = tf.gather(denominator, gradient.indices)
            increment = tf.IndexedSlices(
                lr * gradient.values * tf.math.rsqrt(denominator_slices),
                gradient.indices,
            )

            if self.momentum > 0:
                momentum.assign(self.momentum * momentum)
                momentum.scatter_add(increment)
                variable.assign_add(-momentum)
            else:
                variable.scatter_add(-increment)
        else:
            
#            tf.print('count:',self.count)
            
            variable.assign(variable @ tf.linalg.expm(-lr*gradient))
            
            self.count.assign_add([1])
            
            ''' RMSPROP 
            # Dense gradients.
            velocity.assign(rho * velocity + (1 - rho) * tf.square(gradient))
            if self.centered:
                average_grad.assign(rho * average_grad + (1 - rho) * gradient)
                denominator = velocity - tf.square(average_grad) + self.epsilon
            else:
                denominator = velocity + self.epsilon
            increment = lr * gradient * tf.math.rsqrt(denominator)
            
            if self.momentum > 0:
                momentum.assign(self.momentum * momentum + increment)
                variable.assign_add(-momentum)
            else:
                variable.assign_add(-increment)
            '''

def FFTMackeyGlassTrainer():
    
    #command_start_spacer()
    
    model_name = 'rnn_ffs'
    save_dir = 'results/models/' + model_name
    checkpoint_dir = save_dir + '/checkpoints/'
    checkpoint_dir = os.path.dirname(checkpoint_dir + "training_1/cp.ckpt")
    
    
    # Size of training set on the total MackeyGlass output set
    data_size = (4096+1024)*16
    
    # Get MackeyGlass data
    x_glass, y_glass = np.array(get_mackey_glass(tao=22,delta_x=4,steps=data_size,plot=False))
#    print('y_glass:\n',y_glass,'\nshape:',y_glass.shape,'  dtype:',y_glass.dtype,'\n')
    
    # Get size of testing and training
    valid_sz = int(y_glass.shape[0] / 4)
    train_sz = y_glass.shape[0] - valid_sz
#    print('train_sz',train_sz)
#    print('valid_sz',valid_sz)
#    exit(0)
    
    # Split MackeyGlass data into training and tests sets
    train_data = y_glass[0:train_sz]
    valid_data = y_glass[train_sz:data_size]
    
    ''' ----- INPUT FEATURE PARAMTERS FOR MODEL ----- '''
    feature_size = 64
    sample_size = 1
    output_size = 1
    
    # Modify traind and valid lengths to conform to sample and feature sizes.
    # Train 
    train_length = train_data.shape[0] - (train_data.shape[0] % feature_size)
    train_data = train_data[0:train_length]
    # Test
    test_lenth = train_length
    test_data = train_data.copy()
    # Valid
    valid_length = valid_data.shape[0] - (valid_data.shape[0] % feature_size)
    valid_data = valid_data[0:valid_length]
    
    # AT THIS POINT OUR TRAINING AND VALIDATION DATA ARE DIVISIBLE BY OUR SAMPLE AND FEATURE SIZES #
    
    # Build training and testing data sets from our 'train_data' sets.
    
    # Training:
    max_train_len = train_length - feature_size
    to_skip = feature_size     # Cannot be less than 1.
    temp_batch = []
    for s in range(0,max_train_len,to_skip):
        e = s + feature_size
        temp_batch.append(list(train_data[s:e]))
    temp_batch = np.asarray(temp_batch)
    input_training_data = np.reshape(temp_batch,[temp_batch.shape[0],1,temp_batch.shape[1]])
    
    # Testing:
    # Must be the same size as our training batch size.
    input_test_data = input_training_data.copy()
    
    # Make test data and training data 1-to-1:
    input_training_data = input_training_data[0:-1]
    input_test_data = input_test_data[1::]
#    print('input_training_data:\n',input_training_data,'\nshape:',input_training_data.shape,'  dtype:',input_training_data.dtype,'\n')
#    print('input_test_data:\n',input_test_data,'\nshape:',input_test_data.shape,'  dtype:',input_test_data.dtype,'\n')

    # Validation
    max_valid_len = valid_length - feature_size
    to_skip = feature_size
    temp_batch = []
    for s in range(0,max_valid_len,to_skip):
        e = s + feature_size
        temp_batch.append(list(valid_data[s:e]))
    temp_batch = np.asarray(temp_batch)
    input_valid_data = np.reshape(temp_batch,[temp_batch.shape[0],1,temp_batch.shape[1]])
#    print('input_valid_data:\n',input_valid_data,'\nshape:',input_valid_data.shape,'  dtype:',input_valid_data.dtype,'\n')    
    
    print('Max Training Size:',input_training_data.shape[0])
    print('Max Test Size:',input_test_data.shape[0])
    print('Max Valdiaton Size:',input_valid_data.shape[0])
    
#    exit(0)
    
    wgt_names = ['U','W','z']
    
    do_work = True
#    do_work = False
    
    if do_work is True:
        epoch_size = 3
        num_epochs = 5
        print_weights = False
        print_loss = False
    else:
        epoch_size = 10
        num_epochs = 95
        print_loss = True
        print_weights = False
        do_self_test = False
        
    
    ''' ----- MODEL ----- '''
    with tf.device('/cpu:0'):
        input = tf.keras.Input(shape=(sample_size, feature_size),dtype=tf.float64)
        rnn1 = MyRNN(name=model_name+'_1',dtype=tf.float64)(input)
        model = MyModel(input=input, output=rnn1)
    ''' ----- MODEL ----- '''
    
    # OPTIMIZATION FUNCTION #
    
    myrms = MyRMSProp(feature_size)
#    myrms = tf.keras.optimizers.RMSprop()
    
    # LOSS FUNCTION #
    myloss = None
#    myloss = MyLoss()
#    myloss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE,name='mean_squared_error')
    
    model.compile(loss=myloss,optimizer=myrms,metrics=['loss']) #,'accuracy'])
    print("model.summary:\n",model.summary(),'\n')
    
    # Collect the initial weights for testing
    ini_wgt_dict = build_weight_dict(model,wgt_names)
    
    chkpnt = tf.keras.callbacks.ModelCheckpoint(
                                                 filepath=checkpoint_dir,
                                                 save_weights_only=True,
                                                 monitor='loss',
                                                 mode='min',
                                                 save_best_only=False
                                                )
    tf.keras.backend.clear_session()
    
    mod_fit = model.fit(
                         input_training_data,
                         input_test_data,
                         shuffle=False,
                         batch_size=1,
                         epochs=num_epochs,
                         steps_per_epoch=epoch_size,
                         callbacks=[chkpnt]
                        )
    
#    exit(0)
    
    tf.keras.backend.clear_session()
    model.load_weights(checkpoint_dir)
    
    if print_loss is True:
        
        print('\nmod_fit.params:',mod_fit.params)
        print('mod_fit.history.keys():',mod_fit.history.keys())
        
        print('\n-------------------------- TEST and RESULTS ----------------------------------\n')
        
        # Loss History
        plt.plot(mod_fit.history['loss'])
    #    plt.plot(mod_fit.history['accuracy'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    
    if print_weights is True:
    
        # Collect Final weights for testing
        fin_wgt_dict = build_weight_dict(model,wgt_names)
        
        # Test Final Weights
        
        # 1) See if they still form a unitary matrix after updates
        
        if 'U' in ini_wgt_dict:
            ini_U = ini_wgt_dict['U'].copy()
            print('Initial Matrix [U]')
            _cpx_eigvals('U',ini_U)
            _cpx_pdet('U',ini_U)
            _cpx_ptrace('U',ini_U)
            Ichk_U = ini_U @ tf.math.conj(ini_U).T
            print('Ichk [U]:\n',np.round(Ichk_U,6),'\n')
        if 'W' in ini_wgt_dict:
            ini_W = ini_wgt_dict['W'].copy()
            print('Initial Matrix [W]')
            _cpx_eigvals('W',ini_W)
            _cpx_pdet('W',ini_W)
            _cpx_ptrace('W',ini_W)
            Ichk_W = ini_W @ tf.math.conj(ini_W).T
            print('Ichk [W]:\n',np.round(Ichk_W,6),'\n')
            
        print('\n - - - - - - - - - - - - - - - - - - - - - - - -\n')
        
        if 'U' in fin_wgt_dict:
            fin_U = fin_wgt_dict['U'].copy()
            print('Final Matrix [U]')
            _cpx_eigvals('U',fin_U)
            _cpx_pdet('U',fin_U)
            _cpx_ptrace('U',fin_U)
            Ichk_U = fin_U @ tf.math.conj(fin_U).T
            print('Ichk [U]:\n',np.round(Ichk_U,6),'\n')
        if 'W' in fin_wgt_dict:
            fin_W = fin_wgt_dict['W'].copy()
            print('Final Matrix [W]')
            _cpx_eigvals('W',fin_W)
            _cpx_pdet('W',fin_W)
            _cpx_ptrace('W',fin_W)
            Ichk_W = fin_W @ tf.math.conj(fin_W).T
            print('Ichk [U]:\n',np.round(Ichk_W,6),'\n')
    
    if do_work is True:
        exit(0)
    
    print('STARTING TESTS ... \n')
    
    cnt = 0
    tst_len = 25 #max_valid_len
    print('Test Length:',tst_len)
    
    if tst_len > max_valid_len:
        print('Test Length:',tst_len,'is greater than max_valid_len:',max_valid_len,'  Setting to max_valid_len.')
        tst_len = max_valid_len
    
    # Complex Predict results
    a = 0.
    a_inc = 1./(1.+tst_len)     # Set alpha channel increments
    re_pred_arr = []
    im_pred_arr = []
    alph_arr = []
    colr_arr = []
    setcols = ['r','g','b']
    
    # Float Prediction Results
    res = None
    ave_res = 0.
    ave_val = 0.
    
    # Loop for validation results
    if do_self_test is False:
        
        for v in input_valid_data:
            
            # Get prediction
            pred = tf.get_static_value(model(tf.expand_dims(v,0))).flatten()
            if res is None:
                res = pred.copy()
            else:
                res = np.concatenate([res,pred.copy()])
#            print('pred:\n',pred,'\nshape:',pred.shape,'  dtype:',pred.dtype,'\n')
#            print('res:\n',res,'\nshape:',res.shape,'  dtype:',res.dtype,'\n')
#            if cnt > 3:
#                break
            
            # Exit if greater count is greater than test length
            if cnt >= tst_len:
                print('Breaking early with test length:',tst_len)
                break
            cnt += 1
            
            if (cnt % 10) == 0:
                print('Working ... '+str(cnt))
            
            
    else:
        
        v = input_valid_data[0]
        
        while(True):
            
#            print('\n---------------------------------------------------------------------------')
#            print(str(cnt)+')\nv:\n',v)
            
            # Get prediction
            pred = tf.get_static_value(model(tf.expand_dims(v,0))).flatten()
            if res is None:
                res = pred.copy()
            else:
                res = np.concatenate([res,pred.copy()])
            
            # Exit if greater count is greater than test length
            if cnt >= tst_len:
                print('Breaking early with test length:',tst_len)
                break
            cnt += 1
            
            v = np.reshape(pred,v.shape)
    
#    print('Results:\n',res,'\nshape:',res.shape,'dtype:',res.dtype,'\n')
    
    valid = valid_data.flatten()
#    print('Valid:\n',valid,'\nshape:',valid.shape,'dtype:',valid.dtype,'\n')
    
    # Adjust results to match with start of valid data
    res = np.concatenate([np.zeros(shape=(feature_size),dtype=valid.dtype),res])
    
    # Find smaller length of either results or valid data.
    plt_len = res.shape[0]
    if plt_len >= valid.shape[0]:
        plt_len = valid.shape[0]
        res = res[0:plt_len]
    else:
        valid = valid[0:plt_len]
    
    ave_res = np.mean(res)
    print('Average Results:',ave_res)
    
    ave_valid = np.mean(valid)
    print('Average Valid:',ave_valid)
    
    # Enter if prediction output is real-valued
    if res is not None:
        x_axis = np.arange(plt_len)
        plt.plot(x_axis,valid,'b-')
        plt.plot(x_axis,res,'r--')
        p_ave = ave_res * np.ones_like(res)
        plt.plot(x_axis,p_ave,'g-')
        p_ave = ave_valid * np.ones_like(res)
        plt.plot(x_axis,p_ave,'m--')
        plt.show()
    

''' TESTING SECTION ''' 
if __name__ == "__main__":
    
    #print_import_versions()
#    print("Num GPUs Available: ", tf.config.list_physical_devices())
#    exit(0)
    
#    MackeyGlassTrainer()
    FFTMackeyGlassTrainer()
    
    #ortho = random_orthogonal_initializer(shape=(6,6),connect_percent=0.20,value_range=(-0.4,0.4),dtype="float32")
    #print(ortho)










