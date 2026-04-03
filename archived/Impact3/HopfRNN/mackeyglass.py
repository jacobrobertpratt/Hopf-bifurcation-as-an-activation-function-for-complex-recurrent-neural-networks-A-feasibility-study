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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

''' Custom Imports '''
#from lmu_layer import LMU
from my_layer import MyRNN, MyLoss, MyModel
#from my_layer_dev_cpx import MyRNN, MyLoss, MyModel
#from my_layer_feedforward_tested import MyRNN, MyLoss, MyModel
#from fourier import dft_clocks

# Print line number
from inspect import currentframe, getframeinfo  # USE: getframeinfo(currentframe()).lineno

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
            y_ = y[t] + delta * ((0.2*y[t])/(1+pow(y[t],10)) - 0.1*y[t])
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
            
            # Attempt to add layers.
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
    
        # Split the test sequence into examples.
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
        model = getModel( model_name=model_name , n_samples=n_samples , n_features=n_features )
        
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


class MyRMSProp(tf.keras.optimizers.RMSprop):

    def __init__(self,state_size, **kwargs):
        super(MyRMSProp,self).__init__(name='MyRMSProp',**kwargs)
        self.eye = tf.constant(np.eye(state_size),dtype=tf.float64)
    
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype))
        
        rms = self.get_slot(var, "rms")
        if self._momentum:
            mom = self.get_slot(var, "momentum")
            if self.centered:
                mg = self.get_slot(var, "mg")
                return tf.raw_ops.ResourceApplyCenteredRMSProp(
                                                                var=var.handle,
                                                                mg=mg.handle,
                                                                ms=rms.handle,
                                                                mom=mom.handle,
                                                                lr=coefficients["lr_t"],
                                                                rho=coefficients["rho"],
                                                                momentum=coefficients["momentum"],
                                                                epsilon=coefficients["epsilon"],
                                                                grad=grad,
                                                                use_locking=self._use_locking
                                                                )
            else:
                return tf.raw_ops.ResourceApplyRMSProp(
                                                        var=var.handle,
                                                        ms=rms.handle,
                                                        mom=mom.handle,
                                                        lr=coefficients["lr_t"],
                                                        rho=coefficients["rho"],
                                                        momentum=coefficients["momentum"],
                                                        epsilon=coefficients["epsilon"],
                                                        grad=grad,
                                                        use_locking=self._use_locking
                                                       )
            
            rms_t = (coefficients["rho"] * rms + coefficients["one_minus_rho"] * tf.square(grad))
            rms_t = tf.compat.v1.assign(rms, rms_t, use_locking=self._use_locking)
            denom_t = rms_t
        
        if self.centered:
            tf.print('self.centered -> TRUE')
            mg = self.get_slot(var, "mg")
            mg_t = coefficients["rho"] * mg + coefficients["one_minus_rho"] * grad
            mg_t = tf.compat.v1.assign(mg, mg_t, use_locking=self._use_locking)
            denom_t = rms_t - tf.square(mg_t)
        
        # OLD ->  var_t = var - coefficients["lr_t"] * grad / (tf.sqrt(denom_t) + coefficients["epsilon"])
        if grad.dtype == tf.complex128:
            # standard Gradient Discent Optimization for rotation.
            _lr_t = tf.math.real(coefficients["lr_t"])
            c_lr_t = tf.complex(tf.math.cos(_lr_t),tf.math.sin(_lr_t))
            
            var_t = tf.math.log(var) - coefficients["lr_t"] * tf.math.log(grad)
    #        var_t = tf.math.log(var) - c_lr_t * tf.math.log(grad)
            var_t = tf.math.exp(var_t)
            
        else:

            if ('wgt' in var.name):
                var_t = var - (0.001 * grad)
            else:
                var_t = var - coefficients["lr_t"] * grad / (tf.sqrt(denom_t) + coefficients["epsilon"])
        
        return tf.compat.v1.assign(var, var_t, use_locking=self._use_locking).op

def FFTMackeyGlassTrainer():
    
    #command_start_spacer()
    
    model_name = 'rnn_ffs'
    save_dir = 'results/models/' + model_name
    checkpoint_dir = save_dir + '/checkpoints/'
    checkpoint_dir = os.path.dirname(checkpoint_dir + "training_1/cp.ckpt")
    
    # Size of training set on the total MackeyGlass output set
    data_size = (4096+1024)*8
    
    # Get MackeyGlass data
    x_glass, y_glass = np.array(get_mackey_glass(tao=22,delta_x=4,steps=data_size,plot=False))
#    print('y_glass:\n',y_glass,'\nshape:',y_glass.shape,'  dtype:',y_glass.dtype,'\n')
    
    
    
    # Get size of testing and training
    valid_sz = int(y_glass.shape[0] / 4)
    train_sz = data_size - valid_sz
#    print('valid_sz',valid_sz)
#    print('train_sz',train_sz)
#    exit(0)
    
    # Split MackeyGlass data into training and tests sets
    train_data = y_glass[0:train_sz]
    valid_data = y_glass[train_sz:data_size]
    
    ''' ----- PARAMETERS TO CHANGE ----- '''
    feature_size = 32
    sample_size = 1
    sample_skip = int(feature_size)
    
    cell_size = feature_size
    
    ''' ----- PARAMETERS TO CHANGE ----- '''
    train_size = train_data.shape[0] - (train_data.shape[0] % feature_size)
    train = train_data[0:train_size]
    tbsz = int(train_size-feature_size) # training batch size
    
    # Non-overlapping input series #
    train = train[0:train_size]
    tbatch = []
    t_batch_size = train_size//feature_size
    for j in range(t_batch_size):
        s = j * feature_size
        e = s + feature_size
        tbatch.append(list(train[s:e]))
        t_batch_size+=1
    train = np.asarray(tbatch)
    test = train[1::].copy()
    train = train[0:-1].copy()
    train = np.reshape(train,[train.shape[0],1,train.shape[1]])
    test = np.reshape(test,[test.shape[0],1,test.shape[1]])
#    print('train:\n',train,'\nshape:',train.shape,'  dtype:',train.dtype,'\n')
#    print('test:\n',test,'\nshape:',test.shape,'  dtype:',test.dtype,'\n')
#    exit(0)
    
    # Setup validation data
    valid_size = valid_data.shape[0] - (valid_data.shape[0] % feature_size)
    valid = valid_data[0:valid_size]
    vbsz = valid.shape[0]//feature_size
    
    # Non-overlapping input series #
    vbatch = []
    for j in range(vbsz):
        s = j * feature_size
        e = s + feature_size
        vbatch.append(list(valid[s:e]))
    valid = np.asarray(vbatch)
    valid = np.reshape(valid,[valid.shape[0],1,valid.shape[1]])
#    print('valid:\n',valid,'\nshape:',valid.shape,'  dtype:',valid.dtype,'\n')
#    exit(0)
    
#    do_work = True
    do_work = False
    
    if do_work is True:
        epoch_size = 20
        num_epochs = 45
        print_weights = False
    else:
        epoch_size = 20
        num_epochs = 45
        max_valid_len = vbsz
        print_weights = True
        do_self_test = False
    
    ''' ----- MODEL ----- '''
    with tf.device('/cpu:0'):
        input = tf.keras.Input(shape=(sample_size, feature_size),dtype=tf.float64)
        rnn1 = MyRNN(cell_size=cell_size,sample_skip=sample_skip,name=model_name+'_1',dtype=tf.float64)(input)
        model = MyModel(input=input, output=rnn1)
    ''' ----- MODEL ----- '''
    
    # OPTIMIZATION FUNCTION #
    
#    myrms = MyRMSProp(feature_size)
    myrms = tf.keras.optimizers.RMSprop()
    
    # LOSS FUNCTION #
#    myloss = None
#    myloss = MyLoss()
    myloss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE,name='mean_squared_error')
    
    model.compile(loss=myloss,optimizer=myrms,metrics=['loss']) #,'accuracy'])
    print("model.summary:\n",model.summary(),'\n')
    
    mod_init_weights = (model.get_layer(index=1).get_weights())[:]
    
    chkpnt = tf.keras.callbacks.ModelCheckpoint(
                                                 filepath=checkpoint_dir,
                                                 save_weights_only=True,
                                                 monitor='loss',
                                                 mode='min',
                                                 save_best_only=False
                                                )
    tf.keras.backend.clear_session()
    
    mod_fit = model.fit(
                         train,
                         test,
                         shuffle=False,
                         batch_size=1,
                         epochs=num_epochs,
                         steps_per_epoch=epoch_size,
                         callbacks=[chkpnt]
                        )
    
    tf.keras.backend.clear_session()
    model.load_weights(checkpoint_dir)
    
    
    if print_weights is True:
        
        print('\n\n\n\n\n\n\n\n')
        print('------------------------------- MODEL WEIGHTS -------------------------------\n\n')
        
        rnd = 10
        
        # Print initialized weights
        print('INITIAL WEIGHTS:\n')
        for mw in mod_init_weights:
            if (len(mw.shape) > 1) and (mw.shape[-1] == mw.shape[-2]):
                _mw = np.round(mw,rnd)
                print('real:\n',_mw.real,'\n')
                print('imag:\n',_mw.imag,'\n')
                break
        
        # Print final learned weights
        print('FINAL WEIGHTS:\n')
        mod_final_weights = np.asarray(model.get_layer(index=1).get_weights()).copy()        
        for mw in mod_final_weights:

            if (len(mw.shape) > 1)  and (mw.shape[-1] == mw.shape[-2]):
                _mw = np.round(mw,rnd)
                print('real:\n',_mw.real,'\n')
                print('imag:\n',_mw.imag,'\n')
                break
        
    if do_work is True:
        exit(0)
    
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
    
    print('STARTING TESTS ... \n')
    
    cnt = 0
    tst_len = vbsz
    print('Test Length:',tst_len)
    
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
    
    dot = 0
    
    # Loop for validation results
    if do_self_test is False:
        
        for v in valid:
            
            # Get prediction
            pred = tf.get_static_value(model(tf.expand_dims(v,0))).flatten()
            if res is None:
                res = pred.copy()
            else:
                res = np.concatenate([res,pred.copy()])
            
            # Exit if greater count is greater than test length
            if cnt >= tst_len:
                break
            cnt += 1
            
    else:
        
        v = valid[0]
        
        while(True):
            
#            print('\n---------------------------------------------------------------------------')
#            print(str(cnt)+')\nv:\n',v)
            
            # Get prediction
            pred = tf.get_static_value(model(tf.expand_dims(v,0)))
            
#            print('pred:\n',pred,'\n')
            
            _pred = pred.flatten()
            if res is None:
                res = _pred.copy()
            else:
                res = np.concatenate([res,_pred.copy()])
            
            # Exit if greater count is greater than test length
            if cnt >= tst_len:
                break
            cnt += 1
            
            v = np.reshape(pred,v.shape)
        
    ave_res = np.mean(res)
    print('Average Results:',ave_res)
    
    print('Results:\n',res,'\nshape:',res.shape,'dtype:',res.dtype,'\n')
    
    valid = valid.flatten()
    print('Valid:\n',valid,'\nshape:',valid.shape,'dtype:',valid.dtype,'\n')
    
    # Enter if prediction output is real-valued
    if res is not None:
        sz = res.shape[0]
        true = valid[0:sz]
        ave_val = np.mean(true)
        print('Average Valid:',ave_val)
        if sz > true.shape[0]:
            sz = true.shape[0]
            res = res[0:sz]
        x_axis = np.arange(sz)
        plt.plot(x_axis,true,'b-')
        plt.plot(x_axis,res,'r--')
        p_ave = ave_res * np.ones_like(res)
        plt.plot(x_axis,p_ave,'g-')
        p_ave = ave_val * np.ones_like(res)
        plt.plot(x_axis,p_ave,'m--')
        plt.show()
    

''' TESTING SECTION ''' 
if __name__ == "__main__":
    
    #print_import_versions()
    
#    MackeyGlassTrainer()
    FFTMackeyGlassTrainer()
    
    #ortho = random_orthogonal_initializer(shape=(6,6),connect_percent=0.20,value_range=(-0.4,0.4),dtype="float32")
    #print(ortho)










