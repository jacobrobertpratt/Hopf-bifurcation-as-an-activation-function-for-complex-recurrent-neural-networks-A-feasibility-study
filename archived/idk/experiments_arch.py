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

from data import MackeyGlassGenerator, seqMNISTGenerator, psMNISTGenerator, CopyMemoryGenerator

from model_hopf import HopfLayer, HopfModel

from models import ModelTrainer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

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
def build_weight_dict(model,name_list=[]):
    out_dict = {}
    for wgt in model.get_layer(index=1).weights:
        w = wgt.read_value().numpy()
        for nme in name_list:
            if nme in wgt.name:
                out_dict[nme] = w.copy()
    return out_dict

class MyOptimizer(tf.keras.optimizers.RMSprop):

    def __init__(self, **kwargs):
        super(MyOptimizer,self).__init__(name='MyOptimizer',**kwargs)
        self.count = tf.Variable([0],dtype=tf.int32)
    
    def update_step(self, gradient, variable):
        
        '''Update step given gradient and the associated model variable.'''
        lr = tf.cast(self.learning_rate, variable.dtype)
        
        if 'unt' in variable.name:
            # Special for Unitary #
            variable.assign(variable @ tf.linalg.expm(-lr*gradient))
        
        elif ('hrm' in variable.name):
            # Regular SGD for hermitian #
            variable.assign_add(-lr*gradient)
        
        # Update counter #
        self.count.assign_add([1])

def experiments():
    
    def hopf_callback(input_shape, train_dir):
        
        name='hopf'
        
        input_shape = input_shape[-2::]
        input = tf.keras.Input(shape=input_shape,dtype=tf.float64)
        hopf_layer_1 = HopfLayer(name='hopf_layer_1',dtype=tf.float64)(input)
        model = HopfModel(input=input, output=hopf_layer_1)
       
        optimizer = tf.keras.optimizers.RMSprop()
        loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO,name='mse')
        
        chkpnt_dir = os.path.join(train_dir,name,'checkpoint/cp.ckpt')
        loss_chkpnt = tf.keras.callbacks.ModelCheckpoint(
                                                          filepath=chkpnt_dir,
                                                          save_weights_only=True,
                                                          monitor=loss,
                                                          mode='min',
                                                          save_best_only=False
                                                         )
        
        # Setup traing and evaluation callbacks
        callbacks=[]
        callbacks.append(loss_chkpnt)
        
        model.compile(loss=loss, optimizer=optimizer, metrics=[loss])
        model.summary()
        
        return model, callbacks
    
    def basic_rnn_callback(input_shape, train_dir):
        
        name='basic_rnn'
        
        input_shape = input_shape[-2::]
        input = tf.keras.Input(shape=input_shape,dtype=tf.float64)
        output = tf.keras.layers.SimpleRNN(units=input_shape[-1])(input)
        model = tf.keras.Model(inputs=input, outputs=output)
        
        optimizer = tf.keras.optimizers.RMSprop()
        loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO,name='mse')
        
        chkpnt_dir = os.path.join(train_dir,name,'checkpoint/cp.ckpt')
        loss_chkpnt = tf.keras.callbacks.ModelCheckpoint(
                                                          filepath=chkpnt_dir,
                                                          save_weights_only=True,
                                                          monitor=loss,
                                                          mode='min',
                                                          save_best_only=False
                                                         )
        
        # Setup traing and evaluation callbacks
        callbacks=[]
        callbacks.append(loss_chkpnt)
        
        model.compile(loss=loss, optimizer=optimizer, metrics=[loss])
        model.summary()
        
        return model, callbacks
    
    do_work = False
    
    ''' ----- INPUT FEATURE PARAMTERS FOR MODEL ----- '''
    
    # Testing purposes only -> True => only test, no validate
    do_work = True
    
    # Model params
    input_size = 64
    output_size = 1
    
    # Train params
    batch_size = 1
    epoch_size = 5
    num_epochs = 2
    step_size = 1
    test_split = 0.25       # as a percent(%) of training parameters
    print_loss = True
    do_self_test = False
    print_weights = True
    
    wgt_names = ['U','V']    # Weight names to collect
    
    
    ''' TRAIN MODEL '''
    trainer = ModelTrainer(
                            data_callback=MackeyGlassGenerator,
                            batch_size=batch_size,
                            epoch_size=epoch_size,
                            num_epochs=num_epochs,
                            input_size=input_size,
                            output_size=output_size,
                            step_size=step_size,
                            test_split=test_split,
                            name='mackeyglass'
                           )
    trainer.execute(model_callbacks=[hopf_callback,basic_rnn_callback])
    
#    rnn_model, rnn_history = trainer.train(basic_rnn_callback)
    
    # TESTING ONLY #
#    images, labels = test_data
#    test_data = (images[0:5],labels[0:5])
    # TESTING ONLY #
    
    ''' EVALUATE MODEL '''
#    rnn_results = trainer.validate()
    
    exit(0)
    '''
#    model.compile(loss=myloss,optimizer=myrms,metrics=[myloss]) #,myaccu])
#    print("model.summary:\n",model.summary(),'\n')
    
    # Collect the initial weights for testing
    ini_wgt_dict = build_weight_dict(model,wgt_names)
    
    mod_fit = model.fit(
                         train_data,
                         test_data,
                         shuffle=False,
                         batch_size=1,              # Samples per gradient update
                         epochs=epoch_count,
                         steps_per_epoch=epoch_size,
                         callbacks=[loss_chkpnt]
                        )
    
#    tf.keras.backend.clear_session()
#    model.load_weights(checkpoint_dir)

    if print_loss is True:
        
        print('\nmod_fit.params:',mod_fit.params)
        print('mod_fit.history.keys():',mod_fit.history.keys())
        
        print('\n-------------------------- TEST and RESULTS ----------------------------------\n')
        
        # Loss History
        plt.plot(mod_fit.history['loss'])
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
        
        # Print Initial Weights
        for nme in wgt_names:
            if nme in ini_wgt_dict:
                wgt = ini_wgt_dict[nme].copy()
                print('Initial Matrix ['+nme+']:')
                print(wgt,'\nshape:',wgt.shape,'  dtype:',wgt.dtype,'\n')
    #            _cpx_eigvals('U',wgt)
    #            _cpx_pdet('U',wgt)
    #            _cpx_ptrace('U',wgt)
    #            Ichk_U = wgt @ tf.math.conj(wgt).T
    #            print('Ichk ['+nme+']:\n',np.round(wgt,6),'\n')
            
        print('\n - - - - - - - - - - - - - - - - - - - - - - - -\n')
        
        # Print Final Weights
        for nme in wgt_names:
            if nme in fin_wgt_dict:
                wgt = fin_wgt_dict[nme].copy()
                print('Final Matrix ['+nme+']:')
                print(wgt,'\nshape:',wgt.shape,'  dtype:',wgt.dtype,'\n')
    #            _cpx_eigvals('U',wgt)
    #            _cpx_pdet('U',wgt)
    #            _cpx_ptrace('U',wgt)
    #            Ichk_U = wgt @ tf.math.conj(wgt).T
    #            print('Ichk ['+nme+']:\n',np.round(wgt,6),'\n')
    
    if do_work is True:
        exit(0)
    
    print('STARTING TESTS ... \n')
    
    cnt = 0
    max_valid_len = valid_data.shape[0]
    tst_len = 512 #max_valid_len
    print('Test Length:',tst_len)
    
    if tst_len > max_valid_len:
        print('Test Length:',tst_len,'is greater than max_valid_len:',max_valid_len,'  Setting to max_valid_len.')
        tst_len = max_valid_len
    
    # Float Prediction Results
    res = None
    
    # Loop for validation results
    if do_self_test is False:
        
        for v in valid_data:
            
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
            
            if (cnt % 10) == 0:
                print('Cnt ... '+str(cnt))
        
            
    else:
        
        v = input_valid_data[0]
        
        while(True):
            
#            print('\n---------------------------------------------------------------------------')
#            print(str(cnt)+')\nv:\n',v)
#            _print('v',v)
            
            # Get prediction
            pred = tf.get_static_value(model(tf.expand_dims(v,0)))
            print('pred:\n',pred,'\n')
            
            _pred = pred.flatten()
            if res is None:
                res = _pred.copy()
            else:
                res = np.concatenate([res,_pred.copy()])
            
            v = np.roll(v,-1)
            v[-1][-1] = _pred[0]
            
            # Exit if greater count is greater than test length
            if cnt >= tst_len:
                break
            cnt += 1
    
    truth_data = mkygls.get_truth().flatten()
#    print('Results:\n',res,'\nshape:',res.shape,'dtype:',res.dtype,'\n')
#    print('Valid:\n',valid,'\nshape:',valid.shape,'dtype:',valid.dtype,'\n')
    
    # Find smaller length of either results or valid data.
    plt_len = res.shape[0]
    if plt_len >= truth_data.shape[0]:
        plt_len = truth_data.shape[0]
        res = res[0:plt_len]
    else:
        truth_data = truth_data[0:plt_len]
    
    ave_res = np.mean(res)
    print('Average Results:',ave_res)
    
    ave_val = np.mean(truth_data)
    print('Average Ground Truth:',ave_val)
    
    # Enter if prediction output is real-valued
    if res is not None:
        x_axis = np.arange(plt_len)
        plt.plot(x_axis,truth_data,'b-')
        plt.plot(x_axis,res,'r--')
        p_ave = ave_res * np.ones_like(res)
        plt.plot(x_axis,p_ave,'g-')
        p_ave = ave_val * np.ones_like(res)
        plt.plot(x_axis,p_ave,'m--')
        plt.show()
    '''

''' TESTING SECTION ''' 
if __name__ == "__main__":
    
    experiments()
    
