
import sys
from datetime import datetime

import numpy as np
import scipy

import tensorflow as tf

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()



''' ---------------- '''
def printComplex(name,arr,print_shape=True):
    
    tf.print(name+':')
    tf.print(' real:\n',tf.math.real(arr))
    tf.print(' imag:\n',tf.math.imag(arr))
    if print_shape:
        tf.print('shape:',arr.shape,'    dtype:',arr.dtype)
    tf.print('\n')

''' ---------------- '''
def _print(_str,arr,print_return=True):
    
    tf.print()
    if isinstance(arr,list):
        tf.print(_str+':\n',arr,'\nshape:',len(arr),'type: (list)')
    elif tf.is_tensor(arr):
        if arr.dtype == tf.complex128:
            printComplex(_str,arr)
        else:
            tf.print(_str+':')
            tf.print(arr,'\nshape:',arr.shape,'  dtype:',arr.dtype,'  type:',type(arr))
    else:
        tf.print(_str+':\n',arr,'\ntype:',type(arr))
    
    if print_return is True:
        tf.print()

''' ---------------- '''
def projection_test_nxn(msg='',arr=None):
    if len(msg) == 0:
        msg='(unknown)'
    if arr is None:
        tf.print('[my_layer|projection_text_nxn] Error: input matrix cannot be None value.')
    else:
        proj = tf.math.reduce_sum(arr@arr-arr)
        if (proj.dtype is tf.complex64) or (proj.dtype is tf.complex128):
            tf.print('Projection Test '+msg+':',tf.math.real(proj),',',tf.math.imag(proj),'\n')
        else:
            tf.print('Projection Test '+msg+':',proj,'\n')

def matrix_properties(mat=None):
    
    if mat is None:
        print('[matrix_properties] Error: input matrix was None type.')
        return
    
    # Normal
    norm_tst = tf.math.argmax(tf.math.real(tf.math.abs(mat @ tf.math.conj(mat).T - tf.math.conj(mat).T @ mat)))
    if norm_tst > (1.e-10):
        print('Normal: True')
    else:
        print('Normal: False')
    
    

def _print_matrix(msg='',wgt=None,print_Ichk=False,print_inv=False,print_props=False):
    
    tf.print()
    
    if len(msg) == 0:
        msg = '(unknown)'
    else:
        msg = msg.upper()
    tf.print('_____________________'+msg+'_____________________')
    
    if wgt is None:
        tf.print('[print_matrix] Input Weight was None type.')
        return

    _print('Matrix',wgt)
    Ichk = tf.math.conj(wgt).T @ wgt
    diag_Ichk = tf.linalg.diag_part(Ichk)
    tf.print('Ichk:')
    if print_Ichk is True:
        tf.print('\treal:',tf.math.real(Ichk))
        tf.print('\timag:',tf.math.imag(Ichk))
        tf.print('shape:',Ichk.shape,'    dtype:',Ichk.dtype,'\n')
    tf.print('Ichk diag:')
    for chk in diag_Ichk:
        tf.print('      ',tf.math.real(chk),tf.math.imag(chk),'   abs:',tf.math.real(tf.math.abs(chk)))
    tf.print()
    evls_wgt = tf.linalg.eigvals(wgt)
    tf.print('Eigen Values:')
    for ev in evls_wgt:
        tf.print('      ',tf.math.real(ev),tf.math.imag(ev),'   abs:',tf.math.real(tf.math.abs(ev)))
    print()
    if (print_inv is True):
        wgt_inv = tf.linalg.inv(wgt)
    if print_props is True:
        matrix_properties(wgt)
    trc = tf.linalg.trace(wgt)
    det = tf.linalg.det(wgt)
    tf.print('\ndet:',tf.math.real(det),tf.math.imag(det),'    trace:',tf.math.real(trc),tf.math.imag(trc))
    

''' ---------------- '''
@tf.custom_gradient
def print_gradient(input):
    nothing = input
    def grad(dL):
        _print('Grad dL',dL)
        return dL
    return nothing, grad

