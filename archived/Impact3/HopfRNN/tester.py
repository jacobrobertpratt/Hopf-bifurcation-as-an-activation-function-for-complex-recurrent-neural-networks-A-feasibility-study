
# Standard Imports
import sys
from datetime import datetime

# Library Imports
import numpy as np
import scipy

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import rgb2hex

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import tensorflow_probability as tfp

from keras.utils import losses_utils

''' Local Imports'''
from utils import _print, _print_matrix



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


''' ... '''
if __name__ == "__main__":
    
    order = 12
    
    Q = np.arange(order, dtype=np.float64)
    _print('Q',Q)
    
    R = (2 * Q + 1)[:, None]
    _print('R',R)
    
    j, i = np.meshgrid(Q, Q)
    _print('j',j)
    _print('i',i)
    
    A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
    _print('A',A)
    
    B = (-1.0) ** Q[:, None] * R
    _print('B',B)
    
    exit(0)
    
    # Size of training set on the total MackeyGlass output set
    data_size = (4096+1024)*2
    
    x_glass, y_glass = np.array(get_mackey_glass(tao=22,delta_x=4,steps=data_size,plot=False))
    
    sz = 256*2
    
    x2_glass = np.arange(0,sz)
    y2_glass = y_glass[0:sz]
#    print('x2_glass:',x2_glass)
    print('y2_glass:',y2_glass,'\nshape:',y2_glass.shape,'  dtype:',y2_glass.dtype,'\n')
#    plt.plot(x2_glass,y2_glass,'b-')
#    plt.show()
    
    dft = ((np.exp((-2.j)*np.pi*np.arange(sz)/sz).reshape(-1,1)**np.arange(sz))/np.sqrt(sz))
    print('dft',dft,'\nshape:',dft.shape,'  dtype:',dft.dtype,'\n')
    
    fft = dft @ y2_glass.reshape(-1,1)
#    print('fft',fft,'\nshape:',dft.shape,'  dtype:',dft.dtype,'\n')
    
    # IDEA: Replace this with hopf-bifurcation #
    ave = np.mean(np.abs(fft))*(np.std(np.abs(fft))*2)
    fft[np.abs(fft) < ave] = 0
    print('fft',fft,'\nshape:',dft.shape,'  dtype:',dft.dtype,'\n')
    
    # INVERSE #
    y = np.conjugate(fft).T @ dft
#    print('y',y,'\nshape:',dft.shape,'  dtype:',dft.dtype,'\n')
    y = y[0].T
    
    plt.plot(x2_glass,y2_glass,'b-')
    plt.plot(x2_glass,y,'r--')
    plt.show()
    
    '''
    fft = np.fft.fft(y2_glass)
    print('fft',fft)
    ave = np.mean(np.abs(fft))
    print('ave',ave)
    
    fft[np.abs(fft) < ave] = (0.+0.j)
    print('fft',fft)
    ifft = np.fft.ifft(fft)
    print('ifft',ifft,'\nshape:',ifft.shape,'  dtype:',ifft.dtype,'\n')
    '''
    
#    fft = np.concatenate([fft[0:9],fft[-8::]])
#    print('fft',fft,'  shape:',fft.shape,'  dtype:',fft.dtype,'\n')
#    fft = np.expand_dims(fft,1)
#    print('fft',fft,'  shape:',fft.shape,'  dtype:',fft.dtype,'\n')
#    exp = np.exp(-2.j*np.pi*np.angle(fft))
#    print('exp',exp,'  shape:',exp.shape,'  dtype:',exp.dtype)
#    mat = (exp ** np.arange(0,256))
#    print('mat',mat,'  shape:',mat.shape,'  dtype:',mat.dtype)
#    map = ((fft.T @ mat)/(2*fft.shape[0]))[0]
#    print('map',map,'  shape:',map.shape,'  dtype:',map.dtype)
#    plt.plot(x2_glass,y2_glass,'r-')
#    plt.plot(x2_glass,map,'b-')
#    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    