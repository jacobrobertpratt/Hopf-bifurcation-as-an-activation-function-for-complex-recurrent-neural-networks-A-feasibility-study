''' Standard Imports '''
import os
import sys
import math
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

''' Special Imports '''
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import cm

    
def dft_clocks(size,input=None,show_roots=False):

    scale = 0.4
    
    if show_roots is True and size > 16:
        show_roots = False
    
    size = size
    
    fig, axes = plt.subplots()
    
    roots = np.exp(-2j*np.pi*np.arange(size) / size)
    dft = roots.reshape(-1,1) ** np.arange(size)
    
    cols = dft.shape[0]
    rows = dft.shape[1]
    
    # Map input to dft
    in_dft = None
    in_dft_sc = None
    colmap = None
    if input is not None:
        in_dft = dft * input
        mag = np.absolute(in_dft)
        col_dft = mag - np.min(mag)
        col_dft /= np.max(col_dft)
        colmap = cm.get_cmap('rainbow',8)(col_dft)
        in_dft_sc = in_dft / np.max(mag)
        in_dft_sc *= scale
    
    # Create grid
    centers = []
    for i in range(cols):
        for j in range(rows):
            centers.append((j,i))
    centers = np.asarray(centers).reshape(cols,rows,2)
    centers = np.flip(centers,0)
    
    # Plot Circles
    for i in range(cols):
        for j in range(rows):
            circle = plt.Circle(tuple(centers[i][j]),scale,fill=False)
            axes.add_artist(circle)

    # Plot hands of DFT clock
    _dft = dft * scale
    _roots = roots * scale
    for i in range(cols):
        for j in range(rows):
            x = centers[i][j][0]
            y = centers[i][j][1]
            r = _dft[i][j]
            if show_roots:
                for h in range(size):
                    rt = _roots[h]
                    plt.plot([x,x+rt.real],[y,y+rt.imag],'y--',alpha=0.5)
            plt.plot([x,x+r.real],[y,y+r.imag],'k-')

    # Plot FFT from input Map
    if in_dft_sc is not None:
        for i in range(cols):
            for j in range(rows):
                x = centers[i][j][0]
                y = centers[i][j][1]
                r = in_dft_sc[i][j]
                plt.plot([x,x+r.real],[y,y+r.imag],color=colmap[i][j],linestyle='-')

    if in_dft is not None:
        rowsum = np.round(np.sum(in_dft,axis=1)/size,8)
        print('row sum:\n',rowsum.reshape(-1,1),'\nshape:',rowsum.shape)
    
    
    plt.xlim(-1,rows)
    plt.ylim(-1,cols)
    axes.set_aspect(1)
    plt.axis('off')
    plt.show()
    
    return None

# if __name__ == "__main__":











