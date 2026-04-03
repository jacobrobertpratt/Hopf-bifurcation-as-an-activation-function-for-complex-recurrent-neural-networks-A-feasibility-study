''' Special Imports '''
import os
import sys
import json
from datetime import datetime

if sys.platform == 'win32':
    from win32api import GetSystemMetrics
    screen_resolution = (GetSystemMetrics(0),GetSystemMetrics(1))
    screen_resolution_width = max(screen_resolution)
    screen_resolution_height = min(screen_resolution)
else:
    # Set default
    screen_resolution_width = 2560
    screen_resolution_height = 1440

''' Special Imports '''
# Linear Algebra
import numpy as np

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm

# AI/ML
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

''' ------------------------ MISCELLANEOUS ------------------------ '''

# TODO: build 'setkwargs' function to consolidate code

def runtime_dict():
    dt = datetime.now()    
    rt = {}
    rt['year'] = int(dt.strftime('%y'))
    rt['month'] = int(dt.strftime('%m'))
    rt['day'] = int(dt.strftime('%d'))
    rt['hour'] = int(dt.strftime('%H'))
    rt['min'] = int(dt.strftime('%M'))
    rt['sec'] = int(dt.strftime('%S'))
    return rt
    

''' ------------------------ PRINT FUNCTIONS ------------------------ '''

''' Pints the import version numbers for reference '''
def print_import_versions():
    print("version Used:")
    print("\t    Python:",sys.version)
    print("\t     Numpy:",np.__version__)
    print("\t    OpenCV:",cv2.__version__)
    print("\ttensorflow:",tf.__version__)
    print("\t     Keras:",keras.__version__)
    print("\n\n")


# TODO: implement a print(help()) function thing. -> help(func) prints to stdout the details of the object.



''' - - - - - TENSORFLOW PRINT FUNCTIONS - - - - - '''

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

# Prints complex and non-complex with and without specified index
def _cpx_print(idx=-1,arr=None,**kwargs):
    if (arr.dtype == tf.complex128) or (arr.dtype == tf.complex64) or (arr.dtype == tf.complex):
        if idx >= 0:
            tf.print(str(idx)+') ',tf.math.real(arr),'\t',tf.math.imag(arr))
        else:
            tf.print(
                'Real:\n',tf.math.real(arr),
                '\nImag:\n',tf.math.imag(arr),
                summarize = kwargs['summarize'] if 'summarize' in kwargs else None,
                sep = kwargs['sep'] if 'sep' in kwargs else ' ',
                end = kwargs['end'] if 'end' in kwargs else ' ',
            )
    else:
        if idx >= 0:
            tf.print(str(idx)+') ',arr)
        else:
            tf.print(arr)
    tf.print()


# Prints complex and non-complex values in full or in the condensed way.
def _print(msg='unknown array', arr=None, full=False, **kwargs):
    p_str = msg+':\n'
    if arr is None:
        tf.print(msg + ': array was None type')
        return
    if not full:
        _cpx_print(arr=arr,**kwargs)
    else:
        for r in range(arr.shape[0]):
            _cpx_print(idx=r,arr=arr[r],**kwargs)
    tf.print('type:',type(arr),'\nshape:',arr.shape,'  dtype:',arr.dtype,end='')
    try:
        tf.print('  name:',arr.name.split('\\')[-1].split('/')[-1],end='')
    except:
        pass
    tf.print('\n\n')




# Prints gradients by passing unchanged tensor through function and catching the
#   back-prop configured by tensorflow.
@tf.custom_gradient
def print_gradient(input):
    nothing = input
    def grad(dL):
        _print('Grad dL',dL)
        return dL
    return nothing, grad



''' ------------------------ PLOTTINATORS AND GRAPHINATORS ------------------------ '''

def plot(x_vals, y_vals, **kwargs):
    
    # Check input params #
    if not isinstance(y_vals,list): y_vals = [y_vals]
    
    # Generate colors for n-size y inputs
    y_len = len(y_vals)
    
    to_show = False
    if 'show' in kwargs: to_show = kwargs['show']
    assert isinstance(to_show,bool), 'input show parameter must be boolean'
    
    to_save = False
    if 'save' in kwargs: to_save = kwargs['save']
    assert isinstance(to_save,bool), 'input save parameter must be boolean'
    
    # Set figure specifics
    _dpi = 250
    if 'dpi' in kwargs: _dpi = kwargs['dpi']
    assert isinstance(_dpi,int), 'dpi parameter must be int type'
    
    fig_size = (6.4,4.8)
    if 'figsize' in kwargs: fig_size = kwargs['figsize']
    assert_str = 'figsize must be a 2-tuple of floats representing (width, height)'
    assert (isinstance(fig_size,list) or isinstance(fig_size,tuple)) and (len(fig_size)==2), assert_str
    assert isinstance(fig_size[0],float) and isinstance(fig_size[1],float), assert_str
    
    save_dir = './'
    save_name = 'plot2D_image_'+datetime.now().strftime('%y%m%d_%H%M%S')
    save_format = '.png'
    if to_save:
        if 'dir' in kwargs: save_dir = kwargs['dir']
        assert isinstance(save_format,str), 'format parameter must be string type.'
        
        if 'name' in kwargs: save_name = kwargs['name']
        assert isinstance(save_format,str), 'format parameter must be string type.'
        
        if 'format' in kwargs: save_format = kwargs['format']
        assert isinstance(save_format,str), 'format parameter must be string type.'
        assert_str = 'save format options are \'.jpg\', \'.png\', \'.pdf\', or \'.svg\', default is \'.png\.'
        assert  (save_format == '.png') or (save_format == '.jpg') or (save_format == '.svg') or (save_format == '.pdf'), assert_str
        
    # Create plot paramters -> default is rainbow
    cols = ['#ff0000','#ffa500','#ffff00','#008000','#0000ff','#4b0082','#ee82ee']
    if 'colors' in kwargs: cols = kwargs['colors']
    if isinstance(cols,str): cols = [cols]
    
    # Check that the input colors list contains rgb or rgba strings
    assert isinstance(cols,list) and isinstance(cols[0],str), 'input colors must be \'#rrggbb\' or \'#rrggbb\' type values.'
    c_len = len(cols)    # Get color list length for looping
    
    line_styles = ['-','--','-.',':']
    if 'linestyles' in kwargs: line_styles = kwargs['linestyles']
    assert isinstance(line_styles,list), 'linestyles input argument must be list type.'
    for _ele in line_styles:
        assert isinstance(_ele,str), 'all elements in linestyles must be string types.'
    s_len = len(line_styles)
    
    style_1to1 = False
    if s_len == y_len: style_1to1 = True
    
    # Create line specifics
    line_width = 0.75
    if 'linewidth' in kwargs: line_width = kwargs['linewidth']
    assert isinstance(line_width,float), 'line width must be float type'
    
    title = ''
    if 'title' in kwargs: title = kwargs['title']
    assert isinstance(title,str), 'title parameter must be string type'
    
    x_label = ''
    if 'xlabel' in kwargs: x_label = kwargs['xlabel']
    assert isinstance(x_label,str), 'xlabel parameter must be string type'
    
    y_label = ''
    if 'ylabel' in kwargs: y_label = kwargs['ylabel']
    assert isinstance(y_label,str), 'ylabel parameter must be string type'
    
    legend = []
    if 'legend' in kwargs: legend = kwargs['legend']
    assert isinstance(legend,list), 'legend parameter must be string list'
    
    # Create plot #
    fig = plt.figure(figsize=fig_size,dpi=_dpi)
    ax = fig.add_subplot(1,1,1)
    
    l = -1
    for i in range(y_len):
        c = (i % c_len)             # Increment color used
        if not style_1to1:
            if c == 0:l=(l+1)%s_len     # Increment line-style for each list over cols length
            lstyle = line_styles[l]
        else:
            lstyle = line_styles[i]
        ax.plot(
                 x_vals,
                 y_vals[i],
                 color=cols[c],
                 linestyle=lstyle,
                 linewidth=line_width
                )
    
    # Print the legend if needed.
    if len(legend) > 0: ax.legend(legend)
    
    if len(title) > 0: plt.title(title)
    if len(x_label) > 0: plt.xlabel(x_label)
    if len(y_label) > 0: plt.ylabel(y_label)
    
    # Save the figure
    if to_save:
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        pathname = os.path.join(save_dir,save_name+save_format)
        plt.savefig(pathname,dpi=_dpi)
        
    # Plot the figure
    if to_show:
        plt.show()


''' ------------------------ PRINT FUNCTIONS ------------------------ '''


def readme_writer(filename='',filedir='',**kwargs):
    
    if not os.path.exists(filedir): os.makedirs(filedir)
    
    write_str = ''
    
    summary = ''
    if 'summary' in kwargs:
        summary = kwargs['summary']
        del kwargs['summary']
    
    for lbl, val in kwargs.items():
        splt = ': '
        if len(lbl) == 0: splt = ''
        write_str += lbl.capitalize() + splt + str(val) + '\n'
    
    if len(summary) > 0: write_str = summary + '\n' + write_str
    
    with open(os.path.join(filedir,filename+'.txt'),'w') as fileobj:
        fileobj.write(write_str)
    
    return write_str


''' ------------------------ SAVEINATORS AND LOADINATORS ------------------------ '''

def save_meta(meta=None,dir=''):
    assert meta is not None, 'input meta dictionary was None type'
    assert len(meta) > 0, 'input meta dictionary was empty'
    assert isinstance(meta,dict), 'input meta must be dictionary object.'
    
    # Create directories if they don't exists
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # Convert input meta to json string
    json_str = json.dumps(meta)
    
    # Write meta.json to directory
    metafile = os.path.join(dir,'meta.json')
    with open(metafile,'w') as fileobj:
        fileobj.write(json_str)
    
    return fileobj.closed   # Returns True if files has been closed.
    
def load_meta(dir=''):
    
    metafile = os.path.join(dir,'meta.json')
    
    # Check meta.json file exists in directory
    if not os.path.exists(metafile):
        return {}   # Will be used to create a new dictionary

    # Read json string from file
    json_str = None
    with open(metafile,'r') as fileobj:
        json_str = fileobj.read()
    
    # Make json string into dictionary
    meta = json.loads(json_str)
    
    return meta



''' ------------------------ TENSORFLOW HELP FUNCTION ------------------------ '''

# Given a tf.Model and a 2-tuple of (images, labels), the function 
#   throws a warning if the models output shape is not compatible 
#   with the label data shape.
def ensure_output_shape(model,data):
    assert isinstance(data,tuple), 'input data must be tuple of (images, labels) data.'
    imgs, lbls = data
    tf.ensure_shape(model(tf.expand_dims(imgs[0],0),training=False),[None]+list(lbls.shape)[-2::])


@tf.function
def stack_to_complex(input_stack,dtype=tf.complex128):
    re_stk, im_stk = tf.unstack(input_stack)
    return tf.cast( tf.complex(re_stk, im_stk), dtype=dtype)


@tf.function
def complex_to_stack(input_complex,dtype=tf.float64):
    re_val, im_val = tf.math.real(input_complex), tf.math.imag(input_complex)
    return tf.cast(tf.stack([re_val, im_val]), dtype=dtype)

# Collects the trainable weights and saves them to a dictionary.
#   (Used to compare initial and final weight values.)
def build_weight_dict(model,wgtlst=[]):
    if len(wgtlst) == 0: return {}
    wgtdict = {}
    _wgtlst = wgtlst.copy()
    _vars = model.trainable_variables
    for i in range(len(_wgtlst)):
        name = _wgtlst[i]
        for var in _vars:
            if name in var.name:
                wgtdict[name] = np.asarray(var.read_value())
                break
    return wgtdict