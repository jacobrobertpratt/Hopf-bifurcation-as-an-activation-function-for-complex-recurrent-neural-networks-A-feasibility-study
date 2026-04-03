
# Standard Imports
import os
import sys
from datetime import datetime

# Library Imports
import numpy as np

import matplotlib.pyplot as plt

# Special Imports #
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.datasets import mnist

'''
TODO:
- Make so that datasets can be made into the tf.keras.dataset.Dataset() type class.
    - Use a generator for this
    - Can load information with this
    - Requires changes to the trainer.
'''


class DataGenerator:
    
    ''' DataGenerator '''
    def __init__( self,
                  **kwargs
                 ):
        
        # Holds the data generated from the generate() function (i.e., not manipulated in to size,batch,epochs,...,etc.)
        self.raw_data = None
        self.raw_size = None
        self.raw_shape = ()
        
        # Holds the training and test data for model
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        
        # 2-dim tuple of sizes, matching the training and testing, images and labels.
        self.shape = None
        
        # Ground truth data for plotting and confirmation
        self.gndtru = None
        
        # Training & Testing specific values
        self.input_size = 0
        if 'input_size' in kwargs:
            self.input_size = kwargs['input_size']
            del kwargs['input_size']
        assert self.input_size > 0, '\ninput_size must be set and greater than 0'
            
        self.output_size = 0
        if 'output_size' in kwargs:
            self.output_size = kwargs['output_size']
            del kwargs['output_size']
        assert self.output_size > 0, '\noutput_size must be set and greater than 0'
        
        self.batch_size = 0
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
            del kwargs['batch_size']
        assert self.batch_size > 0, '\nbatch_size must be set and greater than 0'
        
        self.epoch_size = 0
        if 'epoch_size' in kwargs:
            self.epoch_size = kwargs['epoch_size']
            del kwargs['epoch_size']
        assert self.epoch_size > 0, '\nepoch_size must be set and greater than 0'
        
        self.num_epochs = 0
        if 'num_epochs' in kwargs:
            self.num_epochs = kwargs['num_epochs']
            del kwargs['num_epochs']
        assert self.num_epochs > 0, '\nnum_epochs must be set and greater than 0'
        
        self.test_split = 0.25
        if 'test_split' in kwargs:
            self.test_split = kwargs['test_split']
            del kwargs['test_split']
        assert self.test_split > 0.0, '\ntest_split must be set and greater than 0'
        assert self.test_split < 1.0, '\ntest_split must be set and less than 1.0'
        
        # Check and set if we want to use datasets or just numpy arrays
        self.makedataset = False
        if ('makedataset' in kwargs) and isinstance(kwargs['makedataset'],bool):
            self.makedataset = kwargs['makedataset']
        
        # Set the dtype if not defined in subclasses
        self.dtype = np.complex128
        if 'dtype' in kwargs: self.dtype = kwargs['dtype']
        
        # Concatenates the name for subclasses
        self.name = 'DataGenerator'
        if 'name' in kwargs:
            self.name += '_' + kwargs['name']


    ''' DataGenerator '''
    def generate( self):
        # Placeholder for subclass
        return self


    ''' DataGenerator '''
    def set_data(self, train_images, train_labels, test_images, test_labels):
        
        # Tests for input parameters, shapes, matching class parameters, ... etc.
        if  (train_images is None) or (train_labels is None) or (test_images is None) or (test_labels is None):
            return None
        
        # Ensure data-set outer dimension is divisible by the batch-size
        # Total-size of data-set    -> Check types
        if isinstance(train_images, np.ndarray):
            # Check if image and label sets are divisible by the batch size; else, adjust them.
            train_remainder = train_images.shape[0] % self.batch_size
            test_remainder = test_images.shape[0] % self.batch_size
            if (train_remainder > 0) or (test_remainder > 0):
                new_train_size = train_images.shape[0] - train_remainder
                new_test_size = test_images.shape[0] - test_remainder
                train_images = train_images[0:new_train_size]
                train_labels = train_labels[0:new_train_size]
                test_images = test_images[0:new_test_size]
                test_labels = test_labels[0:new_test_size]
        
        # Set the final size of the datasets
        train_shape = (train_images.shape, train_labels.shape)
        test_shape = (test_images.shape, test_labels.shape)
        self.shape = (train_shape, test_shape)
        
        if self.makedataset is True:
            # Create Dataset objects from constructed data
            train_images = tf.data.Dataset.from_tensors(train_images,name=self.name+'_train_images')
            train_labels = tf.data.Dataset.from_tensors(train_labels,name=self.name+'_train_labels')
            test_images = tf.data.Dataset.from_tensors(test_images,name=self.name+'_test_images')
            test_labels = tf.data.Dataset.from_tensors(test_labels,name=self.name+'_test_labels')
            
            # Construct Batch from datasets if batch_size is set
            if self.batch_size > 0:
                train_images = train_images.batch(self.batch_size,drop_remainder=True)
                train_labels = train_labels.batch(self.batch_size,drop_remainder=True)
                test_images = test_images.batch(self.batch_size,drop_remainder=True)
                test_labels = test_labels.batch(self.batch_size,drop_remainder=True)
        
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        
        return self


    ''' DataGenerator '''
    def get_data(self):
        if (self.train_images is None) or (self.train_labels is None) or (self.test_images is None) or (self.test_labels is None):
            return ((None,None),(None,None))
        train = (self.train_images, self.train_labels)
        test = (self.test_images, self.test_labels)
        return (train, test)
    
    
    ''' DataGenerator '''
    def get_shape(self):
        return self.shape

    ''' DataGenerator '''
    def get_attributes(self,attr={}):
        attr['input_size'] = self.input_size
        attr['output_size'] = self.output_size
        attr['raw_size'] = self.raw_size
        attr['batch_size'] = self.batch_size
        attr['epoch_size'] = self.epoch_size
        attr['num_epochs'] = self.num_epochs
        attr['test_split'] = str(self.test_split*100)+'%'
        attr['makedataset'] = self.makedataset
        attr['dtype'] = self.dtype
        (trnimg,trnlbl),(tstimg,tstlbl)=self.shape
        attr['shape'] = '\n  Train Images: '+str(trnimg)+'\n  Train Labels: '+str(trnlbl)
        attr['shape'] += '\n  Test Images: '+str(tstimg)+'\n  Test Labels: '+str(tstlbl)
        return attr






class MackeyGlassGenerator(DataGenerator):


    ''' MackeyGlass Generator '''
    def __init__( self, **kwargs):
        
        super().__init__(**kwargs)
        
        self.raw_size = 4096
        self.raw_data = None
        
        self.tao = 30
        if 'tao' in kwargs:
            self.tao = kwargs['tao']
            del kwargs['tao']
        
        self.delta_x = 10
        if 'delta_x' in kwargs:
            self.delta_x = kwargs['delta_x']
            del kwargs['delta_x']
        
        self.step_size = 1
        if 'step_size' in kwargs:
            self.step_size = kwargs['step_size']
            del kwargs['step_size']


    ''' MackeyGlass Generator 
        Produces the MackeyGlass dataset for input parameters '''
    def mackeyglass(self, size=4096):
        if size != self.raw_size:
            self.raw_size = size
        mkygls = [0.2]
        delta = 1/self.delta_x
        for t in range(int(self.raw_size)):
            y_ = 0.0
            if t < self.tao:
                y_ = mkygls[t] + delta * ((0.2 * mkygls[t])/(1 + pow(mkygls[t],10)) - 0.1 * mkygls[t])
            else:
                y_ = mkygls[t] + ((0.2 * mkygls[t-self.tao])/(1 + pow(mkygls[t-self.tao],10)) - 0.1 * mkygls[t])
            mkygls.append(y_)
        self.raw_data = mkygls.copy()
        self.raw_size = len(self.raw_data)
        self.raw_shape = [self.raw_size]
        return self


    ''' MackeyGlass Generator '''
    def generate(self):
        
        # TODO: Transfer these to the lower code section.
        input_size=self.input_size
        output_size=self.output_size
        batch_size=self.batch_size
        epoch_size=self.epoch_size
        num_epochs=self.num_epochs
        step_size=self.step_size
        test_split=self.test_split
        
        # Make sure that input_size and output_size are nicely divisible by step_size
        assert (output_size % step_size) == 0, 'output_size must be divisible by step_size'
        
        ''' ADJUSTMENT CALCULATION NOTE:
            A calculation is implemented (@ ADJ_RAW_SIZE) to adjust the raw_data size; 
              such that, the final output training set matches the specified input parameters requirments.
              (i.e., we want     size = max(input_size,output_size) * epoch_size * num_epochs
                 total inputs to the model during training and testing)
             
                a = input_size
                b = output_size
                k = max(input_size,output_size) * epoch_size * num_epochs
                p = test_split
                s = step_size
                n = minimum(input_size, output_size)
                f = (2*(n-1) + |a - b| + k*(1 + p)) * s
                
                Therefore: f() is the 'size' paramter passed to the self.generator() function to construct
                 a raw dataset such that the output training and testing datasets are split correctly.
            
        '''
        
        # Forces the generator function to run (@ MKY_FUNC)
        if self.raw_data is None: self.raw_data = []
        
        # Set local input values if they haven't been
        step_size = max(step_size,1)
        
        max_inout = max(input_size,output_size)
        min_inout = min(input_size,output_size)
        
        # Check minimum size given only input over train sizes
        train_size = max_inout * epoch_size * num_epochs
        test_split = int(train_size * test_split)
        
        # Calculate required raw data size and size adjustment
        adj_size = 2*(min_inout - 1) + (max_inout - min_inout)              # ADJ_RAW_SIZE
        size = (train_size + test_split + adj_size)*step_size
        
        # Increase the raw data size if our current raw data is too small
        if size > len(self.raw_data): self.mackeyglass(size)                # MKY_FUNC
        
        # Create the total image data for training and testing
        tmp_raw_data = self.raw_data[0:-output_size].copy()  # Ignore the last output-size of raw data
        tmp_raw_size = len(tmp_raw_data)
        image_data = []
        for s in range(0,tmp_raw_size,step_size):
            e = s + input_size
            if e > tmp_raw_size: break
            image_data.append(tmp_raw_data[s:e].copy())
        image_data = np.asarray(image_data).astype(self.dtype)
        
        # Create the total label data for training and testing
        tmp_raw_data = self.raw_data[input_size::].copy()  # Ignore the first input-size of raw data
        tmp_raw_size = len(tmp_raw_data)
        label_data = []
        for s in range(0,tmp_raw_size,step_size):
            e = s + output_size
            if e > tmp_raw_size: break
            label_data.append(tmp_raw_data[s:e])
        label_data = np.asarray(label_data).astype(self.dtype)
        
        # Note: The above code to generate the image and label datasets #
        #       ensures each set has the same outer dimension.          #
        
        # Split into testing and training image data sets
        train_images = image_data[0:train_size]
        train_labels = label_data[0:train_size]
        test_images = image_data[train_size:train_size+test_split]
        test_labels = label_data[train_size:train_size+test_split]
        
        # Finally reshape the data to fit the input parameters and save to class variables
        train_images = np.reshape(train_images,[train_images.shape[0],1,train_images.shape[1]])
        train_labels = np.reshape(train_labels,[train_labels.shape[0],1,train_labels.shape[1]])
        test_images = np.reshape(test_images,[test_images.shape[0],1,test_images.shape[1]])
        test_labels = np.reshape(test_labels,[test_labels.shape[0],1,test_labels.shape[1]])

        skip = output_size // step_size
        self.gndtru = test_labels.copy()[0::skip].flatten()
        
        super().set_data(train_images,train_labels,test_images,test_labels)
        
        return self

    
    ''' MackeyGlass Generator '''
    # Takes in a set of predictions and outputs,
    #   a 2-tuple (ground truth array, predition array), where the two 
    #   arrays have the same shape. (used for plotting in the trainer save() function)
    def groundtruth_generator(self,preds,maxlen=-1):
        
        preds = preds.copy()
        
        if not isinstance(preds,np.ndarray):
            preds = np.asarray(preds)
        
        gndtru = self.gndtru
        if not isinstance(gndtru,np.ndarray):
            gndtru = np.asarray(gndtru)
        
        stp = self.step_size
        preds = preds[::,-stp::]
        
        gndtru = gndtru.flatten()
        preds = preds.flatten()
        
        trulen = gndtru.shape[0]
        prdlen = preds.shape[0]

        minlen = min(trulen,prdlen)        
        
        # Ensure we have the same length
        if trulen != prdlen:
            gndtru = gndtru[0:minlen]
            preds = preds[0:minlen]
        
        # Limits the maximum output. (Used for plotting.)
        if maxlen > 0 and minlen > maxlen:
            gndtru = gndtru[0:maxlen]
            preds = preds[0:maxlen]
        
        return (gndtru, preds)

    ''' DataGenerator '''
    def get_attributes(self,attr={}):
        attr['tao'] =self.tao
        attr['delta_x'] =self.delta_x
        attr['step_size'] =self.step_size
        return super().get_attributes(attr=attr)


'''
Le, Quoc V., Navdeep Jaitly, and Geoffrey E. Hinton. 
    “A Simple Way to Initialize Recurrent Networks of Rectified Linear Units.” 
    arXiv, April 7, 2015. <https://doi.org/10.48550/arXiv.1504.00941>.
'''
class seqMNISTGenerator(DataGenerator):
    
    ''' seqMNIST Generator '''
    def __init__( self,
                  **kwargs
                 ):
        super().__init__(**kwargs)
        
    ''' seqMNIST Generator '''
    def generate(self):
        (train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0],1,train_images.shape[1]*train_images.shape[2])
        train_labels = train_labels.reshape(train_labels.shape[0],1,1)
        test_images = test_images.reshape(test_images.shape[0],1,test_images.shape[1]*test_images.shape[2])
        test_labels = test_labels.reshape(test_labels.shape[0],1,1)
        
        self.raw_data = {
                          'train':{'images':train_images,'labels':train_labels},
                          'test':{'images':test_images,'labels':test_labels}
                         }
        
        '''
        Setup Data Example:
        https://github.com/nengo/keras-lmu/blob/main/docs/examples/psMNIST.ipynb
        '''
        
        return self

    ''' seqMNIST Generator '''
    def reshape(self):
        print('\nseqMNISTGenerator|reshape: NOT YET IMPLEMENTED!\n')
        return self





'''
Le, Quoc V., Navdeep Jaitly, and Geoffrey E. Hinton. 
    “A Simple Way to Initialize Recurrent Networks of Rectified Linear Units.” 
    arXiv, April 7, 2015. <https://doi.org/10.48550/arXiv.1504.00941>.
'''
class psMNISTGenerator(DataGenerator):
    
    ''' psMNIST Generator '''
    def __init__( self,
                  **kwargs
                 ):
        super().__init__(**kwargs)
        
    ''' psMNIST Generator '''
    def generate(self):
        (train_images, train_labels),(test_images,test_labels) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0],1,train_images.shape[1]*train_images.shape[2])
        train_labels = train_labels.reshape(train_labels.shape[0],1,1)
        test_images = test_images.reshape(test_images.shape[0],1,test_images.shape[1]*test_images.shape[2])
        test_labels = test_labels.reshape(test_labels.shape[0],1,1)
        
        self.raw_data = {
                          'train':{'images':train_images,'labels':train_labels},
                          'test':{'images':test_images,'labels':test_labels}
                         }
        
        # TODO: Do some type of permutation based on the LMU paper example.
        
        '''
            Setup Data Example:
            https://github.com/nengo/keras-lmu/blob/main/docs/examples/psMNIST.ipynb
            '''
        
        
        return self



'''
Dataset Source:
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. 
    Neural computation, 9(8):1735–1780, 1997.
    <https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext>
'''
class CopyMemoryGenerator(DataGenerator):
    
    ''' CopyMemory Generator '''
    def __init__(   self,
                    batch_size=100,
                    epoch_count=1000,
                    num_samples=1,
                    in_bits = 10,
                    out_bits = 8,
                    low_tol = 0.001,
                    high_tol = 1.0,
                    min_seq = 1,
                    max_seq = 20,
                    pad = 0.001,
                    dtype=np.float64
                 ):
        super().__init__()
        
        self.raw_size = epoch_count
        self.raw_data = None
        
        self.num_samples = batch_size
        self.in_bits = in_bits
        self.out_bits = out_bits
        self.low_tol = low_tol
        self.high_tol = high_tol
        self.min_seq = min_seq
        self.max_seq = max_seq
        self.pad = pad
        self.dtype = dtype
        
    ''' CopyMemory Generator '''
    def get_pattern(self):    
        '''
        Generated by:
        Graves A., Wayne, G. and Danihelka, I. (2014) ‘Neural Turing Machines’, 
            Neural and Evolutionary Computing, arXiv:1410.5401 [cs.NE].
            <https://github.com/ajithcodesit/lstm_copy_task>
        '''
        ti = []
        to = []
        for _ in range(self.num_samples):
            seq_len_row = np.random.randint(self.min_seq,self.max_seq+1)
            pat = np.random.randint(low=0,high=2,size=(seq_len_row,self.out_bits)).astype(dtype=self.dtype)
            pat[pat < 1] = self.low_tol
            pat[pat >= 1] = self.high_tol
            x = np.ones(((self.max_seq*2)+2,self.in_bits), dtype=pat.dtype) * self.pad
            x[1:seq_len_row+1,2:] = pat
            x[1:seq_len_row+1,0:2] = self.low_tol
            x[0,:] = self.low_tol
            x[0,1] = 1.0 # Start of sequence
            x[seq_len_row+1,:] = self.low_tol
            x[seq_len_row+1,0] = 1.0 # End of sequence
            y = np.ones(((self.max_seq*2)+2,self.out_bits), dtype=pat.dtype) * self.pad
            y[seq_len_row+2:(2*seq_len_row)+2,:] = pat # No side tracks needed for the output
            ti.append(x.tolist())
            to.append(y.tolist())
        return ti, to
        
    ''' CopyMemory Generator '''
    def generate(self):
        
        '''
        Generated by:
        Graves A., Wayne, G. and Danihelka, I. (2014) ‘Neural Turing Machines’, 
            Neural and Evolutionary Computing, arXiv:1410.5401 [cs.NE].
            <https://github.com/ajithcodesit/lstm_copy_task>
        '''
        
        input = []
        output = []
        for s in range(self.raw_size):
            t_in, t_out = self.get_pattern()
            input.append(t_in)
            output.append(t_out)
        
        input = np.asarray(input)
        output = np.asarray(output)
#        print('input:\n',input,'\nshape:',input.shape,' dtype:',input.dtype,'\n')
#        print('output:\n',output,'\nshape:',output.shape,' dtype:',output.dtype,'\n')
        
        self.raw_data = {'input':input,'output':output}
        
        return self

    ''' CopyMemory Generator '''
    def reshape(self):
        print('\nCopyMemoryGenerator|copy: NOT YET IMPLEMENTED!\n')
        return self
