
# Standard Imports
import os
import sys
from datetime import datetime
import random

# Library Imports
import numpy as np

import matplotlib.pyplot as plt

# Special Imports #
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.datasets import mnist

import proj_utils as utils
from proj_utils import _print

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
        
        self.train_dataset = None
        self.test_dataset = None
        
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
        
        self.epoch_size = 1
        if 'epoch_size' in kwargs:
            self.epoch_size = kwargs['epoch_size']
            del kwargs['epoch_size']
        if self.epoch_size == 0 or self.epoch_size is None: self.epoch_size = 1
        
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
        self.dtype = np.float32
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
    def groundtruth_generator( self , preds , maxlen=-1 ):
        
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
        
        return ( gndtru , preds )

    ''' DataGenerator '''
    def set_data(self, train_images, train_labels, test_images, test_labels):
        
        # Tests for input parameters, shapes, matching class parameters, ... etc.
        if  (train_images is None) or (train_labels is None) or (test_images is None) or (test_labels is None):
            return None
            
        # Ensure data-set outer dimension is divisible by the batch-size
        # Total-size of data-set    -> Check types
        if isinstance( train_images , np.ndarray ):
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
        train_shape = ( train_images.shape , train_labels.shape )
        test_shape = ( test_images.shape , test_labels.shape )
        self.shape = ( train_shape , test_shape )
        
        if self.makedataset is True:
            # Create Dataset objects from constructed data
            train_images = tf.data.Dataset.from_tensors( train_images , name = self.name + '_train_images' )
            train_labels = tf.data.Dataset.from_tensors( train_labels , name = self.name + '_train_labels' )
            test_images = tf.data.Dataset.from_tensors( test_images , name = self.name + '_test_images' )
            test_labels = tf.data.Dataset.from_tensors( test_labels , name = self.name + '_test_labels' )
            
            # Construct Batch from datasets if batch_size is set
            if self.batch_size > 0:
                train_images = train_images.batch( self.batch_size , drop_remainder = True )
                train_labels = train_labels.batch( self.batch_size , drop_remainder = True )
                test_images = test_images.batch( self.batch_size , drop_remainder = True )
                test_labels = test_labels.batch( self.batch_size , drop_remainder = True )
        
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        
#        print('train_images -> shape:',self.train_images.shape,' dtype:',self.train_images.dtype,'\n')
#        exit(0)
        
        return self
        
    def get_dataset( self ):
        if ( self.train_dataset is None ) or ( self.test_dataset is None ):
            return (None,None)
        return self.train_dataset , self.test_dataset

    ''' DataGenerator '''
    def get_data( self ):
        if (self.train_images is None) or (self.train_labels is None) or (self.test_images is None) or (self.test_labels is None):
            return (None,None)
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
    def mackeyglass( self , size = 4096 ):
        if size != self.raw_size:
            self.raw_size = size
        mkygls = [ 0.2 ]
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
        
        return self.raw_data


    ''' MackeyGlass Generator '''
    def generate( self ):
        
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
        
        ''' MackeyGlass Generator 
        Produces the MackeyGlass dataset for input parameters '''
        def l_mackeyglass( size , inival = 0.2 , del_x = 10 , tao = 30 ):
            mkygls = [ inival ]
            delta = 1 / del_x
            for t in range( int( size ) ):
                if t < tao:
                    yt = mkygls[t] + delta * ((0.2 * mkygls[t])/(1 + pow( mkygls[t] , 10 ) ) - 0.1 * mkygls[t])
                else:
                    yt = mkygls[t] + ((0.2 * mkygls[t-tao])/(1 + pow(mkygls[t-tao],10)) - 0.1 * mkygls[t])
                mkygls.append(yt)
            
            return np.asarray( mkygls ).astype( np.float32 )[1::]
        
        cwd = os.getcwd()
        print( cwd )
        
        ds_dir = os.path.join( cwd , 'datasets' )
        print( ds_dir )
        
        if not os.path.exists( ds_dir ): os.makedirs( ds_dir )
        
        sz = 5e4
        print( sz )
        
        ## Create Data ##
        dsfn = os.path.join( ds_dir , 'mackey_glass_' + str( int( sz ) ) + '.npy' )
        if os.path.exists( dsfn ):
            ds = np.load( dsfn , allow_pickle=True )
#            print( "loaded" )
        else:
            ds = l_mackeyglass( sz )
            np.save( dsfn , ds , allow_pickle = True )
#            print( "saved" )
        
        tstsz = int( ds.shape[0] * self.test_split )
        
        train_data , test_data = ds[0:-tstsz] , ds[-tstsz::]
        
        self.train_dataset = tf.keras.utils.timeseries_dataset_from_array(
            list( train_data ),
            targets = None,
            sequence_length=self.input_size,
            sequence_stride=1,
            sampling_rate=1,
            batch_size=self.batch_size,
            shuffle=False,
            seed=None,
            start_index=None,
            end_index=None
        )
        
        self.test_dataset = tf.keras.utils.timeseries_dataset_from_array(
            list( test_data ),
            targets = None,
            sequence_length=self.input_size,
            sequence_stride=1,
            sampling_rate=1,
            batch_size=self.batch_size,
            shuffle=False,
            seed=None,
            start_index=None,
            end_index=None
        )
        
        return self
        
        '''
        print( 'train_images:' , train_images.shape )
        print( 'train_labels:' , train_labels.shape )
        print( 'test_images:' , test_images.shape )
        print( 'test_labels:' , test_labels.shape )
        #'''
        
        bsz = self.batch_size
        
        trnbsz = train_images.shape[0] // self.batch_size
        tstbsz = test_images.shape[0] // self.batch_size
#        print( 'trnbsz' , trnbsz )
#        print( 'tstbsz' , tstbsz )
        
        ## Clip to batch sizes ##
        train_images = train_images[0:trnbsz*bsz]
        train_labels = train_labels[0:trnbsz*bsz]
        test_images = test_images[0:tstbsz*bsz]
        test_labels = test_labels[0:tstbsz*bsz]
        
        '''
        print( 'train_images:' , train_images.shape )
        print( 'train_labels:' , train_labels.shape )
        print( 'test_images:' , test_images.shape )
        print( 'test_labels:' , test_labels.shape )
        #'''
        
#        train_images = np.reshape( train_images , [ trnbsz , self.batch_size , train_images.shape[-1] ] )
#        train_labels = np.reshape( train_labels , [ trnbsz , self.batch_size , train_labels.shape[-1] ] )
#        test_images = np.reshape( test_images , [ tstbsz , self.batch_size , test_images.shape[-1] ] )
#        test_labels = np.reshape( test_labels , [ tstbsz , self.batch_size , test_labels.shape[-1] ] )
        
        '''
        print( 'train_images:' , train_images.shape )
        print( 'train_labels:' , train_labels.shape )
        print( 'test_images:' , test_images.shape )
        print( 'test_labels:' , test_labels.shape )
#        exit(0)
        #'''
        
        '''
        train_images = np.expand_dims( train_images , 1 )
        train_labels = np.expand_dims( train_labels , 1 )
        test_images = np.expand_dims( test_images , 1 )
        test_labels = np.expand_dims( test_labels , 1 )
        print( 'train_images:' , train_images.shape )
        print( 'train_labels:' , train_labels.shape )
        print( 'test_images:' , test_images.shape )
        print( 'test_labels:' , test_labels.shape )
        exit(0)
        #'''
        
        '''
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
        
        # Split into testing and training image data sets
        train_images = image_data[0:train_size]
        train_labels = label_data[0:train_size]
        test_images = image_data[train_size:train_size+test_split]
        test_labels = label_data[train_size:train_size+test_split]
#       '''
        
        # Finally reshape the data to fit the input parameters and save to class variables
        '''
        train_images = np.reshape(train_images,[ train_images.shape[0] , train_images.shape[1] , 1 ] )
        train_labels = np.reshape(train_labels,[ train_labels.shape[0] , train_labels.shape[1] , 1 ] )
        test_images = np.reshape(test_images,[ test_images.shape[0] , test_images.shape[1] , 1 ] )
        test_labels = np.reshape(test_labels,[ test_labels.shape[0] , test_labels.shape[1] , 1 ] )
        '''
        train_images = np.reshape( train_images , [ train_images.shape[0] , 1 , train_images.shape[1] ] )
        train_labels = np.reshape( train_labels , [ train_labels.shape[0] , 1 , train_labels.shape[1] ] )
        test_images = np.reshape( test_images , [ test_images.shape[0] , 1 , test_images.shape[1] ] )
        test_labels = np.reshape( test_labels , [ test_labels.shape[0] , 1 , test_labels.shape[1] ] )
#       '''
        
        '''
        print( 'train_images:' , train_images.shape )
        print( 'train_labels:' , train_labels.shape )
        print( 'test_images:' , test_images.shape )
        print( 'test_labels:' , test_labels.shape )
        exit(0)
        #'''
        
        skip = output_size // step_size
        self.gndtru = test_labels.copy()[0::skip].flatten()
        
        #'''
        ax = ( 0 , 2 , 1 )
        train_images = train_images.transpose( ax )
        train_labels = train_labels.transpose( ax )
        test_images = test_images.transpose( ax )
        test_labels = test_labels.transpose( ax )
        #'''

        '''
        print( 'train_images:' , train_images.shape )
        print( 'train_labels:' , train_labels.shape )
        print( 'test_images:' , test_images.shape )
        print( 'test_labels:' , test_labels.shape )
#        exit(0)
        #'''
        
        super().set_data( train_images.copy() , train_labels.copy() , test_images.copy() , test_labels.copy() )
        
#        super().set_dataset( train_images.copy() , train_labels.copy() , test_images.copy() , test_labels.copy() )

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

        minlen = min( trulen , prdlen )
        
        # Ensure we have the same length
        if trulen != prdlen:
            gndtru = gndtru[0:minlen]
            preds = preds[0:minlen]
        
        # Limits the maximum output. (Used for plotting.)
        if maxlen > 0 and minlen > maxlen:
            gndtru = gndtru[0:maxlen]
            preds = preds[0:maxlen]
        
        return ( gndtru , preds )

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
class psMNISTGenerator( DataGenerator ):
    
    ''' psMNIST Generator '''
    def __init__( self , **kwargs ):
        super().__init__( **kwargs )
        
    ''' psMNIST Generator '''
    def generate( self ):
        
        '''
        Setup Data Example:
        https://github.com/nengo/keras-lmu/blob/main/docs/examples/psMNIST.ipynb
        '''
        
        seed = 123 # random.randint( 0 , 1e6 )
        tf.random.set_seed( seed )
        np.random.seed( seed )
        rng = np.random.RandomState( seed )
#        print( seed )
#        exit(0)
        
        ( train_images , train_labels ) , ( test_images , test_labels ) = tf.keras.datasets.mnist.load_data()
        
        self.gndtru = test_labels.flatten()
        
        train_images = train_images / 255
        test_images = test_images / 255
        
        ''' # Show Image #
        plt.figure()
        plt.imshow(np.reshape(train_images[0], ( 28 , 28 ) ), cmap="gray")
        plt.axis("off")
        plt.title(f"Sample image of the digit '{train_labels[0]}'")
        plt.show()
        exit(0)
#       '''
        
        train_images = train_images.reshape( ( train_images.shape[0] , -1 , 1 ) )
        test_images = test_images.reshape( ( test_images.shape[0] , -1 , 1 ) )
        
        ''' # 5 x 5 grid of first 25 images -> pre-permutation #
        fig1 , axs1 = plt.subplots( 5 , 5 )
        i = 0
        for r in range( 5 ):
            for c in range( 5 ):
                axs1[r,c].imshow(train_images[i].reshape(28,28), cmap="gray")
                axs1[r,c].axis('off')
                i += 1
#        plt.show()
#        '''
        
        perm = rng.permutation( train_images.shape[1] )
        train_images = train_images[:, perm]
        test_images = test_images[:, perm]
        
        ''' # 5 x 5 grid of first 25 images -> post-permutation #
        fig2 , axs2 = plt.subplots( 5 , 5 )
        i = 0
        for r in range( 5 ):
            for c in range( 5 ):
                axs2[r,c].imshow(train_images[i].reshape(28,28), cmap="gray")
                axs2[r,c].axis('off')
                i += 1
        plt.show()
        exit(0)
#        '''
        
        '''
        ## If Using tf.keras.losses.CategoricalCrossentropy() ##
        # Turn the lables into 10-class bit-vectors w/ 1 in the argument position #
        train_labels = tf.keras.utils.to_categorical( train_labels , num_classes=10 )
        test_labels = tf.keras.utils.to_categorical( test_labels , num_classes=10 )
        '''
        ## If Using tf.keras.losses.SparseCategoricalCrossentropy( from_logits = True ) ##
        train_labels = train_labels.reshape( train_labels.shape[0] , 1 , 1 )
        test_labels = test_labels.reshape( test_labels.shape[0] , 1 , 1 )
#       '''
        '''
        print( 'train_labels:\n' , train_labels[0:10] , '  shape:' , train_labels.shape , '  dtype:' , train_labels.dtype , '  type:', type( train_labels ) , '\n' )
        print( 'test_labels:\n' , test_labels[0:10] , '  shape:' , test_labels.shape , '  dtype:' , test_labels.dtype , '  type:', type( test_labels ) , '\n' )
        exit(0)
        #'''
        
#        ''' ## TESTING ONLY ## -> used becuase full 60K take a bit of time.
        trnsz = 100
        tstsz = 25
        train_images = train_images[0:trnsz]
        train_labels = train_labels[0:trnsz]
        test_images = test_images[0:tstsz]
        test_labels = test_labels[0:tstsz]
#       '''
        
#        ax = ( 0 , 2 , 1 )
#        train_images = train_images.transpose( ax )
#        test_images = test_images.transpose( ax )
        
#        print( 'train_images[0]:\n' , train_images[0:5] , '  shape:' , train_images.shape , '  dtype:' , train_images.dtype , '\n' )
        
        super().set_data( train_images , train_labels , test_images , test_labels )
        
        return self


    ''' psMNIST Generator '''
    # Takes in a set of predictions and outputs,
    #   a 2-tuple (ground truth array, predition array), where the two 
    #   arrays have the same shape. (used for plotting in the trainer save() function)
    def groundtruth_generator( self , preds , maxlen = -1 ):
        
        preds = preds.copy()
        
        if not isinstance( preds , np.ndarray ):
            preds = np.asarray( preds )
        print( 'Entered: groundtruth_generator() for psMNIST' )
        print( 'gndtru:' , self.gndtru.shape )
        print( 'preds:' , preds.shape )
        exit(0)
        
        gndtru = self.gndtru
        if not isinstance( gndtru , np.ndarray ):
            gndtru = np.asarray( gndtru )
        
        gndtru = gndtru.flatten()
        preds = preds.flatten()
        
        return ( gndtru , preds )


'''
Dataset Source:
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. 
    Neural computation, 9(8):1735–1780, 1997.
    <https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext>
'''
class CopyMemoryGenerator(DataGenerator):
    
    ''' CopyMemory Generator '''
    def __init__( 
        self,
        in_bits = 10,
        out_bits = 8,
        low_tol = 1.e-3,
        high_tol = 1.0,
        min_seq = 1,
        max_seq = 20,
        pad = 1.e-3,
        dtype = np.float32,
        name = 'cpymem',
        **kwargs
     ):
     
        super().__init__( **kwargs )
        
        self.raw_size = None
        self.raw_data = None
        
        self.in_bits = in_bits
        self.out_bits = out_bits
        self.low_tol = low_tol
        self.high_tol = high_tol
        self.min_seq = min_seq
        self.max_seq = max_seq
        self.pad = pad
        self.dtype = dtype
        
    ''' CopyMemory Generator 
        Creates a batch set of input vectors. Output shape is (batch, idk , bits)'''
    def get_pattern(
                        self,
                        num_samples = 100,
                        max_sequence = 20,
                        min_sequence = 1,
                        in_bits = 10,
                        out_bits = 8,
                        pad = 0.001,
                        low_tol = 0.001,
                        high_tol = 1.0
                    ):
        ''' SOURCE:
        Graves A., Wayne, G. and Danihelka, I. (2014) ‘Neural Turing Machines’, 
            Neural and Evolutionary Computing, arXiv:1410.5401 [cs.NE].
            <https://github.com/ajithcodesit/lstm_copy_task>
        
        PAPER:
        A. Graves, G. Wayne, and I. Danihelka, “Neural Turing Machines.” arXiv, Dec. 10, 2014.
        Accessed: Nov. 09, 2023. [Online]. Available: http://arxiv.org/abs/1410.5401
        '''
        
        ti = []
        to = []

        for _ in range( num_samples ):
            
            seq_len_row = np.random.randint( low=min_sequence , high=max_sequence+1 )
#            print( 'seq_len_row' , seq_len_row )
            
            pat = np.random.randint( low=0 , high=2 , size=( seq_len_row , out_bits ) ).astype(np.float32)
#            print( 'pat.shape' , pat.shape )
            
            # Applying tolerance (So that values don't go to zero and cause NaN errors)
            pat[pat < 1] = low_tol
            pat[pat >= 1] = high_tol
            
            # Padding can be added if needed
            x = np.ones( ( (max_sequence*2)+2 , in_bits ) , dtype=pat.dtype ) * pad
            y = np.ones(((max_sequence*2)+2,out_bits), dtype=pat.dtype) * pad # Side tracks are not produced
#            print( 'y.shape' , np.asarray( y ).shape )
#            print( 'x.shape' , np.asarray( x ).shape )

            # Creates a delayed output (Target delay)
            x[1:seq_len_row+1,2:] = pat
            y[seq_len_row+2:(2*seq_len_row)+2,:] = pat # No side tracks needed for the output

            x[1:seq_len_row+1,0:2] = low_tol
            x[0,:] = low_tol
            x[0,1] = 1.0                        # Start of sequence
            x[seq_len_row+1,:] = low_tol
            x[seq_len_row+1,0] = 1.0            # End of sequence
            
            ti.append( y )
            to.append( x )
            
        return ti , to
        
        
    def urnn_get_data( self , time_steps , n_data , n_sequence ):
        
        seq = np.random.randint(1, high=9, size=(n_data, n_sequence))
        zeros1 = np.zeros((n_data, time_steps-1))
        zeros2 = np.zeros((n_data, time_steps))
        marker = 9 * np.ones((n_data, 1))
        zeros3 = np.zeros((n_data, n_sequence))

        x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
        y = np.concatenate((zeros3, zeros2, seq), axis=1).astype('int32')
        
        return x.T, y.T
        
        
    ''' CopyMemory Generator '''
    def generate( self ):
        
        self.raw_size = self.epoch_size * self.num_epochs if self.raw_size is None else self.raw_size
        
#        '''
        input = []
        output = []
        for s in range( 100 ):
            # Returns lists of lists
            t_in , t_out = self.get_pattern(
                num_samples = 100,
                max_sequence = self.max_seq,
                min_sequence = self.min_seq,
                in_bits = self.in_bits,
                out_bits = self.out_bits,
                pad = self.pad,
                low_tol = self.low_tol,
                high_tol = self.high_tol
            )
            input.append( t_in )
            output.append( t_out )
        
        input = np.asarray( input )
        output = np.asarray( output )
#        print('input -> shape:',input.shape,' dtype:',input.dtype,'\n')
#        print('output -> shape:',output.shape,' dtype:',output.dtype,'\n')
#        exit(0)
#       '''
        
        '''
        # --- Set data params ---------------- #
        time_steps = 100
        n_sequence = 10
        n_input = self.input_size
        n_output = self.output_size
        n_train = int( 1e5 )
        n_test = int( 1e4 )
        num_batches = self.epoch_size
        
        # --- Create data --------------------
        train_x, train_y = self.urnn_get_data( time_steps , n_train, n_sequence )
        test_x, test_y = self.urnn_get_data( time_steps , n_test, n_sequence )
        
        train_images = train_x.transpose()
        train_labels = train_y.transpose()
        test_images = test_x.transpose()
        test_labels = test_y.transpose()
#        '''
        
#        def reshp_data( nparr , inpsz ):
#            return tf.reshape( nparr , [ nparr.shape[0] // inpsz , nparr.shape[1] , inpsz ] )
        
#        train_images = reshp_data( train_images , n_input )
#        train_labels = reshp_data( train_labels , n_input )
#        test_images = reshp_data( test_images , n_input )
#        test_labels = reshp_data( test_labels , n_input )
        
        split = int( input.shape[0] * self.test_split )
#        print( 'split' , split )
        
        train_labels , train_images = input[0:-split] , output[0:-split]
#        train_images , train_labels = input[0:-split] , output[0:-split]
#        print('A) train_images -> shape:',train_images.shape,' dtype:',train_images.dtype,'\n')
#        print('A) train_labels -> shape:',train_labels.shape,' dtype:',train_labels.dtype,'\n')
        
        test_labels , test_images = input[-split::] , output[-split::]
#        test_images , test_labels = input[-split::] , output[-split::]
#        print('A) test_images -> shape:',test_images.shape,' dtype:',test_images.dtype,'\n')
#        print('A) test_labels -> shape:',test_labels.shape,' dtype:',test_labels.dtype,'\n')
        
        '''
        def double_concat( tensor ):
            return np.concatenate( np.concatenate( tensor ) )
        train_images = double_concat( train_images ).astype( self.dtype )
        train_labels = double_concat( train_labels ).astype( self.dtype )
        test_images = double_concat( test_images ).astype( self.dtype )
        test_labels = double_concat( test_labels ).astype( self.dtype )
        
        train_images = np.expand_dims( train_images , 1 )
        train_labels = np.expand_dims( train_labels , 1 )
        test_images = np.expand_dims( test_images , 1 )
        test_labels = np.expand_dims( test_labels , 1 )
        '''
        
#        '''
        train_images = np.concatenate( train_images )
        train_labels = np.concatenate( train_labels )
        test_images = np.concatenate( test_images )
        test_labels = np.concatenate( test_labels )
#       '''
        
        '''
        print('train_images -> shape:',train_images.shape,' dtype:',train_images.dtype,'\n')
        print('train_labels -> shape:',train_labels.shape,' dtype:',train_labels.dtype,'\n')
        print('test_images -> shape:',test_images.shape,' dtype:',test_images.dtype,'\n')
        print('test_labels -> shape:',test_labels.shape,' dtype:',test_labels.dtype,'\n')
        exit(0)
#       '''

        '''
        seq = self.input_size
        inpt = train_images[1]
        oupt = train_labels[1]
        
        fig , axs = plt.subplots( 2 , 1 )
        idx = 5
        fig.suptitle('Copy Memory Task (Max Sequence Length '+str(seq)+')')
        fig.supxlabel('Time ---->')
        img0 = axs[0].imshow( inpt )
        axs[0].set_ylabel('Image')
        axs[0].set_aspect('auto')
        
        axs[1].imshow( oupt )
        axs[1].set_ylabel('Label')
        axs[1].set_aspect('auto')
        
        fig.colorbar( img0 , ax = axs , orientation='vertical' , fraction=0.1 )
        
#        plt.tight_layout()
        plt.show()
        
        exit(0)
#       '''
        
        self.gndtru = test_labels.copy()
        
        self.set_data( train_images , train_labels , test_images , test_labels )
        
        return self


    ''' CopyMemory Generator '''
    # Takes in a set of predictions and outputs,
    #   a 2-tuple (ground truth array, predition array), where the two 
    #   arrays have the same shape. (used for plotting in the trainer save() function)
    def groundtruth_generator( self , preds , maxlen = -1 ):
        
        preds = preds.copy()
        
        if not isinstance(preds,np.ndarray):
            preds = np.asarray(preds)
        
        gndtru = self.gndtru
        if not isinstance(gndtru,np.ndarray):
            gndtru = np.asarray(gndtru)
        
        gndtru = gndtru.flatten()
        preds = preds.flatten()
        
        return ( gndtru , preds )

    ''' CopyMemory Generator '''
    def reshape(self):
        print('\nCopyMemoryGenerator|copy: NOT YET IMPLEMENTED!\n')
        return self




'''
Le, Quoc V., Navdeep Jaitly, and Geoffrey E. Hinton. 
    “A Simple Way to Initialize Recurrent Networks of Rectified Linear Units.” 
    arXiv, April 7, 2015. <https://doi.org/10.48550/arXiv.1504.00941>.
'''
class seqMNISTGenerator( DataGenerator ):
    
    ''' seqMNIST Generator '''
    def __init__( self,
                  **kwargs
                 ):
        super().__init__(**kwargs)
        
    ''' seqMNIST Generator '''
    def generate(self):
        ''' Setup Data Example:
        https://github.com/nengo/keras-lmu/blob/main/docs/examples/psMNIST.ipynb
        '''
        (train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0],1,train_images.shape[1]*train_images.shape[2])
        train_labels = train_labels.reshape(train_labels.shape[0],1,1)
        test_images = test_images.reshape(test_images.shape[0],1,test_images.shape[1]*test_images.shape[2])
        test_labels = test_labels.reshape(test_labels.shape[0],1,1)
        
        self.raw_data = {
                          'train':{'images':train_images,'labels':train_labels},
                          'test':{'images':test_images,'labels':test_labels}
                         }
        
        return self
        
        
    ''' seqMNIST Generator '''
    def reshape(self):
        print('\nseqMNISTGenerator|reshape: NOT YET IMPLEMENTED!\n')
        return self




# TODO: Adding problem see https://github.com/ratschlab/uRNN/blob/master/adding_problem.py
