
# Standard Imports
import os
import sys
from datetime import datetime
from inspect import isfunction

import json

# Library Imports
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import tensorflow as tf
import keras

# Local imports
import proj_utils as utils
from proj_utils import _print

from test_model import TestRNNLayer

def build_weight_dict(model,name_list=[]):
    out_dict = {}
    for wgt in model.get_layer(index=1).weights:
        w = wgt.read_value().numpy()
        for nme in name_list:
            if nme in wgt.name:
                out_dict[nme] = w.copy()
    return out_dict

class ModelTrainer():
    
    ''' ModelTrainer '''
    def __init__(
                    self,
                    data_callback,
                    **kwargs
                 ):
        
        self.name = 'unknown_trainer'
        if 'name' in kwargs:
            self.name = kwargs['name']
        
        # Build directories
        self.trainer_dir = os.getcwd() + '\\training_runs\\' + self.name
#        self.chkpnt_dir = os.path.dirname(model_dir + '/results/checkpoints/training_1/cp.ckpt')
        
        # Setup training specific parameters
        self.batch_size = 1
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        assert self.batch_size > 0, self.name + ': batch_size cannot be less than 1.    batch_size = ' + str(self.batch_size)
        
        self.epoch_size = 1
        if 'epoch_size' in kwargs:
            self.epoch_size = kwargs['epoch_size']
        assert self.epoch_size > 0, self.name + ': epoch_size cannot be less than 1.    epoch_size = ' + str(self.epoch_size)
        
        self.num_epochs = 1
        if 'num_epochs' in kwargs:
            self.num_epochs = kwargs['num_epochs']
        assert self.num_epochs > 0, self.name + ': num_epochs cannot be less than 1.    num_epochs = ' + str(self.num_epochs)
        
        # The training and testing data (can be either a list-of-list, np.ndarray, or tf.dataset)
        self.dataobj = data_callback(**kwargs)
        data = self.dataobj.generate()
        

        
        # Function callback for getting ground truth data matching prediction data
        self.groundtruth_callback = self.dataobj.groundtruth_generator
        
        # Each of training and testing are tuples of images and labels
        self.train_data, self.test_data = data.get_data()
        self.train_shape, self.test_shape = data.get_shape()
        
        # Current model generated in train() from model_callbacks
        self.models = {}
        self.model_idx = 0


    ''' ModelTrainer '''
    def execute(
                 self,
                 model_callbacks,
                 model_load=False,
                 model_directory='',
                 fit_valid_split = 0.0,
                 verbose=1,
                 **kwargs
                ):
        
        # Check input callbacks list
        if isfunction(model_callbacks): model_callbacks = [model_callbacks]
        if not isinstance(model_callbacks,list): model_callbacks = list(model_callbacks)
        
        # Train and Test Data
        trn_imgs, trn_lbls = self.train_data
        tst_imgs, tst_lbls = self.test_data
        
        if verbose > 0:
            print('\nData Size:')
            print('Train Images:',self.train_data[0].shape)
            print('Train Labels:',self.train_data[1].shape)
            print(' Test Images:',self.test_data[0].shape)
            print(' Test Labels:',self.test_data[1].shape)
            print()
        
        tf.keras.backend.clear_session()
        
        # Check if we are suppose to load the model & for a valid directory
        if model_load is True:
            
            # Check that all required parameters have been provided.
            assert_str = '\n'
            assert_str += 'setting input parameter \'model_load\' to True, requires a valid \'model_directory\' parameter.\n'
            assert isinstance(model_directory,str) and len(model_directory) > 0, assert_str
            
            # Check that all required parameters are correct
            # File directory exists
            assert_str = '\n'
            assert_str += 'unable to verify model directory exists, ensure full path and correct file extension.\n'
            assert_str += '\tparameter: model_directory = ' + model_directory + '\n'
            assert os.path.exists(model_directory), assert_str
            
            # File directory has correct format
            assert_str = '\n'
            assert_str = 'Model must be \'keras\', \'tf\', or \'hdf5\' formats'
            dir_frmt = model_directory.split('\\')[-1]
            assert ('.keras' in dir_frmt) or ('.tf' in dir_frmt) or ('.hdf5' in dir_frmt), assert_str
            
            custnme = keras.utils.generic_utils.get_registered_name(TestRNNLayer)
            custobj = keras.utils.generic_utils.get_registered_object('test_model>TestRNNLayer')
            
            cust_obj = {'test_model>TestRNNLayer':TestRNNLayer}
            # Attempt to load the model
            model = tf.keras.models.load_model(
                        model_directory,
                        custom_objects = {'TestRNNLayer':TestRNNLayer}
                    )
            model.summary()
            
            # Add Prediction and Evalutaion steps
            pred = model.predict(
                tst_imgs,
                batch_size=self.batch_size,
                verbose=verbose
            )
            
            # Plot and save results based on ground truth
            gt, est = self.groundtruth_callback(pred, 2048)
            x_vals = np.arange(gt.shape[0])
            legend=['Ground Truth','Predicted']
            utils.plot(
                        x_vals,
                        [gt,est],
                        colors=['b','r'],
                        linestyles=['-','--'],
                        title=self.name.capitalize() + ' Results - ' + model.name,
                        xlabel='Time',
                        legend=legend,
                        show=True,
                        save=False,
                        dir='',
                        name=model.name
                       )
            
            return False
            
        else:
            
            rt_dict = utils.runtime_dict()
            
            # Itterate the model_callbacks list.
            #   Test for compatibility with the dataset.
            for model_callback in model_callbacks:
                
                # Clear the backend stuff.
                tf.keras.backend.clear_session()
                
                ''' CREATE MODEL '''
                model_struct = model_callback(trn_imgs[0].shape, trn_lbls[0].shape, self.batch_size, self.trainer_dir)
                
                # Set initial model runtime
                model_struct['runtime'] = rt_dict
                
                ''' TEST DATASET AND MODEL COMPATIBILITY '''
                # Test for output shape compatibility
                #utils.ensure_output_shape(model_struct['model'], self.train_data)
                
                # Load dictionary item to trainer models dictionary.
                modstr = model_struct['name']+'_'+str(self.model_idx)
                self.models[modstr] = model_struct
                self.model_idx += 1
            
            for name, model_struct in self.models.items():
                
                model = model_struct['model']
                
                iniwgt_list, finwgt_list = None, None
                
                # Add a copy of the specified weights to the structure BEFORE training,
                #   if a list of names was provided in the model_callback functions 
                #   (see the models.py file for more details on where this is added)
                if 'wgtlst' in model_struct:
                    model_struct['iniwgt'] = utils.build_weight_dict(model,model_struct['wgtlst'])
                
    #            print('INITIAL:\n')
    #            print('trainable Variables:\n',model.trainable_variables[0],'\n\n\n\n')
    #            print('trainable Weights:\n',model.trainable_weights[0],'\n\n\n\n')
    #            print('trainable Weights:\n',model.weights[0],'\n\n\n\n')
                
                ''' TRAIN '''
                model_struct['hist'] = model.fit( 
                                                  x=trn_imgs, 
                                                  y=trn_lbls,
                                                  batch_size=self.batch_size,
                                                  epochs=self.num_epochs,
                                                  validation_split=fit_valid_split,
                                                  verbose=verbose,
                                                  shuffle=False,
                                                  steps_per_epoch=self.epoch_size,
                                                  callbacks=model_struct['callbacks']
                                                )
                # Clear the model and Load weights.
                tf.keras.backend.clear_session()
                model.load_weights(model_struct['chkpnt_dirs']['loss'])
                
    #            print('FINAL:\n')
    #            print('trainable Variables:\n',model.trainable_variables[0],'\n\n\n\n')
    #            print('trainable Weights:\n',model.trainable_weights[0],'\n\n\n\n')
    #            print('trainable Weights:\n',model.weights[0],'\n\n\n\n')
                
                # Add a copy of the specified weights to the structure AFTER training.
                #   (This is the same as above, but after the weights have been updated)
                if 'wgtlst' in model_struct:
                    model_struct['finwgt'] = utils.build_weight_dict(model,model_struct['wgtlst'])
                
                # Note: Initial and final weights are saved in the trainer.save() function #
                
                # TODO: hopf-rnn not working with current predict function
                #       (need to see what is wrong with my model)
                
                # Make sure requested batch-shape is compatible to reshape before prediction
                assert (tst_imgs.shape[0] % self.batch_size) == 0, 'total image dataset size must be divisible by the set batch-size.'
                
                ''' Predict on output '''
                if verbose > 0: print('\nPredict:')
                model_struct['pred'] = model.predict(
                                                      tst_imgs,
                                                      batch_size=self.batch_size,
                                                      verbose=verbose
                                                     )
                if verbose > 0: print('\n')
                if verbose > 1: print('pred:',model_struct['pred'])
                
                if verbose > 0: print('\nEvaluate:')
                model_struct['eval'] = model.evaluate(
                                                        tst_imgs,
                                                        tst_lbls,
                                                        verbose=verbose,
                                                        batch_size=self.batch_size
                                                      )
                if verbose > 0: print('\n')
                if verbose > 1: print('eval:',model_struct['eval'])
                
        return True


    ''' ModelTrainer 
        Saves each model ran through the trainer.
        In the specified model directory.'''
    def save(
              self,
              max_plot_len=1024,
              show_plot=False,
              save_plot=True,
              save_weights=False,
              verbose=0
             ):
        
        if len(self.models) == 0:
            return self
        
        for k, v in self.models.items():
            
            # For saving a text summary file.
            readme = {}
            
            readme['dataset'] = self.name.capitalize()
            data_attr = self.dataobj.get_attributes()
            for dk, dv in data_attr.items():
                readme[dk] = dv
            
            # Get generation
            gen = int(v['gdir'].split('_')[-1])
            v['gen'] = gen
            
            ''' - - - - - SAVE MODEL - - - - - '''
            
            # Save the model in the gen_## file directory
            filedir = os.path.join(v['gdir'],'model')
            if not os.path.exists(filedir):
                os.mkdir(filedir)
            
            filename = os.path.join(filedir,v['name']+'_model'+'.keras')
            v['model'].save(filename)
            
            # Get Model Summary as string.
            sumstr = []
            v['model'].summary(print_fn=lambda x: sumstr.append(x), expand_nested=True, show_trainable=True)
            readme['summary'] = '\n'.join(sumstr)
            
            ''' - - - - - SAVE MODEL WEIGHTS AND COMPARISON - - - - - '''
            
            if save_weights:
                
                iniwgt, finwgt = None, None
                
                # Initial weight values, loaded before training (see execute() function)
                if 'iniwgt' in v:
                    iniwgt = v['iniwgt']
                    for wk, wv in iniwgt.items():
                        fn = 'iniwgt_'+wk
                        filename = os.path.join(filedir,fn+'.npy')
                        np.save(filename,wv,allow_pickle=True)
                        v[fn+'_dir'] = filename
                
                # Final weight values, loaded after training (see execute() function)
                if 'finwgt' in v:
                    finwgt = v['finwgt']
                    for wk, wv in finwgt.items():
                        fn = 'finwgt_'+wk
                        filename = os.path.join(filedir,fn+'.npy')
                        np.save(filename,wv,allow_pickle=True)
                        v[fn+'_dir'] = filename
                
                # Compare initial and final weights
                if (iniwgt is not None) and (finwgt is not None) and (len(iniwgt) == len(finwgt)):
                        # itterates the initial and final saved weights.
                        for wk in iniwgt.keys():
                            
                            if wk not in finwgt: continue
                            
                            # Elementwise Standard difference
                            fn = 'wgt_diff_' + wk
                            wgtcng = iniwgt[wk] - finwgt[wk]
                            filename = os.path.join(filedir,fn+'.npy')
                            np.save(filename,wgtcng,allow_pickle=True)
                            v[fn+'_dir'] = filename
                            
                            # Elementwise Absolute Difference
                            fn = 'wgt_absdiff_' + wk
                            wgtcng = np.abs(iniwgt[wk] - finwgt[wk])
                            filename = os.path.join(filedir,fn+'.npy')
                            np.save(filename,wgtcng,allow_pickle=True)
                            v[fn+'_dir'] = filename
                            
                            # TODO: some matrix comparison style difference maybe
                            # TODO: Generate plots for these differences
                
            ''' - - - - - SAVE TRAINING RESULTS - - - - - '''
            # Save the model results in the gen_## file directory
            filedir = os.path.join(v['gdir'],'results')
            if not os.path.exists(filedir): os.mkdir(filedir)
            
            readme_metrics = []
            if 'hist' in v:
                hist = v['hist']
                v['fit'] = hist.params
                metrics = []
                legend = []
                for hk, hv in hist.history.items():
                    v[hk] = hv
                    metrics.append(np.asarray(hv))
                    legend.append(hk)
                    if 'val_' not in hk: readme_metrics.append(hk)      # Add string for later text file 'readme'
                x_vals = np.arange(len(metrics[0]))
                y_vals = metrics
                # Plot Metrics
                utils.plot( 
                            x_vals,
                            y_vals,
                            colors=['b','g','r','m'],
                            title='Metric vs. Epoch - '+v['name'] + ' - Gen: ' + str(gen),
                            xlabel='Epochs',
                            ylabel='Metrics',
                            legend=legend,
                            show=show_plot,
                            save=save_plot,
                            dir=filedir,
                            name='metrics'
                           )
            
            ''' - - - - - SAVE EVALUATION RESULTS - - - - - '''
            # Prediction values and plot
            if 'pred' in v:
                
                # Save predictions as np.ndarray in file
                filename = os.path.join(filedir,'pred.npy')
                np.save(filename,np.asarray(v['pred']),allow_pickle=True)
                v['pred_dir'] = filename
                
                # Plot and save results based on ground truth
                gt, est = self.groundtruth_callback(v['pred'],max_plot_len)
                x_vals = np.arange(gt.shape[0])
                legend=['Ground Truth','Predicted']
                utils.plot( 
                            x_vals,
                            [gt,est],
                            colors=['b','r'],
                            linestyles=['-','--'],
                            title=self.name.capitalize() + ' Results - '+v['name'] + ' - Gen: ' + str(gen),
                            xlabel='Time',
                            legend=legend,
                            show=show_plot,
                            save=save_plot,
                            dir=filedir,
                            name=v['name']
                           )
            
            # Save Evaluation
            if 'eval' in v:
                if not isinstance(v['eval'],list): v['eval'] = [v['eval']]
                # Add to summary text file
                rmmlen = len(readme_metrics)
                if len(v['eval']) == rmmlen:
                    readme['']=''
                    readme['Metrics']=''
                    for i in range(rmmlen):
                        readme[readme_metrics[i]] = v['eval'][i]
                np.save(
                        os.path.join(filedir,'eval.npy'),
                        np.asarray(v['eval']),
                        allow_pickle=True
                        )
                v['eval_dir'] = filename
                
            # Write a summary that contains all relevant information to the generation folder
            rmmw = utils.readme_writer(filename='Summary',filedir=v['gdir'],**readme)
            if verbose > 1: print(rmmw)
            
            # Delete all non-json-able dictionary elements.
            if 'model' in v: del v['model']
            if 'callbacks' in v: del v['callbacks']
            if 'hist' in v: del v['hist']
            if 'pred' in v: del v['pred']
            if 'eval' in v: del v['eval']
            if 'finwgt' in v: del v['finwgt']
            if 'iniwgt' in v: del v['iniwgt']
            
            ''' Saving Training Meta '''
            # Save the remailing structure
            jsonstr = json.dumps(v)
            filename = os.path.join(v['gdir'],'meta.json')
            with open(filename,'w') as fileobj:
                fileobj.write(jsonstr)
            
        return self
