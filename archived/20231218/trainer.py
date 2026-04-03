
# Standard Imports
import os
import sys
import time
import math
import csv
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

## Local Imports ##
import proj_utils as utils
from proj_utils import _print , save_meta, load_meta


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
        if 'name' in kwargs: self.name = kwargs['name']
        
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
                 model_directory='',
                 fit_valid_split = 0.0,
                 do_prediction=False,
                 do_evaluation=False,
                 verbose=1,
                 testing=None,
                 load_chkpnt = None,
                 **kwargs
                ):
        
        # Check input callbacks list
        if isfunction(model_callbacks): model_callbacks = [model_callbacks]
        if not isinstance(model_callbacks,list): model_callbacks = list(model_callbacks)
        
        # Train and Test Data
        trn_imgs, trn_lbls = self.train_data
        tst_imgs, tst_lbls = self.test_data
        
        '''
        plt.figure()
        plt.imshow( trn_imgs[0].reshape(8, -1), cmap = 'gray' )
        plt.axis( "off" )
        plt.title(f"Permuted sequence of the digit '{trn_lbls[0]}' (reshaped to 98 x 8)")
        plt.show()
        exit(0)
        '''
        
        if verbose > 0:
            print('\nData Size:')
            print('Train Images:',self.train_data[0].shape)
            print('Train Labels:',self.train_data[1].shape)
            print(' Test Images:',self.test_data[0].shape)
            print(' Test Labels:',self.test_data[1].shape)
            print()
        
        tf.keras.backend.clear_session()
        
        # Run quick tests without training.
        if (testing is not None) and isinstance(testing, int) and (testing > 0):
            
            _pstr = '\n'*2 + '-'*20 + ' TESTING (' + str(testing) + ') ' + '-'*20 + '\n'*2
            print( _pstr*3 )
            
            assert len(model_callbacks) > 0, 'trying to test with trainer.py but the model callbacks list was empty'
            
            for model_callback in model_callbacks:
                
                # Clear the backend stuff.
                tf.keras.backend.clear_session()
                
                ''' CREATE MODEL '''
                model_struct = model_callback( trn_imgs[0].shape , trn_lbls[0].shape , self.batch_size , self.trainer_dir )
                
                assert 'model' in model_struct, 'trying to test with trainer.py but the model was not in model_struct after creation'
                
                # Generate sudo batch stuff
                
                bs = trn_imgs.shape[0] // self.batch_size
                sudo_batch = trn_imgs[0:bs*self.batch_size]
                rshparr = [bs,self.batch_size,sudo_batch.shape[-2],sudo_batch.shape[-1]]
                sudo_batch = np.reshape(sudo_batch, rshparr)
                
                for i in range(testing):
                    model_struct['model'](sudo_batch[i])
                    
                return True
            
        else:
            
            rt_dict = utils.runtime_dict()
            
            # Itterate the model_callbacks list.
            #   Test for compatibility with the dataset.
            for model_callback in model_callbacks:
                
                # Clear the backend stuff.
                tf.keras.backend.clear_session()
                
                ''' CREATE MODEL '''
                model_struct = model_callback(
                    trn_imgs[0].shape,
                    trn_lbls[0].shape,
                    self.batch_size,
                    self.trainer_dir
                )
                
                # Set initial model runtime
                model_struct['runtime'] = rt_dict
                
                ''' TEST DATASET AND MODEL COMPATIBILITY '''
                # Test for output shape compatibility
                #utils.ensure_output_shape(model_struct['model'], self.train_data)
                
                # Load dictionary item to trainer models dictionary.
                modstr = model_struct['name']+'_'+str(self.model_idx)
                self.models[modstr] = model_struct
                self.model_idx += 1
            
#            for name , model_struct in self.models.items():
                
                model = model_struct['model']
                
                iniwgt_list, finwgt_list = None, None
                
                # Add a copy of the specified weights to the structure BEFORE training,
                #   if a list of names was provided in the model_callback functions 
                #   (see the models.py file for more details on where this is added)
                if 'wgtlst' in model_struct:
                    model_struct['iniwgt'] = utils.build_weight_dict(model,model_struct['wgtlst'])
                
                '''
                print('INITIAL:\n')
                print('A:\n',model.trainable_variables[0],'\n\n')
                print('B:\n',model.trainable_weights[0],'\n\n')
                print('C:\n',model.weights[0],'\n\n')
                print('D:\n',model.get_weights()[0],'\n\n\n\n')
#               '''
                
                if isinstance( load_chkpnt , str ):
                    last_point = tf.train.latest_checkpoint( load_chkpnt )
                    model.load_weights( last_point )
                
#                '''
                model_struct['train_time_start'] = time.time()
                '''
                try:
                '''
                model_struct['hist'] = model.fit( 
                    x = trn_imgs, 
                    y = trn_lbls,
                    batch_size = self.batch_size,
                    epochs = self.num_epochs,
                    validation_split = fit_valid_split,
                    verbose = verbose,
                    shuffle = model_struct['shuffle'],
                    steps_per_epoch = self.epoch_size,
                    callbacks = model_struct['callbacks']
                )
                '''
                except:
                    print( 'Trainer -> failed in model.fit for ' + model_struct['name'] + ' ... ' )
                    tf.keras.backend.clear_session()
                    model.load_weights( model_struct['chkpnt_dirs']['loss'] )
                    if 'hist' in model_struct: del model_struct['hist']
                '''
                
                if verbose > 0 and 'hist' in model_struct:
                    print('Train History:')
                    for k , v in model_struct['hist'].history.items():
                        vals = np.asarray( v )
                        mx = np.round( np.max( vals ) , 4 )
                        mn = np.round( np.min( vals ) , 4 )
                        av = np.round( np.mean( vals ) , 4 )
                        print( '\t'+k+':\n\t','max:', mx,' min:',mn,' mean:',av,'\n' )
                    
                model_struct['train_time_end'] = time.time()
#               '''
                
                '''
                print('FINAL:\n')
                print('A:\n',model.trainable_variables[0],'\n\n')
                print('B:\n',model.trainable_weights[0],'\n\n')
                print('C:\n',model.weights[0],'\n\n')
                print('D:\n',model.get_weights()[0],'\n\n\n\n')
                exit(0)
#               '''
                
                # Add a copy of the specified weights to the structure AFTER training.
                #   (This is the same as above, but after the weights have been updated)
                if 'wgtlst' in model_struct:
                    model_struct['finwgt'] = utils.build_weight_dict(model,model_struct['wgtlst'])
                
                # Note: Initial and final weights are saved in the trainer.save() function #
                
                # TODO: hopf-rnn not working with current predict function
                #       (need to see what is wrong with my model)
                
                # Make sure requested batch-shape is compatible to reshape before prediction
                assert (tst_imgs.shape[0] % self.batch_size) == 0, 'total image dataset size must be divisible by the set batch-size.'
                
                if do_prediction:
                    
                    model_struct['pred_time_start'] = time.time()
                    
                    ''' Predict on output '''
                    if verbose > 0: print('\nPredict:')
                    model_struct['pred'] = model.predict(
                        tst_imgs,
                        batch_size = self.batch_size,
                        verbose = verbose
                    )
                    if verbose > 0: print('\n')
                    if verbose > 1: print('pred:',model_struct['pred'])
                    
                    model_struct['pred_time_end'] = time.time()
                    
                    
                if do_evaluation:
                    
                    start_time = time.time()
                    
                    model_struct['eval_time_start'] = time.time()
                    
                    if verbose > 0: print('\nEvaluate:')
                    model_struct['eval'] = model.evaluate(
                                                            tst_imgs,
                                                            tst_lbls,
                                                            verbose=verbose,
                                                            batch_size=self.batch_size
                                                          )
                    if verbose > 0: print('\n')
                    if verbose > 1: print('eval:',model_struct['eval'])
                    
                    model_struct['eval_time_end'] = time.time()
                    
                    
        return True


    ''' ModelTrainer 
        Saves each model ran through the trainer.
        In the specified model directory.'''
    def save(
              self,
              max_plot_len = 1024,
              show_plot = False,
              save_plot = False,
              show_weights = False,
              save_weights = False,
              verbose = 0
             ):
        
        if len(self.models) == 0:
            return self
        
        csvhdr = []
        csvstr = []
        
        def setcsv( msg , val ):
            csvhdr.append( msg )
            csvstr.append( str( val ) )
            
        for k, v in self.models.items():
            
            tf.keras.backend.clear_session()
            
            print('Saving Model:')
            print('  name:', v['name'])
            
            setcsv( 'name' , v['name'] )
            
            # For saving a text summary file.
            readme = {}
            
            readme['dataset'] = self.name.capitalize()
            data_attr = self.dataobj.get_attributes()
            for dk, dv in data_attr.items():
                readme[dk] = dv
            
            # Get generation
            gen = int(v['gdir'].split('_')[-1])
            v['gen'] = gen
            print('  Gen:', gen)
            
            setcsv( 'gen' , v['gen'] )
            
            ''' - - - - - SAVE MODEL - - - - - '''
            
            # Save the model in the gen_## file directory
            mfiledir = os.path.join(v['gdir'],'model')
            if not os.path.exists(mfiledir):
                os.makedirs( mfiledir )
            
            ''' - - - - - SAVE TRAINING RESULTS - - - - - '''
            # Save the model results in the gen_## file directory
            rfiledir = os.path.join(v['gdir'],'results')
            if not os.path.exists(rfiledir):
                os.makedirs( rfiledir )
            
            filename = os.path.join( mfiledir , v['name']+'_model'+'.keras' )
            try:
                v['model'].save( filename )
            except:
                print('failed to save model to file location.\nFilename:',filename,'\nLocation:',mfiledir)
                print('possibly not able to load all the .keras requirements ... continuing to next section.')
                print()
            
            # Get Model Summary as string.
            sumstr = []
            v['model'].summary(print_fn=lambda x: sumstr.append(x), expand_nested=True, show_trainable=True)
            readme['summary'] = '\n'.join(sumstr)
            
            ''' - - - - - SAVE MODEL WEIGHTS AND COMPARISON - - - - - '''
            
            got_dict_weights = True
            if ('iniwgt' not in v) and ('finwgt' not in v):
                got_dict_weights = False
            
            if (show_weights or save_weights) and got_dict_weights:
                
                iniwgt, finwgt = None, None
                
                # Initial weight values, loaded before training (see execute() function)
                if 'iniwgt' in v:
                    iniwgt = v['iniwgt']
                    for wk, wv in iniwgt.items():
                        fn = 'iniwgt_'+wk
                        filename = os.path.join(mfiledir,fn+'.npy')
                        np.save(filename, wv, allow_pickle=True)
                        v[fn+'_dir'] = filename
                        
                        # Create weight matrix.
                        utils.plot(
                            wv,
                            title='Trainable Initial Weight ' + wk.capitalize() + '  gen: ' + str(gen),
                            xlabel='X',
                            ylabel='Y',
                            show=show_weights,
                            save=save_weights,
                            dir=rfiledir,
                            name=fn
                        )
                        
                # Final weight values, loaded after training (see execute() function)
                if 'finwgt' in v:
                    finwgt = v['finwgt']
                    for wk, wv in finwgt.items():
                        fn = 'finwgt_'+wk
                        filename = os.path.join(mfiledir, fn+'.npy')
                        np.save(filename, wv, allow_pickle=True)
                        v[fn+'_dir'] = filename
                    
                        # Create weight matrix.
                        utils.plot(
                            wv,
                            title='Trainable Final Weight ' + wk.capitalize() + ' - Gen: ' + str(gen),
                            xlabel='X',
                            ylabel='Y',
                            show=show_weights,
                            save=save_weights,
                            dir=rfiledir,
                            name=fn
                        )
                    
                # Compare initial and final weights
                if ( iniwgt is not None ) and ( finwgt is not None ) and ( len( iniwgt ) == len( finwgt ) ):
                    
                        # itterates the initial and final saved weights.
                        for wk in iniwgt.keys():
                            
                            if wk not in finwgt: continue
                            
                            # Elementwise Standard difference
                            fn = 'wgt_diff_' + wk
                            wv = iniwgt[wk] - finwgt[wk]
                            filename = os.path.join(mfiledir, fn+'.npy')
                            np.save(filename, wv, allow_pickle=True)
                            v[fn+'_dir'] = filename
                                
                            # Create weight matrix.
                            utils.plot(
                                wv,
                                title='Trainable Weight Difference - ' + wk.capitalize() + ' - Gen: ' + str(gen),
                                xlabel='X',
                                ylabel='Y',
                                show=show_weights,
                                save=save_weights,
                                dir=rfiledir,
                                name=fn
                            )
                            
                            # Elementwise Absolute Difference
                            fn = 'wgt_absdiff_' + wk
                            wv = np.abs(iniwgt[wk] - finwgt[wk])
                            filename = os.path.join(mfiledir, fn+'.npy')
                            np.save(filename, wv, allow_pickle=True)
                            v[fn+'_dir'] = filename
                            
                            # Create weight matrix.
                            utils.plot(
                                wv,
                                title='Trainable Weight Abs-Difference - ' + wk.capitalize() + ' - Gen: ' + str(gen),
                                xlabel='X',
                                ylabel='Y',
                                show=show_weights,
                                save=save_weights,
                                dir=rfiledir,
                                name=fn
                            )
                            
                            # TODO: maybe some other matrix comparison style difference maybe.
                            #       - like change in orthogonality, change in eigen-vectors ... etc.
            
            def get_time_str( msg ):
                rt = v[ msg + '_end' ] - v[ msg + '_start' ]
                rt_hrs = rt // 3600
                v[ msg + '_hrs'] = rt_hrs
                rt_min = ( rt - 3600 * rt_hrs ) // 60
                v[ msg + '_min'] = rt_min
                rt_sec = ( rt - 3600 * rt_hrs - 60 * rt_min )
                rt_rem = math.fmod( rt_sec , 1.0e2 ) - math.floor( rt_sec )
                v[ msg + '_rem'] = rt_rem
                rt_sec = math.floor( rt_sec )
                v[ msg + '_sec'] = rt_sec
                return '{:.1f} (hrs)  {:.1f} (min)  {:.1f} (sec)  {:.2f} (0.01 sec)'.format( rt_hrs , rt_min , rt_sec , rt_rem )
            
            ## Print run-times
            if 'train_time_start' in v and 'train_time_end' in v: readme['Training RunTime: '] = get_time_str( 'train_time' )
            if 'pred_time_start' in v and 'pred_time_end' in v: readme['Prediction RunTime: '] = get_time_str( 'pred_time' )
            if 'eval_time_start' in v and 'eval_time_end' in v: readme['Evalutaion RunTime: '] = get_time_str( 'eval_time' )
            
            readme[''] = ''
            readme['Metrics'] = ''
            
            readme_metrics = []
            if 'hist' in v:
                hist = v['hist']
                v['fit'] = hist.params
                metrics = []
                legend = []
                for hk, hv in hist.history.items():
                    v[hk] = hv
                    nphv = np.asarray( hv )
                    metrics.append( nphv )
                    filename = os.path.join(rfiledir,'train_'+hk+'.npy')
                    np.save( filename , nphv , allow_pickle=True )
                    legend.append( hk )
                    if 'val_' not in hk: readme_metrics.append( hk )      # Add string for later text file 'readme'
                    
                    tmpstr = 'Training '
                    mx = np.round( np.max( nphv ) , 5 )
                    mxst = tmpstr + 'Max: ' + hk
                    readme[ mxst ] = mx
                    setcsv( mxst , mx )
                    
                    mn = np.round( np.min( nphv ) , 5 )
                    mnst = tmpstr + 'Min: ' + hk
                    readme[ mnst] = mn
                    setcsv( mnst , mn )
                    
                    av = np.round( np.mean( nphv ) , 5 )
                    avst = tmpstr + 'Ave: ' + hk
                    readme[ avst ] = av
                    setcsv( avst , av )
                    
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
                            dir=rfiledir,
                            name='metrics'
                           )
            
            ''' - - - - - SAVE EVALUATION RESULTS - - - - - '''
            # Prediction values and plot
            if 'pred' in v:
                
                try:
                    # Save predictions as np.ndarray in file
                    filename = os.path.join(rfiledir,'pred.npy')
                    np.save( filename , np.asarray( v['pred'] ) , allow_pickle = True )
                    v['pred_dir'] = filename
                    
                    # Plot and save results based on ground truth
                    gt , est = self.groundtruth_callback( v['pred'] , max_plot_len )
                    ave = np.mean( est ) * np.ones_like( est )
                    x_vals = np.arange(gt.shape[0])
                    legend=['Ground Truth','Predicted']
                    utils.plot(
                                x_vals,
                                [ gt , est , ave ],
                                colors=['b','r','g'],
                                linestyles=['-','--','--'],
                                title=self.name.capitalize() + ' Results - '+v['name'] + ' - Gen: ' + str(gen),
                                xlabel='Time',
                                legend=legend,
                                show=show_plot,
                                save=save_plot,
                                dir=rfiledir,
                                name=v['name']
                               )
                except:
                    pass
            
            # Save Evaluation
            if 'eval' in v:
                
                if not isinstance(v['eval'],list): v['eval'] = [v['eval']]
                
                # Add to summary text file
                rmmlen = len(readme_metrics)
                if len(v['eval']) == rmmlen:
                    for i in range( rmmlen ):
                        evlstr = 'Eval '+readme_metrics[i]
                        readme[evlstr] = np.round( np.asarray( v['eval'][i] ) , 5 )
                        setcsv( evlstr , readme[evlstr] )
                np.save(
                        os.path.join(rfiledir,'eval.npy'),
                        np.asarray(v['eval']),
                        allow_pickle=True
                        )
                v['eval_dir'] = filename
            
            def set_csv_times( msg , key ):
                if key+'_hrs' in v : setcsv( msg + ' (hrs)' , v[ key + '_hrs' ] )
                if key+'_min' in v : setcsv( msg + ' (min)' , v[ key + '_min' ] )
                if key+'_sec' in v : setcsv( msg + ' (sec)' , v[ key + '_sec' ] )
                if key+'_rem' in v : setcsv( msg + ' (1/100 sec)' , v[ key + '_rem'] )
            
            set_csv_times( 'Training' , 'train_time' )
            set_csv_times( 'Prediction' , 'pred_time' )
            set_csv_times( 'Evaluation' , 'eval_time' )
            
            # Write a summary that contains all relevant information to the generation folder
            rmmw = utils.readme_writer( filename = 'Summary' , filedir = v['gdir'] , **readme )
            if verbose > 1: print( rmmw )
            
            # Delete all non-json-able dictionary elements.
            if 'model' in v: del v['model']
            if 'callbacks' in v: del v['callbacks']
            if 'hist' in v: del v['hist']
            if 'pred' in v: del v['pred']
            if 'eval' in v: del v['eval']
            if 'finwgt' in v: del v['finwgt']
            if 'iniwgt' in v: del v['iniwgt']
            
            if 'gdir' in v : setcsv( 'dir' , v['gdir'] )
            
            csvdir = os.path.join( v['dir'] , 'metrics.csv' )
            
            ## Check Load and Save the Model Meta File ##
            model_meta = load_meta( v['dir'] )
            if model_meta['csvset'] == 0:
                with open( csvdir , 'w' ) as csvfile:
                    csvfile.write( ','.join(csvhdr) + ',\n' )
                    csvfile.write( ','.join( csvstr ) + ',\n' )
                model_meta['csvset'] = 1
            else:
                with open( csvdir , 'a' ) as csvfile:
                    csvfile.write( ','.join( csvstr ) + ',\n' )
            ret = save_meta( model_meta , v['dir'] )
            
            ''' Saving Training Meta '''
            # Save the remaining structure
            jsonstr = json.dumps( v )
            filename = os.path.join(v['gdir'],'meta.json')
            with open(filename,'w') as fileobj:
                fileobj.write(jsonstr)
                
        return self



