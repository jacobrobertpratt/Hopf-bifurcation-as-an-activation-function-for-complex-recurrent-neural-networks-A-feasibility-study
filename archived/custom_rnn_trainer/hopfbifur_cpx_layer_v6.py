import os
import math
import collections
import time 

import numpy as np
from scipy.stats import unitary_group

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

''' Local Imports '''
import proj_utils as utils
#from proj_utils import _print
from activations import HopfBifur
from initializers import Hermitian , Unitary

'''
TODO:
- Add multiple cells
- Copy this code over to the use_RNN_class version to see if performance is different.
- Try and see what the output of the ODESover looks like and if we can use some regularizer
    on that data for the output state. 
- Make stuff into complex next.
'''


def _print( msg , arr , **kwargs):
    
    def p_func( msg , arr , **kwargs):
        if arr.dtype.is_complex:
            tf.print('\n'+msg+':\nreal:\n', tf.math.real(arr),'\nimag:\n',tf.math.imag(arr),'\nshape:',arr.shape,'  dtype:',arr.dtype,'\n',**kwargs)
        else:
            tf.print('\n'+msg+':\n', arr,'\nshape:', arr.shape,'  dtype:', arr.dtype,'\n',**kwargs)
    
    if tf.nest.is_nested(arr):
        tf.nest.map_structure(p_func,[msg]*len(arr) , arr)
    else:
        p_func( msg , arr , **kwargs)


@tf.keras.utils.register_keras_serializable('hopfbifur_cpx_layer')
class HopfBifurCpxRNNCell(tf.keras.layers.Layer):
    
    def __init__(
                    self,
                    unit_size=8,
                    activation=None,
                    **kwargs
                  ):
        
        super(HopfBifurCpxRNNCell, self).__init__(**kwargs)
        
        # Size of input cells
        self.unit_size = unit_size
        
        # Set default activation or input activation.
        self.activation = HopfBifur( dtype=self.dtype ) if activation is None else activation
        
        
    
    def build(self, input_shape):
        
        batch_size = input_shape[0]
        input_size = input_shape[-1]
        
#        ''' Creates a unitary weight matrix from
        def gen_unitary_weight_matrix(sz):
            wgt = unitary_group.rvs(sz)
            wgt = tf.linalg.logm(wgt)
            wgt = (wgt + tf.math.conj(wgt)[::-1,::-1])/2.
            wgt = wgt / tf.math.pow(tf.linalg.det(wgt),(1/sz))
            return tf.linalg.expm(wgt)
        
        unitary_wgt_ini = gen_unitary_weight_matrix(input_shape[-1])
        log_wgt_ini = tf.linalg.logm(unitary_wgt_ini)
        
        self.A = self.add_weight(
            name='A_herm',
            shape=( batch_size , self.unit_size, self.unit_size ),
            dtype=self.dtype,
            initializer=Hermitian( self.unit_size ),
            trainable=True
        )
#        print('self.A',self.A)
        
        self.B = self.add_weight(
            name='B_herm',
            shape=( batch_size , self.unit_size, self.unit_size),
            dtype=self.dtype,
            initializer=Hermitian( self.unit_size ),
            trainable=True
        )
#        print('self.B',self.B)
#        exit(0)
        
        super(HopfBifurCpxRNNCell, self).build(input_shape)
        
        self.built=True


    @tf.custom_gradient
    def map_unitary_kernel(self, input, kernel):
        output = tf.linalg.matmul( input , kernel )
        def grad(gL):
            gL_cT = tf.transpose(tf.math.conj(gL),[0,2,1])
            J = tf.linalg.matmul( gL_cT, input)
            J_cT = tf.math.conj(tf.transpose(J,[0,2,1]))
            lMJ = tf.linalg.matmul( kernel, J_cT)
            rMJ = tf.linalg.matmul( lMJ, kernel)
            J_sub_rMJ = tf.math.divide( tf.math.subtract(J, rMJ), 2.)
            lMJcT = tf.math.conj(tf.transpose(kernel,[0,2,1]))
            gk = tf.linalg.matmul( lMJcT, J_sub_rMJ)
            gi = tf.linalg.matmul( gL, lMJcT)
            return gi, gk
        return output, grad


#    @tf.custom_gradient
    def map_hermitian_kernel(self, input, kernel):
        
        rshp_vec = tf.reshape( input , [kernel.shape[0],1,kernel.shape[-1]] )
        output = tf.linalg.matmul( rshp_vec , kernel )
        
        '''
        def grad( gL ):
            
            evals, evecs = tf.linalg.eigh( kernel )
            expvals = tf.math.exp(evals)
#            _print( 'expvals' , expvals )
            
            gL_T = tf.linalg.matrix_transpose( gL )
            conj_gL_T = tf.math.conj( gL_T )
            
            rshp_vec_T = tf.linalg.matrix_transpose( rshp_vec )
            rshp_conj_vec_T = tf.math.conj( rshp_vec_T )
            
            kernel_T = tf.linalg.matrix_transpose( kernel )
            
            E_T = tf.linalg.matrix_transpose( evecs )
            conj_E_T = tf.math.conj( E_T )
            
            rshp_evals = tf.reshape( evals , output.shape )
            rshp_expvals = tf.reshape( expvals , output.shape )
            evals_T = tf.linalg.matrix_transpose( rshp_evals )
            expvals_T = tf.linalg.matrix_transpose( rshp_expvals )
            
            # Calculate the G matrix 
            diff_expvals = tf.math.subtract( expvals_T , rshp_expvals )
            diff_evals = tf.math.subtract( evals_T , rshp_evals )
#            _print( 'diff_expvals' , diff_expvals )
#            _print( 'diff_evals' , diff_evals )
            
            G_offdiag = tf.math.divide_no_nan( diff_expvals , diff_evals)       # i neq j
            G_offdiag = tf.complex(
                tf.math.real( G_offdiag ),
                tf.math.abs( tf.math.imag( G_offdiag ) )
            ) # Take care of sign stuff
            
            G_diag = tf.linalg.diag( expvals , k = 0 )
#            _print( 'G_diag' , G_diag )
            
            G = G_offdiag + G_diag
#            _print( 'G' , G )
            
            # Generate the V - hermitian matrix from upstream gradient and
            inr_vec = tf.linalg.matmul( rshp_vec , rshp_conj_vec_T )
            norm_vec = tf.math.sqrt( inr_vec )
            unit_vec = tf.math.divide_no_nan( rshp_vec , norm_vec )
#            _print( 'unit_vec' , unit_vec )
            
#            gL_unit_vec = tf.linalg.matmul( conj_gL_T , unit_vec )
            V = tf.linalg.matmul( conj_gL_T , unit_vec )
            
            V_bar = tf.linalg.matmul( tf.linalg.matmul( conj_E_T , V ) , evecs )
#            _print( 'V_bar' , V_bar )
            
            G_dot_V = tf.math.multiply( G , V_bar )
            Dv_K = tf.linalg.matmul( tf.linalg.matmul( evecs , G_dot_V ) , conj_E_T )
            
            # Create Dv_K normal projection 
            Dv_K_outer = tf.linalg.matmul( tf.linalg.matrix_transpose( tf.math.conj( Dv_K ) ) , Dv_K )
            Dv_K_inner = tf.linalg.matmul( Dv_K , tf.linalg.matrix_transpose( tf.math.conj( Dv_K ) ) )
            Dv_K_norm = tf.linalg.trace( Dv_K_inner )
            Dv_K_norm = tf.expand_dims( tf.expand_dims( Dv_K_norm , 1 ) , 1 )
            
            Dv_K_projN = tf.math.divide_no_nan( Dv_K_outer , Dv_K_norm )
            
            gk = -Dv_K
            gi = tf.linalg.matmul( gL , tf.math.conj( kernel_T ) )
            gi = tf.reshape( gi , input.shape )
            return gi, gk
        '''
        
        return output #, grad


    def call(self, inputs, states, training=False):
        
#        tf.print('\n'+'- '*20+' Cell '+'- '*20+'\n')
        
        tf.print('inputs',type(inputs))
        tf.print('states',type(states))
        
        prev_state = states[0] if tf.nest.is_nested(states) else states
        
        '''
        other_inputs= inputs[1]
        _print('other inputs',other_inputs)
        
        other_states= states[1]
        _print('other states',other_states)
        
        # Check for further nesting & un-nest if needed.
        prev_state = states[0] if tf.nest.is_nested(states) else states
        new_inputs = inputs[0] if tf.nest.is_nested(inputs) else inputs
        _print( 'new_inputs' , new_inputs )
        _print( 'prev_state' , prev_state )
        
        
        
        kern_map = self.map_hermitian_kernel( new_inputs , self.A)
        _print('kern_map',kern_map)
        
        rec_kern_map = self.map_hermitian_kernel( prev_state , self.B)
        _print('rec_kern_map',rec_kern_map)
        
        new_state = kern_map + rec_kern_map
#        _print('new_state',new_state)
        
        output = self.activation( new_state )
        
        new_state = tf.reshape( new_state , prev_state.shape )
        output = tf.reshape( output , new_inputs.shape )
        
        # check for shape compatibility & re-nest if the previous one was nested.
        new_state = [new_state] if tf.nest.is_nested(states) else new_state
        '''
        
        new_state = prev_state + 0.25
        
        new_state = [new_state] if tf.nest.is_nested(states) else new_state
        
        return inputs , new_state
    
    
    def get_config(self):
        base_config = super(HopfBifurCpxRNNCell, self).get_config()
        config = {
                    "units": self.units,
                    "activation": tf.keras.activations.serialize(self.activation)
                    }
        return dict(list(base_config.items()) + list(config.items()))








@tf.keras.utils.register_keras_serializable('hopfbifur_cpx_layer')
class HopfBifurCpxRNNLayer(tf.keras.layers.Layer):
    
    def __init__(
                   self,
                   unit_count=1,
                   unit_size=None,
                   activation=None,
                   **kwargs
                  ):

        super(HopfBifurCpxRNNLayer, self).__init__(**kwargs)
        
        # How many cells we have.
        self.unit_count = unit_count
        assert self.unit_count > 0, 'The number of units the model needs must be greater than 0. (Check HopfBifurCpxRNNLayer input paramters.)'
        
        # The state size of each cells, if None then defaults to square-root of input dimensions and cast as an integer.
        self.unit_size = unit_size
        
        # To hold the cell(s) for the model.
        self.cells = None
        
        # Set the activation function
        self.activation = activation
        
    
    
    def build(self, input_shape):
        
        batch_size = input_shape[0]
        time_size = input_shape[-2]
        input_size = input_shape[-1]
        
        # Check and see if data is in time-major format (shouldn't be, but who knows)
        self.time_major = True if time_size > 1 else False
        
        # Set unit_size if not specified by user.
        if self.unit_size is None:
            self.unit_size = int(math.sqrt(float(input_shape[-1])))
            if self.unit_size < 8: self.unit_size = 8
            print('\nHopfBifurCpxRNNLayer: input unit_size was not specified, defaulting to a unit_size of 8\n')
        
        # Generate cell -> Only use 1 for now, until it's working better.
        if self.cells is None:
            self.cells = []
            for i in range(self.unit_count):
                cell = HopfBifurCpxRNNCell(
                    unit_size=self.unit_size,
                    activation=self.activation,
                    dtype=self.dtype,
                    name='hopf_cell_%s' % i
                )
                self.cells.append(cell)
        
        # From tf.keras.layers.basernn
        def get_step_input_shape(shape):
            if isinstance(shape, tf.TensorShape):
                shape = tuple(shape.as_list())
            # remove the timestep from the input_shape
            return (shape[0],) + shape[2:]
        
        step_input_shape = tf.nest.map_structure(
            get_step_input_shape, input_shape
        )
        
        # Build cell(s)
        if not self.cells[0].built:
            for cell in self.cells:
                with tf.name_scope(cell.name):
                    cell.build( step_input_shape )         # Last build shape -> self.cells.build( input_shape[1::] )
                    assert cell.built , 'HopfBifurCpxRNNLayer failed to build.'
        
#        assert self.unit_size == self.cells.unit_size, 'Cell unit size and layer unit size are miss-matched. Check code.'
        
        # From cells unit-size, build layer state_shape.
        state_shape = [ input_shape[0], self.unit_size ]
        ones_init = tf.ones(
            shape=state_shape,
            dtype=self.dtype
        )
        self.states = tf.Variable(
            ones_init,
            trainable=False,
            name='states'
        )
        
        self.global_state = tf.Variable(
            initial_value=tf.zeros( shape=( 8 , 8 ) , dtype=self.dtype ),
            shape=( 8 , 8 ),
            dtype=self.dtype,
            trainable=False,
            synchronization=tf.VariableSynchronization.ON_WRITE,
            name='global_state'
        )
        
        super( HopfBifurCpxRNNLayer, self ).build( input_shape )
        
        self.built = True
        
        
    def set_global_state( self , curr_state , all_states):
    
        curr_state = curr_state[0] if tf.nest.is_nested(curr_state) else curr_state
        
        new_states = curr_state + ( tf.math.reduce_mean( all_states , axis=0 ) / tf.math.reduce_mean(all_states) )
        
        new_states = [new_states] if tf.nest.is_nested(all_states) else new_states
        
        return new_states
    
    
    
    def call(self, inputs, training=False):
        
#        tf.print('\n'+'-'*25+' Layer '+'-'*25+'\n')
        
        inputs = tf.cast( inputs, dtype=self.dtype )
        
        # TODO: Apply forward mapping with state stuff here
        #       return_state must be True for this.
        
        ''' Perform the RNN operation on batch
        cell_outs, cell_outputs, cell_states = self.backend_rnn_call(
            self.cells[0],
            inputs,
            self.states,
            training=training,
            dbg=False
        )
        
        self.states.assign(cell_states)
        '''
#        inputs = tf.unstack(inputs)
        
        cell_outs , cell_states , global_state = self.local_rnn_call(
            self.cells,
            inputs,
            self.global_state,
            training=training,
            agg_func = self.set_global_state
        )
#        _print('cell_outs',cell_outs)
#        _print('cell_states',cell_states)
#        _print('global_state',global_state)
        
        # TODO: Apply aggregation stuff and put back into states.
        #   Using Multiple Cells and cells States now.
        
        tf.ensure_shape( global_state , self.global_state.shape )
        self.global_state.assign( global_state )
#        '''
        
        outputs = cell_outs  # tf.math.reduce_mean(cell_outs, axis=0)
        outputs = outputs[0] if tf.nest.is_nested(outputs) else outputs
        
        # TODO: outputs mapping ( convert from complex -> floating )
        
        outputs = tf.math.real( tf.math.abs( outputs ) )
#        _print( 'outputs' , outputs )
        
        return outputs
    
    
    
    def get_config(self):
        base_config = super(HopfBifurCpxRNNLayer, self).get_config()
        if 'cells' in base_config: del base_config['cells']
        return base_config
    
    @classmethod
    def from_config(self, config):
        return self(**config)
        
        
        
    def local_rnn_call( self , cells , inputs , state , training=False , agg_func=tf.math.reduce_mean ):
        
        # Make into inputs list
        if not tf.nest.is_nested(inputs): inputs = tf.unstack(inputs)
        elif len(inputs) == 1: inputs = tf.unstack(inputs[0])
        
        assert len(inputs) > 0, 'Inputs given to local_rnn_call was length 0.'
        
        time_steps = 1
        input_shape = tf.shape(inputs[0])
        time_steps_t = 1
        
        # Get cell call function
#        cell_call_fn = cells.__call__
        cell_call_fns = [ cell.__call__ for cell in self.cells ]
        
        # Build input TensorArray to be passed to while-loop
        input_tensarr = tf.TensorArray(
            dtype = inputs[0].dtype,
            size = len(inputs),
            tensor_array_name = 'curr_input_tensarr'
        )
        for i, _input in enumerate(inputs):
            input_tensarr = input_tensarr.write( i , _input )
        
        # Get cell output shape values
#        _tmp_out , _tmp_state = cell_call_fn( inputs[0], state , training=False )
        _tmp_out , _tmp_state = cell_call_fns[0]( inputs[0], state , training=False )
        
        # Build return output and state tensor array
        out_shape = [len(cell_call_fns)]+list(_tmp_out.shape)
        new_out_ta = tf.TensorArray(
            dtype=_tmp_out.dtype,
            size = time_steps_t,
            element_shape=out_shape,
            tensor_array_name='new_output_tensarr'
        )
        
        state_shape = [len(cell_call_fns)]+list(_tmp_state.shape)
        new_state_ta_t = tf.TensorArray(
            dtype=_tmp_state.dtype,
            size = time_steps_t,
            element_shape=state_shape,
            tensor_array_name='new_state_tensarr'
        )
        
        time = tf.constant(0, dtype="int32", name="time")
        
        # We only specify the 'maximum_iterations' when building for XLA since
        # that causes slowdowns on GPU in TF.
        max_iterations = None
        
        while_loop_kwargs = {
            "cond": lambda time, *_: time < time_steps,
            "maximum_iterations": max_iterations,
            "parallel_iterations": 32,
            "swap_memory": True
        }
        
        def _step( time , output_ta_t, state_ta_t , *state ):
            
            curr_input = input_tensarr.read(time)
            
            def cell_caller( caller ):
                return caller( curr_input , state , training=training )

            # Call Cell
            called_cells = tf.nest.map_structure( cell_caller , cell_call_fns )
            
            # Map cells to output TensorArrays
            cell_outs = tf.nest.flatten([ o for o , s in called_cells])
            cell_states = tf.nest.flatten([ s for o , s in called_cells])
            cell_outs = tf.stack(cell_outs)
            cell_states = tf.stack(cell_states)
            
            new_state = agg_func( state , cell_states )
            new_state = [new_state] if not tf.nest.is_nested(new_state) else new_state
            
            # Write to TensorArrays
            output_ta_t = output_ta_t.write( time , cell_outs )
            state_ta_t = state_ta_t.write( time , cell_states )
            
            return (time + 1 , output_ta_t , state_ta_t ) + tuple(new_state)
        
        final_outputs = tf.compat.v1.while_loop(
            body=_step,
            loop_vars=( time, new_out_ta , new_state_ta_t ) + tuple([state]),
            **while_loop_kwargs,
        )
        ftime , cell_outs , cell_states , global_state = final_outputs
        
        cell_outs = cell_outs.stack()
        cell_states = cell_states.stack()
        global_state = global_state
        
        cell_outs = tf.reshape( cell_outs , out_shape )
        cell_states = tf.reshape( cell_states , state_shape )
        
        return cell_outs , cell_states , global_state
        


#    @tf.function
    def backend_rnn_call(self, cell, inputs, states, training=False, dbg=False):
        ''' Input Params.
            inputs: must have a (batch, time, values) shape.
            states: must have a (batch, values) shape.
        '''
        
        # Calculate timesteps (grabs the batch dimension)
        
        input_shape = tf.keras.backend.int_shape(inputs)
        timesteps = input_shape[0] if self.time_major else input_shape[1]
        
        nested_states = [states] if not tf.nest.is_nested(states) else states
        
        # Get cell's calling function
        cell_call_fn = ( cell.__call__ if callable(cell) else cell.call )
        
        # Callback for backend.rnn() operations.
        def step(step_inputs, step_states):
            step_states = step_states[0]
            output, new_states = cell_call_fn(step_inputs, step_states, training=training)
            if not tf.nest.is_nested(new_states): new_states = [new_states]
            return output, new_states
        
        lastout, outputs, states = tf.keras.backend.rnn(
            step,
            inputs,
            nested_states,
            return_all_outputs=True
        )

        if tf.nest.is_nested(lastout):
            _lastout = lastout[0]
            if dbg: tf.print('lastout is nested.')
        else:
            _lastout = lastout
        
        if tf.nest.is_nested(outputs):
            _outputs = outputs[0]
            if dbg: tf.print('outputs is nested.')
        else:
            _outputs = outputs
        
        if tf.nest.is_nested(states):
            _states = states[0]
            if dbg: tf.print('states is nested.')
        else:
            _states = states
        
#        _print('_lastout',_lastout)
#        _print('_outputs',_outputs)
#        _print('_states',_states)
        
        return (_lastout, _outputs, _states)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
'''




class HopfCell(tf.keras.layers.Layer):
    
    def build(self, input_shape):
        
        super(HopfCell,self).build(input_shape=input_shape)

        in_sz = input_shape[-1]
        self.in_sz = in_sz
        
        root = np.exp(-2j*np.pi*np.arange(in_sz)/in_sz).reshape(-1,1)
        self.root = tf.constant(root,dtype=tf.complex128)
        
        # Normal dft
        dft = root**np.arange(in_sz)
        self.dft = tf.constant(dft,dtype=tf.complex128)
        
        # Unitary dft
        udft = dft / np.sqrt(in_sz)
        self.udft = tf.constant(udft,dtype=tf.complex128)
        
        zeros = np.zeros(shape=(in_sz,1))
        self.zeros  = tf.constant(zeros,dtype=tf.complex128)
        
        ones = np.ones(shape=(in_sz,1))
        self.ones  = tf.constant(ones,dtype=tf.complex128)
        
        self.eye = tf.constant(np.eye(in_sz),dtype=tf.complex128)
        
        def make_unitary_zero_det_weight(sz,nme='U',do_save=True):
            
            def gen_wgt_matrix(t_sz):
                wgt = unitary_group.rvs(t_sz)
                wgt = tf.linalg.logm(wgt)
                wgt = (wgt + tf.math.conj(wgt)[::-1,::-1])/2.
                wgt = wgt / tf.math.pow(tf.linalg.det(wgt),(1/in_sz))
                wgt = tf.linalg.expm(wgt)
                return wgt
            
            if do_save is True:
                if os.path.exists('rnn_mat_'+nme+'.npy'):
                    wgt = np.load('rnn_mat_'+nme+'.npy',allow_pickle=True)
                    # Check saved shape and replace if not the same.
                    if wgt.shape[-1] != sz:
                        wgt = gen_wgt_matrix(sz)
                        np.save('rnn_mat_'+nme+'.npy',wgt,allow_pickle=True)
#                        tf.print('Generating new matrix for '+nme)
                else:
                    wgt = gen_wgt_matrix(sz)
                    np.save('rnn_mat_'+nme+'.npy',wgt,allow_pickle=True)
#                    tf.print('Generating new matrix for '+nme)
            else:
                wgt = gen_wgt_matrix(sz)
#                tf.print('Generating new matrix for '+nme)
            return wgt
        
        def make_random_normal_vec(sz,nme='z',do_save=True):
            if do_save is True:
                if os.path.exists('rnn_vec_'+nme+'.npy'):
                    vec = np.load('rnn_vec_'+nme+'.npy',allow_pickle=True)
                    # Check saved shape and replace if not the same.
                    if vec.shape[0] != sz:
                        vec = np.random.rand(sz).reshape(-1,1)
                        np.save('rnn_vec_'+nme+'.npy',vec,allow_pickle=True)
#                        tf.print('Generating new vector for '+nme)
                else:
                    vec = np.random.rand(sz).reshape(-1,1)
                    np.save('rnn_vec_'+nme+'.npy',vec,allow_pickle=True)
#                    tf.print('Generating new vector for '+nme)
            else:
                vec = np.random.rand(sz).reshape(-1,1)
#                tf.print('Generating new vector for '+nme)
            return vec
        
        def make_range_vec(sz,nme='z',do_save=True):
            if do_save is True:
                if os.path.exists('rnn_vec_'+nme+'.npy'):
                    vec = np.load('rnn_vec_'+nme+'.npy',allow_pickle=True)
                    # Check saved shape and replace if not the same.
                    if vec.shape[0] != sz:
                        vec = np.arange(1,sz+1).reshape(-1,1)
                        np.save('rnn_vec_'+nme+'.npy',vec,allow_pickle=True)
#                        tf.print('Generating new vector for '+nme)
                else:
                    vec = np.arange(1,sz+1).reshape(-1,1)
                    np.save('rnn_vec_'+nme+'.npy',vec,allow_pickle=True)
#                    tf.print('Generating new vector for '+nme)
            else:
                vec = np.arange(1,sz+1).reshape(-1,1)
#                tf.print('Generating new vector for '+nme)
            return vec
        
        # Set to True if we want to reuse the initialization matrices for common training
        _save = False
        _train = True
        _sz = in_sz
        

#        A = make_unitary_zero_det_weight(_sz,nme='A',do_save=_save)
#        A = (A + tf.math.conj(A).T)/2.
#        self.A = tf.Variable(   A,
#                                name='A_hrm',
#                                dtype=tf.complex128,
#                                trainable=_train
#                                )
        
#        B = make_unitary_zero_det_weight(_sz,nme='B',do_save=_save)
#        B = (B + tf.math.conj(B).T)/2.
#        self.B = tf.Variable(   B,
#                                name='B_hrm',
#                                dtype=tf.complex128,
#                                trainable=_train
#                                )
       
#        G = make_unitary_zero_det_weight(_sz,nme='G',do_save=_save)
#        self.G = tf.Variable(   G,
#                                name='G_unt',
#                                dtype=tf.complex128,
#                                trainable=_train
#                                )
        
        U = make_unitary_zero_det_weight(_sz,nme='U',do_save=_save)
        self.U = tf.Variable(   U,
                                name='U_unt',
                                dtype=tf.complex128,
                                trainable=_train
                                )
        
        V = make_unitary_zero_det_weight(_sz,nme='V',do_save=_save)
        self.V = tf.Variable(   V,
                                name='V_unt',
                                dtype=tf.complex128,
                                trainable=_train
                                )
        
        z = make_random_normal_vec(in_sz,'z',do_save=_save)
        z = self.U @ z
        self.z = tf.Variable(
                              initial_value=z,
                              name='z',
                              dtype=tf.complex128,
                              trainable=False
                             )
        
        self.counter = tf.Variable([0],dtype=tf.float64,trainable=False)
        
        self.built = True


#    @tf.function
    @tf.custom_gradient
    def unitary_map(self,M,v):
        Mv = M @ v
        def grad(gL):
            J = gL*tf.math.conj(v).T
            gM = (tf.math.conj(M).T @ ((J - M @ tf.math.conj(J).T @ M)/2.))
            gv = tf.zeros_like(v)
            return gM, gv
        return Mv, grad

    @tf.custom_gradient
    def herm_map(self,M,v):
        Mv = M @ v
        def grad(gL):
            lmdas, W = tf.linalg.eigh(M)
            lmdas = tf.expand_dims(lmdas,1)
            exp_lmdas = tf.math.exp(lmdas)
            G = tf.math.divide_no_nan((exp_lmdas - exp_lmdas.T),(lmdas - lmdas.T))
            J = (v @ tf.math.conj(gL).T + gL @ tf.math.conj(v).T)/2.
            VG = (tf.math.conj(W).T @ J @ W) * G
            gM = W @ VG @ tf.math.conj(W).T
            gv = tf.math.conj(M).T @ gL
            return gM, gv
        return Mv, grad


    @tf.custom_gradient
    def cos_sin_map(self,C,S,z,x):
    # C = (U + tf.math.conj(U).T)/2.    &    S = (U - tf.math.conj(U).T)/2.j
    #       where U is a unitary matrix.
        Mv = C @ z + S @ x
        def grad(gL):
            lmdas_c, W_c = tf.linalg.eigh(C)
            lmdas_s, W_s = tf.linalg.eigh(S)
            lmdas_c = tf.expand_dims(lmdas_c,1)
            lmdas_s = tf.expand_dims(lmdas_s,1)
            cos_lmdas = tf.math.cos(lmdas_c)
            sin_lmdas = tf.math.sin(lmdas_s)
            G_c = tf.math.divide_no_nan((cos_lmdas - cos_lmdas.T),(lmdas_c - lmdas_c.T))
            G_s = tf.math.divide_no_nan((sin_lmdas - sin_lmdas.T),(lmdas_s - lmdas_s.T))
            J_c = (z @ tf.math.conj(gL).T + gL @ tf.math.conj(z).T)/2.
            J_s = (x @ tf.math.conj(gL).T + gL @ tf.math.conj(x).T)/2.
            VG_c = (tf.math.conj(W_c).T @ J_c @ W_c) * G_c
            VG_s = (tf.math.conj(W_s).T @ J_s @ W_s) * G_s
            gC = W_c @ VG_c @ tf.math.conj(W_c).T
            gS = W_s @ VG_s @ tf.math.conj(W_s).T
            gz = tf.zeros_like(z)
            gx = tf.zeros_like(x)
            return gC, gS, gz, gx
        return Mv, grad


    @tf.custom_gradient
    def exp_map(self,a,b):
        N = tf.math.exp(tf.math.conj(a)*b)*(-1.+0.j)
        def grad(gL):
            ga = gL*N*b
            gb = gL*N*tf.math.conj(a)
            return ga, gb
        return N, grad


#    @tf.function
    def call(self,input,training=False):
        
#        tf.print('\n-----------------------------------------------------------------------------\n',self.counter[0],')\n')
        
        sz = (0.+self.in_sz)
        
        z = self.z
        x = input
#        _print('x',x)
        
#        A = self.A
#        B = self.B
        U = self.U
        V = self.V
#        E = self.E
        
#        Az = self.herm_map(A,z)
#        Bx = self.herm_map(B,x)
#        _print('Az',Az)
#        _print('Bx',Bx)
        
        Uz = self.unitary_map(U,z)
        Vx = self.unitary_map(V,x)
#        _print('Uz',Uz)
#        _print('Vx',Vx)
        
#        a = Az + Bx
#        a = self.cos_sin_map(A,B,z,x)
        a = Uz + Vx
        b = self.ones*(-1.+0.j)
#        b = self.exp_map(Uz,Vx)
#        _print('a',a,True)
#        _print('b',b,True)
        
#        _print('z',z,True)
        
        # Map to the limit cycle #
        
        z_t = self.hopf_ODE(z,a,b)
        
#        z_t = tf.math.tanh(a)
        
#        _print('z_t',z_t,True)
        
        # Pulling the real-value and re-mapping it to the DFT fixes floating point erros that explode the model.
        self.z.assign(z_t)
        self.counter.assign_add([1.])
        
        return tf.reshape(z_t,input.shape)
    
    
class HopfLayer(tf.keras.layers.Layer):
    
    def __init__(self,**kwargs):
        
        super(HopfLayer, self).__init__(**kwargs)
        
        # Creates state_size number of RNNCells with each cell being state_size large.
#        self.cells = [MyRNNCell(self.state_size,cell_id=j,name='rnncell_'+str(j)) for j in range(self.state_size)]
        self.cell = HopfCell(name='hopf')
        
        self.counter = tf.Variable([0],dtype=tf.float64,trainable=False)



    def build(self,input_shape):
        
        super(HopfLayer,self).build(input_shape=input_shape)
        
        in_sz = input_shape[-1]
        self.in_sz = in_sz
        
        self.cell.build(input_shape[1::])
        
        self.built = True


    def call(self, input, training=False):

#        tf.print('\n-----------------------------------------------------------------------------------------------------------------------\nCount',self.counter[0],'\n')
        
        _input = tf.reshape(input,[self.in_sz,1])
#        _print('_input',_input)
        
        
        # Make multiple calls ... maybe.
        y_t = self.cell.call(_input,training=training)
#        _print('y_t',y_t)
        
#        exp = tf.math.exp(1.j*tf.math.angle(y_t))
#        _print('exp',exp)
        

#        y_hat = tf.math.conj(y_t).T @ exp
        y_hat = tf.math.conj(y_t).T @ y_t
#        _print('y_hat',y_hat)

        self.counter.assign_add([self.in_sz])
        
        return tf.math.real(y_hat) # tf.reshape(y_hat,_input.shape)



class HopfModel(tf.keras.Model):


    def __init__(self, input, output):
        
        super(HopfModel, self).__init__(input, output)
        
#        self.my_loss = MyLoss()
        
#        self.my_accu = tf.keras.metrics.KLDivergence(name='kullback_leibler_divergence', dtype=None)
#        self.my_accu = tf.keras.metrics.CosineSimilarity(name='cosine_similarity', dtype=None, axis=-1)
        
        self.my_prec = None # tf.keras.metrics.Precision(thresholds=None, top_k=None, class_id=None, name='precision', dtype=None)
        
        self.results = {}
        
        self.my_loss = None
        
        self.my_accu = None
        
#        if self.my_loss is not None:
#            self.results['loss'] = tf.constant([0.],dtype=tf.float64)
        
#        if self.my_accu is not None:
#            self.results['accuracy'] = tf.constant([0.],dtype=tf.float64)

#        if self.my_prec is not None:
#            self.results['precision'] = tf.constant([0.],dtype=tf.float64)


    def train_step(self, data):
        
#        tf.print('\n___________________________________________________START__________________________________________________\n')

        input, image = data
        input = tf.cast(input,dtype=tf.float64)
        image = tf.cast(image,dtype=tf.float64)
        
        with tf.GradientTape() as tape:
            
#            tf.print('\n-----------------------------------------------------------------------------------------------------\nPrediction:\n')
            
#            _print('HopfModel -> input',input)
            
            pred = tf.cast(self(input, training=True),dtype=input.dtype)  # Forward pass
#            pred = tf.reshape(pred,image.shape)
#            _print('HopfModel -> pred',pred)
#            _print('HopfModel -> image',image)
            
            # My Local Loss for multi-dim loss stuffs
#            loss = tf.math.pow(image - pred,2)/(0.+pred.shape[-1])
#            _print('loss',loss)
            
            loss = self.compiled_loss(image,pred)
#            loss = self.myloss
#            loss = tf.math.pow(pred - image,2)/(0.+pred.shape[-1])
#            _print('HopfModel -> loss',loss)
            
            if self.my_loss is not None:
                self.results['loss'] = tf.math.reduce_mean(loss)
    #            _print('HopfModel -> accuracy',self.my_accu.result())
            
#            if self.my_accu is not None:
#                self.my_accu.update_state(image,pred)
#                self.results['accuracy'] = self.my_accu.result()
    #            _print('HopfModel -> accuracy',self.my_accu.result())
            
#            if self.my_prec is not None:
#                self.my_prec.update_state(image,pred)
#                self.results['precision'] = self.my_prec.result()
    #            _print('HopfModel -> precision',self.my_prec.result())
            
#            tf.print('\n-----------------------------------------------------------------------------------------------------\nBackward Gradient:\n')
            
            # Compute gradients 
            vars = self.trainable_variables
#            for v in vars:
#                _print('HopfModel -> train_vars',v)
#                unt = tf.linalg.expm(v)
#                untI = tf.matmul(tf.transpose(unt),unt)
#                _print('V untI',untI)
                
            grad = tape.gradient(loss, vars)
            #tf.print('HopfModel:train_step -> Gradients:\n')
#            for g in _grad:
#                _print('HopfModel -> grads',g)
#                unt = tf.linalg.expm(g)
#                untI = tf.matmul(tf.transpose(unt),unt)
#                _print('G untI',untI)
            
        wgts = self.trainable_weights
        #tf.print('Wgts trainable_weights:\n',_wgts,'\n')
        
        self.optimizer.apply_gradients(zip(grad,wgts))
        
        # Updated trainable weights
        #tf.print('Wgts POST-apply_gradients:')
        #for _w in _wgts:
        #    printComplex('HopfModel:train_step -> _w', _w, print_shape=True)
        
        #tf.print('\n___________________________________________________END__________________________________________________\n\n')
        
        return self.results


class MyLoss(tf.keras.losses.Loss):

    def __init_(self, reduction=tf.keras.losses.Reduction.NONE, name=None, **kwargs):
        super().__init_(reduction=reduction, name=name, **kwargs)

    @tf.custom_gradient
    def call(self, true, pred):
        
        _pred = tf.cast(pred,true.dtype)
        _true = tf.reshape(true,_pred.shape)

        _loss = tf.math.squared_difference(_pred,_true)
        
        def grad(dL): # dL returns [1.,1.,...,1.] same shape as _loss.shape
            dL_dT = dL * tf.gradients(_loss,true)
            dL_dP = dL * tf.gradients(_loss,pred)
            return dL_dT, dL_dP
        
        return _loss, grad # Returns complex tensor as error


'''