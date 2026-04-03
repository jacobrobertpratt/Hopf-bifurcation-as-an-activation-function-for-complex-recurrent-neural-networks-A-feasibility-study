''' Standard Imports '''
import os

''' Special Imports '''
import numpy as np

import tensorflow as tf

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

'''
Also get Citation for NeuralODE paper. Chen et al., 2018
Citation for tensorflow Probability:
J. V. Dillon et al., “TensorFlow Distributions.” arXiv, Nov. 28, 2017. doi: 10.48550/arXiv.1711.10604.      '''
import tensorflow_probability as tfp


''' Local Imports '''
from proj_utils import _print


@tf.function
def stack_to_complex(stk):
    re, im = tf.unstack(stk)
    return tf.complex(re, im)

@tf.function
def complex_to_stack(cpx):
    return tf.stack([tf.math.real(cpx), tf.math.imag(cpx)])


@tf.function
def calc_cpx_Hopf(t,z,a,b):
    re_z, im_z = tf.unstack(z)
    re_a, im_a = tf.unstack(a)
    re_b, im_b = tf.unstack(b)
    z = tf.complex(re_z,im_z)
    a = tf.complex(re_a,im_a)
    b = tf.complex(re_b,im_b)
    
    dz = (a + b*tf.math.conj(z)*z)*z
    
    return tf.stack([tf.math.real(dz),tf.math.imag(dz)])

@tf.function
def hopf_Diff_EQ(t,state,re_a,im_a,re_b,im_b):
    re_z, im_z = tf.unstack(state)
    z_stk = tf.stack([re_z,im_z])
    a_stk = tf.stack([re_a,im_a])
    b_stk = tf.stack([re_b,im_b])
    re_dz, im_dz = tf.unstack(self.calc_cpx_Hopf(t,z_stk,a_stk,b_stk))
    return tf.stack([re_dz,im_dz])

@tf.function
def complex_hopf_ODE(z,a,b):
    t_0 = 0.
    t_1 = 2*np.pi
    t_r = tf.linspace(t_0, t_1, num=int(t_1)*4)
    re_z, im_z = tf.math.real(z), tf.math.imag(z)
    re_a, im_a = tf.math.real(a), tf.math.imag(a)
    re_b, im_b = tf.math.real(b), tf.math.imag(b)
    ode_state = tf.stack([re_z,im_z])
    ode_const = {'re_a':re_a,'im_a':im_a,'re_b':re_b,'im_b':im_b}
    DormPrnc = tfp.math.ode.DormandPrince()
    ode = DormPrnc.solve(
                          self.hopf_Diff_EQ,
                          t_0,
                          ode_state,
                          solution_times=t_r,
                          constants=ode_const
                         )
    ode_t = ode.states[-1]
    re_zt, im_zt = tf.unstack(ode_t)
    zt = tf.complex(re_zt,im_zt)
    return zt


@tf.function
def polar_hopf_DiffEQ(t, z, a):
    zz = (z*z)
    dz = (a - zz)*z
    tf.debugging.check_numerics(dz,'')
    return dz


@tf.function
def polar_hopf_ODE(n, z, a):
    
    t_0 = 0.
    t_1 = 2*np.pi
    t_r = tf.linspace(t_0, t_1, num=int(t_1)*4)
    
    ode_state = z
    ode_const = {'a':a}
    DormPrnc = tfp.math.ode.DormandPrince()
    ode = DormPrnc.solve(
                          polar_hopf_DiffEQ,
                          t_0,
                          ode_state,
                          solution_times=t_r,
                          constants=ode_const
                         )
    z_t = ode.states[-1]
    
    return z_t


@tf.keras.utils.register_keras_serializable('activations')
class HopfBifur(tf.keras.layers.Layer):
    """
    
    """
    def __init__(self, **kwargs):
        
        super(HopfBifur, self).__init__(**kwargs)
        
        # TODO: Add time-range, t0 -> tn
        # TODO: granularity -> currently at 4 in tf.linspace 't_r' value.
        # TODO: Other ODESolver Specifications
        # TODO: Maybe find a distribution strategy, or ensure 'GPU' usage to speed things up.
    
    def build(self, input_shape):
        
        self.zeros = tf.constant(
            tf.zeros(shape=input_shape),
            dtype=self.dtype
        )
        
        self.ones = tf.constant(
            tf.ones(shape=input_shape),
            dtype=self.dtype
        )
        
        self.t_n = input_shape[-1]
        
        self.built=True
    
    # Real-valued hopf-bifurcation activation.
    @tf.function
    def run_polar_hopf(self, states, alphas):
        return polar_hopf_ODE(self.t_n, states, alphas)
    
    
    # Complex-valued hopf-bifurcation activation.
    @tf.function
    def call_complex_hopf(self, state, alpha, beta=None):
        assert False, 'Not yet implemented'
        return state
    
    
    def call(self, states, alphas):
        
#        '''
        if (states.dtype == alphas.dtype == tf.float32) or (states.dtype == alphas.dtype == tf.float64):
            outputs = self.run_polar_hopf(states, alphas)
#        elif (state.dtype == alpha.dtype == tf.float32) or (state.dtype == alpha.dtype == tf.float64):
#            outputs = self.call_complex_hopf(states, alphas)
#        '''
        else:
            outputs = states
#        output = tf.keras.activations.tanh(states)
        
        return outputs
    
    
    
    
    
    
    def get_config(self):
        config = {}
        base_config = super(HopfBifur, self).get_config()
        if 'input_spec' not in base_config:
            config['input_spec'] = self.input_spec[0].get_config()
        return dict(list(base_config.items()) + list(config.items()))

