
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


class MyOptimizer(tf.keras.optimizers.RMSprop):

    def __init__(self, **kwargs):
        super(MyOptimizer,self).__init__(name='MyOptimizer',**kwargs)
        self.count = tf.Variable([0],dtype=tf.int32)
    
    def update_step(self, gradient, variable):
        
        '''Update step given gradient and the associated model variable.'''
        lr = tf.cast(self.learning_rate, variable.dtype)
        
        if 'unt' in variable.name:
            # Special for Unitary #
            variable.assign(variable @ tf.linalg.expm(-lr*gradient))
        
        elif ('hrm' in variable.name):
            # Regular SGD for hermitian #
            variable.assign_add(-lr*gradient)
        
        # Update counter #
        self.count.assign_add([1])
