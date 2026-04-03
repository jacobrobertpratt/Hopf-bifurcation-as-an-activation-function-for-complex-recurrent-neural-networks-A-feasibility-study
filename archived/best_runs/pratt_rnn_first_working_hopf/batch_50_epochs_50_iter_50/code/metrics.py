
import tensorflow.compat.v2 as tf

from keras.dtensor import utils as dtensor_utils
from keras.metrics import base_metric
from keras.utils import losses_utils
from tensorflow.python.util.tf_export import keras_export

@keras_export("keras.metrics.NormRootMeanSquaredError")
class NormRootMeanSquaredError(base_metric.Mean):
    
    @dtensor_utils.inject_mesh
    def __init__(self, name="norm_root_mean_squared_error", dtype=None):
        
        super().__init__(name, dtype=dtype)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)
        
        err_sqr_diff = tf.math.squared_difference(y_pred, y_true)
        
        ideal_sq = tf.math.square(y_true)
        norm_err_sqr_diff = tf.math.divide_no_nan(err_sqr_diff,ideal_sq)
        
        return super().update_state(norm_err_sqr_diff, sample_weight=sample_weight)
    
    
    def result(self):
        return tf.sqrt(tf.math.divide_no_nan(self.total, self.count))
