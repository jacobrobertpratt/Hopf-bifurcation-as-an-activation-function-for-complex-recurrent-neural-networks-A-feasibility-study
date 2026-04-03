

import tensorflow.compat.v2 as tf

from keras.optimizers.optimizer_v2 import optimizer_v2


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

class MyOptimizer(optimizer_v2.OptimizerV2):
    
    _HAS_AGGREGATE_GRAD = False

    def __init__(
        self,
        learning_rate=0.001,
        momentum=0.0,
        nesterov=False,
        name="MyOptimizer",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)

        self._momentum = False
        if (
            isinstance(momentum, tf.Tensor)
            or callable(momentum)
            or momentum > 0
        ):
            self._momentum = True
        if isinstance(momentum, (int, float)) and (
            momentum < 0 or momentum > 1
        ):
            raise ValueError(
                f"`momentum` must be between [0, 1]. Received: "
                f"momentum={momentum} (of type {type(momentum)})."
            )
        self._set_hyper("momentum", momentum)

        self.nesterov = nesterov

    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                print('var:',var)
                self.add_slot(var, "momentum")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["momentum"] = tf.identity(
            self._get_hyper("momentum", var_dtype)
        )
        
    def _resource_apply_dense(self, grad, var, apply_state=None):

        _IS_UNITARY = False
        _IS_HERMITIAN = False
        if 'unit' in var.name:
            _IS_UNITARY = True
#            tf.print( 'GOT HERE -> UNIT' )
        elif 'herm' in var.name:
            _IS_HERMITIAN = True
#            tf.print( 'GOT HERE -> HERM' )
        else:
#            tf.print( 'GOT HERE -> PASS' )
            pass
        
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
        
        if _IS_UNITARY:
            
#            tf.print( 'Got here ... ' )
#            tf.print( 'grad:\n', tf.math.real( grad ) , tf.math.imag( grad ) , '\n' )
            
            lr = coefficients["lr_t"] * (-1.+0.j)
#            tf.print( 'lr:', tf.math.real( lr ) , tf.math.imag( lr ) )
            
            # Map the gradient to a Unitary Matrix #
            unitary_grad = tf.linalg.expm( lr * grad ) # This created a unitary matrix changed by e^(-lr*Tgrad)
#            tf.print( 'unitary_grad:\n', tf.math.real( unitary_grad ) , tf.math.imag( unitary_grad ) , '\n' )
            
            # Create the new Unitary matrix #
            new_var = tf.linalg.matmul( var , unitary_grad )
#            tf.print( 'new_var:\n', tf.math.real( new_var ) , tf.math.imag( new_var ) , '\n' )
#            ichk = tf.linalg.matmul( new_var , new_var , adjoint_a = True )
#            tf.print( 'ichk:\n', tf.math.real( ichk ) , tf.math.imag( ichk ) , '\n' )
            
            return tf.raw_ops.AssignVariableOp(
                resource=var.handle,
                value=new_var,
                validate_shape=False
            )
            
        elif _IS_HERMITIAN:
            new_var = var - (grad * coefficients["lr_t"])
            return tf.raw_ops.AssignVariableOp(
                resource=var.handle,
                value = new_var,
                validate_shape=False
            )
        else:
            return tf.raw_ops.ResourceApplyGradientDescent(
                var=var.handle,
                alpha=coefficients["lr_t"],
                delta=grad,
                use_locking=self._use_locking
            )

    def _resource_apply_sparse_duplicate_indices(
        self, grad, var, indices, **kwargs
    ):
        if self._momentum:
            return super()._resource_apply_sparse_duplicate_indices(
                grad, var, indices, **kwargs
            )
        else:
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = kwargs.get("apply_state", {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)
            
            return tf.raw_ops.ResourceScatterAdd(
                resource=var.handle,
                indices=indices,
                updates=-grad * coefficients["lr_t"],
            )
    
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # This method is only needed for momentum optimization.
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        momentum_var = self.get_slot(var, "momentum")
        return tf.raw_ops.ResourceSparseApplyKerasMomentum(
            var=var.handle,
            accum=momentum_var.handle,
            lr=coefficients["lr_t"],
            grad=grad,
            indices=indices,
            momentum=coefficients["momentum"],
            use_locking=self._use_locking,
            use_nesterov=self.nesterov,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    "learning_rate"
                ),
                "decay": self._initial_decay,
                "momentum": self._serialize_hyperparameter("momentum"),
                "nesterov": self.nesterov,
            }
        )
        return config
