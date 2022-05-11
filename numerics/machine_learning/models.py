import tensorflow as tf
import numpy as np
from numerics.utilities.misc import *
from numerics.integration.matrices import *
import os
from numerics.machine_learning.cell import Rcell


class GRNNmodel(tf.keras.Model):

    def __init__(self,
                params,
                dt,
                total_time,
                train_path = "",
                cov_in=np.eye(2),
                stateful=False):

        super(GRNNmodel,self).__init__()
        self.total_time = total_time
        self.params = params

        self.x0 = tf.convert_to_tensor(np.array([[1.,0.]]).astype(np.float32))
        self.cov_in = tf.convert_to_tensor(cov_in.astype(np.float32))
        self.dt = dt

        self.total_loss = Metrica(name="total_loss")
        self.target_params = Metrica(name="target_params")
        self.gradient_history = Metrica(name="grads")
        self.recurrent_layer = tf.keras.layers.RNN([Rcell( dt=dt, params=params  )], return_sequences=True, stateful=stateful)

        self.stateful = stateful
        self.train_path = train_path


    @property
    def initial_state(self):
        """
        shape: (batch, time_step, features)
        """
        return [[self.x0 , self.cov_in[tf.newaxis]]]

    @property
    def metrics(self):
        return [self.total_loss, self.target_params, self.gradient_history]

    def call(self, inputs):
        return self.recurrent_layer(inputs, initial_state = self.initial_state)

    @tf.function
    def train_step(self, data):
        inputs, dys = data

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            preds = self(inputs)
            diff = tf.squeeze(preds - dys)
            loss = tf.reduce_sum(tf.einsum('bj,bj->b',diff,diff))# - self.total_time)/(2*self.C_coeff*(self.dt**(3/2)))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss.update_state(loss)
        self.coeffsA.update_state(self.trainable_variables[0])
        self.gradient_history.update_state(grads)

        return {k.name:k.result() for k in self.metrics}


class Metrica(tf.keras.metrics.Metric):
    """
    This helps to monitor training (for instance one out of different losses),
    but you can also monitor gradients magnitude for example.
    """
    def __init__(self, name):
        super(Metrica, self).__init__()
        self._name=name
        self.metric_variable = tf.convert_to_tensor(np.zeros((2,2)).astype(np.float32))

    def update_state(self, new_value):
        self.metric_variable = new_value

    def result(self):
        return self.metric_variable

    def reset_states(self):
        self.metric_variable = tf.convert_to_tensor(np.zeros((2,2)).astype(np.float32))


# class CustomCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         keys = list(logs.keys())
#         histories = self.model.history.history
#         keys_histories = list(histories.keys())
#         for k,v, in histories.items():
#             np.save(self.model.train_path+"{}".format(k), v, allow_pickle=True)
#         print("End epoch {} of training; got log keys: {}".format(epoch, keys))
