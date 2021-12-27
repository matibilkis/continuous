import tensorflow as tf
import numpy as np
from tensorflow.python.training.tracking.data_structures import NoDependency
from tensorflow.python.framework.tensor_shape import TensorShape
from misc import get_def_path
import os

class Rcell(tf.keras.layers.Layer):
    def __init__(self,state_size = NoDependency([2, TensorShape([2,2])]), coeffs=None):
        self.state_size = state_size   ## Favorite: state_size = NoDependency([2, TensorShape([2,2])])
        super(Rcell, self).__init__()
        self.C, self.D, self.dt = coeffs
        self.max_update = 1e-1

    def build(self, input_shape):
        self.coeffs_A = self.add_weight(shape=(2, 2),
                                      initializer='uniform',
                                      name='kernel')
        self.built = True

    def call(self, inputs, states):
        dy = inputs
        sts, cov = states

        output = tf.einsum('ij,bj->bi',self.C, sts)*self.dt

        xicov = tf.einsum('bij,jk->bik',cov,tf.transpose(self.C)) + tf.transpose(self.D)
        A_minus_xiC = self.coeffs_A - tf.einsum('bij,jk->bik',xicov,self.C)

        dx = tf.einsum('bij,bj->bi',A_minus_xiC, sts)*self.dt + tf.einsum('bij,bj->bi', xicov, dy)
        x = sts + tf.clip_by_value(dx,-self.max_update,self.max_update)

        cov_dt = tf.einsum('ij,bjk->bik',self.coeffs_A,cov) + tf.einsum('bij,jk->bik',cov, tf.transpose(self.coeffs_A)) + self.D - tf.einsum('bij,bjk->bik',xicov, tf.transpose(xicov, perm=[0,2,1]))
        new_cov = cov + cov_dt*self.dt

        new_states = [x, tf.clip_by_value(new_cov, -1,1)]
        return output, [new_states]



class GRNNmodel(tf.keras.Model):
    """
    This is the Machine Learning model, where one defines the layers.
    In our case we have a single layer composed of a single (recurrent) unit, which is the GaussianDynamics_RecurrentCell one.
    """

    def __init__(self, coeffs,traj_details, cov_in=tf.eye(2), stateful=False):
        super(GRNNmodel,self).__init__()
        self.C, self.D, self.dt, self.total_time = coeffs
        self.cov_in = cov_in

        self.total_loss = Metrica(name="total_loss")
        self.coeffsA = Metrica(name="Coeffs_A")
        self.gradient_history = Metrica(name="grads")
        self.recurrent_layer = tf.keras.layers.RNN([Rcell(coeffs=coeffs[:-1])], return_sequences=True, stateful=stateful)

        periods, ppp, train_id, path = traj_details
        if path == "":
            path = get_def_path() + "{}periods/{}ppp/".format(periods,ppp)
        self.train_path = path+"training/train_id_{}/".format(train_id)
        os.makedirs(self.train_path, exist_ok=True)



    @property
    def initial_state(self):
        """
        shape: (batch, time_step, features)
        """
        x0 = tf.convert_to_tensor(np.array([[1,0]]).astype(np.float32))
        Sig0 = self.cov_in
        return [[x0 , Sig0[tf.newaxis]]]

    @property
    def metrics(self):
        return [self.total_loss, self.coeffsA, self.gradient_history]

    def call(self, inputs):
        return self.recurrent_layer(inputs, initial_state = self.initial_state)

    @tf.function
    def train_step(self, data):
        inputs, dys = data
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            preds = self(inputs)
            loss = tf.keras.losses.MeanSquaredError()(dys,preds)*(dys.shape[1]/self.total_time)   #dys.shape[1] is number of data_points  In this way loss should go to 1
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


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        histories = self.model.history.history
        keys_histories = list(histories.keys())
        for k,v, in histories.items():
            np.save(self.model.train_path+"{}".format(k), v, allow_pickle=True)
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))
