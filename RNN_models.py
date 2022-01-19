import tensorflow as tf
import numpy as np
from tensorflow.python.training.tracking.data_structures import NoDependency
from tensorflow.python.framework.tensor_shape import TensorShape
from misc import get_def_path
import os

class Rcell(tf.keras.layers.Layer):
    def __init__(self,
                state_size = NoDependency([2, TensorShape([2,2])]),
                params=None,
                dt=None):

        self.state_size = state_size
        super(Rcell, self).__init__()

        self.eta, self.gamma, self.Lambda, self.omega, self.n = params
        self.dt = dt

        self.A_diag = np.diag([-.5*self.gamma]*2).astype(np.float32)
        #A = np.array([[-.5*gamma, omega], [-omega, -0.5*gamma]])
        self.D = np.diag([(self.gamma*(self.n+0.5)) + self.Lambda]*2).astype(np.float32)
        self.C = np.diag([np.sqrt(4*self.eta*self.Lambda)]*2).astype(np.float32)
        self.symplectic = np.array([[0.,1.],[-1.,0.]]).astype(np.float32)
        self.w=1

    def build(self, input_shape):
        self.coeffs_A = self.add_weight(shape=(self.w, self.w),
                                      initializer='uniform',
                                      name='kernel')
        self.built = True

    def call(self, inputs, states):
        dy = inputs
        sts, cov = states

        A = self.coeffs_A
        if self.w == 1:
            A = A*self.symplectic + self.A_diag

        output = tf.einsum('ij,bj->bi',self.C, sts)*self.dt

        xicov = tf.einsum('bij,jk->bik',cov,tf.transpose(self.C)) + tf.transpose(self.D)
        A_minus_xiC = A - tf.einsum('bij,jk->bik',xicov,self.C)

        dx = tf.einsum('bij,bj->bi',A_minus_xiC, sts)*self.dt + tf.einsum('bij,bj->bi', xicov, dy)
        x = sts + dx

        cov_dt = tf.einsum('ij,bjk->bik',A,cov) + tf.einsum('bij,jk->bik',cov, tf.transpose(A)) + self.D - tf.einsum('bij,bjk->bik',xicov, tf.transpose(xicov, perm=[0,2,1]))
        new_cov = cov + cov_dt*self.dt

        new_states = [x, new_cov]
        return output, [new_states]



class GRNNmodel(tf.keras.Model):
    """
    This is the Machine Learning model, where one defines the layers.
    In our case we have a single layer composed of a single (recurrent) unit, which is the GaussianDynamics_RecurrentCell one.
    """

    def __init__(self,
                params,
                dt,
                total_time,
                traj_details,
                x0=tf.convert_to_tensor(np.array([[1,0]]).astype(np.float32)),
                cov_in=tf.eye(2),
                stateful=False):

        super(GRNNmodel,self).__init__()
        self.total_time = total_time

        self.x0 = x0
        self.cov_in = cov_in

        self.total_loss = Metrica(name="total_loss")
        self.coeffsA = Metrica(name="Coeffs_A")
        self.gradient_history = Metrica(name="grads")
        self.recurrent_layer = tf.keras.layers.RNN([Rcell( dt=dt, params=params  )], return_sequences=True, stateful=stateful)

        periods, ppp, train_id, itraj = traj_details
        self.stateful = stateful

        path = get_def_path() + "{}periods/{}ppp/{}/".format(periods,ppp, itraj)
        self.train_path = path+"training/train_id_{}/".format(train_id)
        os.makedirs(self.train_path, exist_ok=True)



    @property
    def initial_state(self):
        """
        shape: (batch, time_step, features)
        """
        x0 = self.x0
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
            diff = tf.squeeze(preds - dys)
            loss = tf.reduce_sum(tf.einsum('bj,bj->b',diff,diff))/(2*self.total_time) #this 2 comes from
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
