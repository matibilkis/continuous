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

    def build(self, input_shape):
        self.coeffs_A = self.add_weight(shape=(2, 2),
                                      initializer='uniform',
                                      name='kernel')
        self.built = True

    def call(self, inputs, states):
        dy = inputs
        sts, cov = states

        A = self.coeffs_A

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

    def __init__(self, coeffs,traj_details,x0=tf.convert_to_tensor(np.array([[1,0]]).astype(np.float32)), cov_in=tf.eye(2), stateful=False):
        super(GRNNmodel,self).__init__()
        self.C, self.D, self.dt, self.total_time = coeffs

        self.x0 = x0
        self.cov_in = cov_in

        self.total_loss = Metrica(name="total_loss")
        self.coeffsA = Metrica(name="Coeffs_A")
        self.gradient_history = Metrica(name="grads")
        self.recurrent_layer = tf.keras.layers.RNN([Rcell(coeffs=[self.C, self.D, self.dt])], return_sequences=True, stateful=stateful)

        periods, ppp, train_id, itraj = traj_details
        self.stateful = stateful

        path = get_def_path() + "{}periods/{}ppp/{}itraj/".format(periods,ppp, itraj)
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

    

    

    
