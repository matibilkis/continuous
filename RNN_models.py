import tensorflow as tf
import numpy as np

class RecurentCellState(tf.keras.layers.Layer):
    def __init__(self,state_size, coeffs=None):
        self.state_size = state_size
        super(RecurentCellState, self).__init__()
        self.C, self.A, self.D, self.dt = coeffs

    def build(self, input_shape):
        self.coeffs_A = self.add_weight(shape=(2, 2),
                                      initializer='uniform',
                                      name='kernel')

        self.built = True

    def call(self, inputs, states, output_states=False):
        prev_output = states[0]
        cov = states[1]

        xicov = tf.einsum('ij,jk->ik',cov,ct(self.C))
        bathed_xicovs = tf.expand_dims(xicov, axis=0)
        dy = inputs

        batched_A_minus_xiC = self.coeffs_A - tf.einsum('bij,jk->bik',batched_xicovs,self.C)

        dx = tf.einsum('bij,bj->bi',batched_A_minus_xiC, prev_output)*self.dt + tf.einsum('bij,bj->bi', batched_xicovs, dy)
        x = prev_output + dx
        output = tf.einsum('ij,bj->bi',self.C, prev_output)*self.dt

        new_cov = tf.einsum('ij,jk->ik',cov, self.coeffs_A) + tf.einsum('ij,jk->ik',self.coeffs_A,cov) - tf.einsum('ij,jk->ik',self.coeffs_A,ct())
        if output_states == True:
            return x, [x, self.covariance]
        else:
            return output, [x, self.covariance]


class GaussianRecuModel(tf.keras.Model):
    """
    This is the Machine Learning model, where one defines the layers.
    In our case we have a single layer composed of a single (recurrent) unit, which is the GaussianDynamics_RecurrentCell one.
    """

    def __init__(self, coeffs, batch_size=1):
        super(GaussianRecuModel,self).__init__()
        self.C, self.A, self.D, self.dt = coeffs

        self.total_loss = Metrica(name="total_loss")
        self.coeffsA = Metrica(name="Coeffs_A")
        self.recurrent_layer_state = tf.keras.layers.RNN([RecurentCellState(2,  coeffs=coeffs)], return_sequences=True)
        self.batch_size = batch_size

    @property
    def initial_state(self):
        """
        shape: (batch, time_step, features)
        """
        return tf.convert_to_tensor(np.array([1.,0.]).astype(np.float32))[tf.newaxis, :]

    @property
    def metrics(self):
        return [self.total_loss, self.coeffsA]

    def call(self, inputs):
        return self.recurrent_layer(inputs, initial_state = self.initial_state)

    @tf.function
    def train_step(self, data):
        inputs, batched_signals = data
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            preds = self(inputs)
            loss = tf.keras.losses.MeanSquaredError()(preds, batched_signals)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss.update_state(loss)
        self.coeffsA.update_state(self.trainable_variables[0])
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
