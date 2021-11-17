
import tensorflow as tf
import numpy as np

tf.keras.backend.set_floatx("float32")
class ModelA(tf.keras.Model):
    def __init__(self, coeffs):
        """
        Encoder network
        """
        super(ModelA,self).__init__()
        self.gru_layer = tf.keras.layers.GRU(2, return_sequences=True, stateful=True)
        self.A_coeffs = tf.keras.layers.Dense(4,kernel_initializer=tf.random_uniform_initializer(),bias_initializer = tf.keras.initializers.Zeros())
        self.loutput = tf.keras.layers.Dense(4,kernel_initializer=tf.random_uniform_initializer(),bias_initializer = tf.keras.initializers.Zeros())
        self.total_loss = Metrica(name="total_loss")
        self.C, self.A, self.D, self.dt = coeffs

    def call(self, inputs):
        f = self.gru_layer(inputs)   #gru_layer(tfsignals[:,:10,:], initial_state=tf.convert_to_tensor([[1.,0.]]))
        #f = self.loutput(f)
        return f

    @property
    def metrics(self):
        """
        this helps monitring training
        """
        return [self.total_loss]

    #@tf.function
    def train_step(self, data):
        x, batched_signals = data
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            states = self(batched_signals)
            Cxdt = tf.einsum('ij,btj->bti',self.C, states)*self.dt
            loss = tf.keras.losses.MeanSquaredError()(Cxdt, batched_signals)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss.update_state(loss)
        return {k.name:k.result() for k in self.metrics}


class Metrica(tf.keras.metrics.Metric):
    """
    This helps to monitor training (for instance each loss)
    """
    def __init__(self, name):
        super(Metrica, self).__init__()
        self._name=name
        self.metric_variable = self.add_weight(name=name, initializer='zeros')

    def update_state(self, new_value, sample_weight=None):
        self.metric_variable.assign(new_value)

    def result(self):
        return self.metric_variable

    def reset_states(self):
        self.metric_variable.assign(0.)
