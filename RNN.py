
import tensorflow as tf
import numpy as np

#tf.keras.backend.set_floatx("float32")

class MinimalRNNCell(tf.keras.layers.Layer):
    ### some helpful reference https://stackoverflow.com/questions/60185290/implementing-a-minimal-lstmcell-in-keras-using-rnn-and-layer-classes
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0] #i guess that states[0] is because you put [output]
        h = tf.keras.backend.dot(inputs, self.kernel)
        output = h + tf.keras.backend.dot(prev_output, self.recurrent_kernel)
        return output, [output]


class RecModel(tf.keras.Model):
    def __init__(self, units):
        self.total_loss = Metrica(name="total_loss")

        self.recurrent_layer = tf.keras.layers.RNN(MinimalRNNCell(units))

    def call(self, inputs):
        f = self.recurrent_layer(inputs)   #gru_layer(tfsignals[:,:10,:], initial_state=tf.convert_to_tensor([[1.,0.]]))
        return f

    @property
    def metrics(self):
        return [self.total_loss]


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

#batched_A = tf.stack(tf.split(tf.repeat([A], len(signals), axis=0), batch_size))
    #@tf.function
    def train_step(self, data):
        batched_xicovs, batched_signals = data
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            states = self(batched_signals)

            batched_A_minus_xiC = self.A - tf.einsum('btij,jk->btik',batched_xicovs,self.C) ## this should get predicted A
            dx1 = tf.einsum('ij,btj->bti',batched_A_minus_xiC, states)*self.dt
            dx2 = dx1 + batched_signals


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
