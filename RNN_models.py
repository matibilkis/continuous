import tensorflow as tf
import numpy as np

class MinimalRNNCell(tf.keras.layers.Layer):

    def __init__(self, units, coeffs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__()
        self.C, self.A, self.D, self.dt = coeffs


    def build(self, input_shape):
        # print(input_shape)
        self.coeffs_A = self.add_weight(shape=(2, 2),
                                      initializer='uniform',
                                      name='kernel')

        # self.recurrent_kernel = self.add_weight(
        #     shape=(self.units, self.units),#
        #     initializer='uniform',
        #     name='recurrent_kernel')

        self.built = True

    def call(self, inputs, states, output_states=False):

        prev_output = states[0]

        batched_xicovs, dy = inputs
        batched_A_minus_xiC = self.coeffs_A - tf.einsum('bij,jk->bik',batched_xicovs,self.C)

        dx = tf.einsum('bij,bj->bi',batched_A_minus_xiC, prev_output)*self.dt + tf.einsum('bij,bj->bi', batched_xicovs, dy)
        x = prev_output + dx
        output = tf.einsum('ij,bj->bi',self.C, prev_output)*self.dt
        if output_states == True:
            return x, [x]
        else:
            return output, [x]


class RecModel(tf.keras.Model):
    def __init__(self, coeffs, batch_size=1):
        super(RecModel,self).__init__()

        self.total_loss = Metrica(name="total_loss")
        self.rec_layer = tf.keras.layers.RNN([MinimalRNNCell(2,  coeffs=coeffs)], return_sequences=True)
        self.C, self.A, self.D, self.dt = coeffs
        self.batch_size = batch_size

    @property
    def initial_state(self):
        return tf.repeat(np.array([[1.,0]]).astype(np.float32), self.batch_size, axis=0)


    def call(self, inputs):
        f = self.rec_layer(inputs, initial_state = self.initial_state)
        return f

    @property
    def metrics(self):
        return [self.total_loss]

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
        return {k.name:k.result() for k in self.metrics}

#
#
#
# class ModelA(tf.keras.Model):
#     def __init__(self, coeffs):
#         """
#         Encoder network
#         """
#         super(ModelA,self).__init__()
#         self.gru_layer = tf.keras.layers.GRU(2, return_sequences=True, stateful=True)
#         self.A_coeffs = tf.keras.layers.Dense(4,kernel_initializer=tf.random_uniform_initializer(),bias_initializer = tf.keras.initializers.Zeros())
#         self.loutput = tf.keras.layers.Dense(4,kernel_initializer=tf.random_uniform_initializer(),bias_initializer = tf.keras.initializers.Zeros())
#         self.total_loss = Metrica(name="total_loss")
#         self.C, self.A, self.D, self.dt = coeffs
#
#     def call(self, inputs):
#         f = self.gru_layer(inputs)   #gru_layer(tfsignals[:,:10,:], initial_state=tf.convert_to_tensor([[1.,0.]]))
#         #f = self.loutput(f)
#         return f
#
#     @property
#     def metrics(self):
#         """
#         this helps monitring training
#         """
#         return [self.total_loss]
#
# #batched_A = tf.stack(tf.split(tf.repeat([A], len(signals), axis=0), batch_size))
#     #@tf.function
#     def train_step(self, data):
#         batched_xicovs, batched_signals = data
#         with tf.GradientTape() as tape:
#             tape.watch(self.trainable_variables)
#             states = self(batched_signals)
#
#             batched_A_minus_xiC = self.A - tf.einsum('btij,jk->btik',batched_xicovs,self.C) ## this should get predicted A
#             dx1 = tf.einsum('ij,btj->bti',batched_A_minus_xiC, states)*self.dt
#             dx2 = dx1 + batched_signals
#
#
#             Cxdt = tf.einsum('ij,btj->bti',self.C, states)*self.dt
#             loss = tf.keras.losses.MeanSquaredError()(Cxdt, batched_signals)
#         grads = tape.gradient(loss, self.trainable_variables)
#         self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
#         self.total_loss.update_state(loss)
#         return {k.name:k.result() for k in self.metrics}
#
#
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
