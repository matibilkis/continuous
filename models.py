
import tensorflow as tf
import numpy as np

tf.keras.backend.set_floatx("float32")
class ModelA(tf.keras.Model):
    def __init__(self, quantities):
        """
        Encoder network
        """
        super(ModelA,self).__init__()
        self.l1 = tf.keras.layers.Dense(32,kernel_initializer=tf.random_uniform_initializer(),bias_initializer = tf.keras.initializers.Zeros())
        self.l2 = tf.keras.layers.Dense(32,kernel_initializer=tf.random_uniform_initializer(),bias_initializer = tf.keras.initializers.Zeros())
        self.loutput = tf.keras.layers.Dense(4, bias_initializer = tf.keras.initializers.Zeros())
        self.total_loss = Metrica(name="total_loss")

        target_A, xicovs, means, signals, C, dt = quantities

        self.target_A = tf.cast(tf.convert_to_tensor(target_A), np.float32)
        self.tfxicovs = tf.cast(tf.convert_to_tensor(xicovs), np.float32)
        self.tfmeans = tf.cast(tf.convert_to_tensor(means), np.float32)
        self.tfsignals = tf.cast(tf.convert_to_tensor(signals), np.float32)

        self.C = C
        self.dt = dt


    def call(self, inputs):
        f = tf.nn.sigmoid(self.l1(inputs))
        f = tf.nn.sigmoid(self.l2(f))
        f = self.loutput(f)
        return f

    def spit_A(self):
        return tf.reshape(self(tf.ones((1,1))),(2,2))

    @property
    def metrics(self):
        """
        this helps monitring training
        """
        return [self.total_loss]

    def train_step(self, data):
        x, tfsignals = data

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            A_pred = self.spit_A()
            dx0 = A_pred - tf.einsum('tij,jl->til',self.tfxicovs[:-1], self.C)
            dx1 = tf.einsum('tij,tj->ti', dx0, self.tfmeans[:-1])*self.dt
            dx2 = dx1 + tf.einsum('tik,tk->ti', self.tfxicovs[:-1], self.tfsignals)
            x = dx2 + self.tfmeans[:-1]    #this gives tfmeansx[1:]
            lt1 = tf.einsum('ij,tj->ti',self.C,x)*self.dt
            loss = tf.keras.losses.MeanSquaredError()(lt1[:-1], self.tfsignals[1:])

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
