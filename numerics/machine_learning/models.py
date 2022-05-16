from numerics.utilities.misc import *
from numerics.integration.matrices import *
import tensorflow as tf



#The input of the recurrent_layer should be of the shape   shape = (batch_size,   time_steps,   features)
class GRCell(tf.keras.layers.Layer):
    def __init__(self,
                units=5,### units = 5 [x,p, Vx, Vp, Cov(x,y)]
                params=[],
                dt= 1e-4,
                initial_parameters=np.zeros(2).astype(np.float32),
                true_parameters=np.zeros(2).astype(np.float32),
                cov_in = np.zeros((2,2)).astype(np.float32),
                initial_states = np.array([[20., 20., 0, 0, 0,]]).astype(np.float32),
                 #inn_state=np.zeros(5).astype(np.float32),
                **kwargs):

        self.units = units
        self.state_size = units   ### this means that the internal state is a "units"-dimensional vector

        self.xi, self.kappa, self.omega, self.eta = params
        self.dt = dt
        self.A_matrix, self.D_matrix, self.E_matrix, self.B_matrix = genoni_matrices(*params, type="32")
        self.XiCov = genoni_xi_cov(self.A_matrix,self.D_matrix,self.E_matrix,self.B_matrix,params).astype("float32")[tf.newaxis]
        self.cov_in = tf.convert_to_tensor(cov_in.astype(np.float32))[tf.newaxis]
        self.XiCovC = np.dot(self.XiCov,-np.sqrt(2)*self.B_matrix.T)


        self.initial_parameters = initial_parameters
        self.x_signal = tf.convert_to_tensor(np.array([1.,0.]).astype(np.float32))
        self.true_parameters = tf.convert_to_tensor(true_parameters.astype(np.float32))

        self.initial_states = tf.convert_to_tensor(initial_states)
        super(GRCell, self).__init__(**kwargs)


    def call(self, inputs, states):
        inns = tf.squeeze(inputs)
        time, dy = inns[0], inns[1:][tf.newaxis]

        sts = states[0][:,:2]
        cov = self.cov_in

        XiCov = (self.E_matrix - tf.einsum('bij,jk->bik',cov,self.B_matrix))/np.sqrt(2)
        XiCovC = tf.matmul(XiCov,-np.sqrt(2)*self.B_matrix.T)

        output = tf.einsum('ij,bj->bi',-np.sqrt(2)*self.B_matrix.T, sts)*self.dt
        dx = tf.einsum('bij,bj->bi',self.A_matrix - XiCovC, sts)*self.dt + tf.einsum('bij,bj->bi', XiCov, dy) + self.training_params[0][0]*self.dt#tf.cos(self.true_parameters[1]*time)*self.dt*self.x_signal ##  + params...
        x = sts + dx

        cov_dt = tf.einsum('ij,bjk->bik',self.A_matrix,cov) + tf.einsum('bij,jk->bik',cov, tf.transpose(self.A_matrix)) + self.D_matrix - 2*tf.einsum('bij,bjk->bik',XiCov, tf.transpose(XiCov, perm=[0,2,1]))
        new_cov = cov + cov_dt*self.dt

        new_states = tf.concat([x, tf.zeros((1,3))],axis=-1)
        return output, [new_states]


    def build(self, input_shape):
        self.training_params = self.add_weight(shape=(1, 2),
                                      initializer='uniform',
                                      name='kernel')
        self.training_params[0].assign( self.initial_parameters)
        self.built = True

    def get_initial_state(self,inputs=None, batch_size=1, dtype=np.float32):
        #return tf.zeros( tuple([batch_size]) + tuple([self.state_size]), dtype=dtype)
        return self.initial_states

    def reset_states(self,inputs=None, batch_size=1, dtype=np.float32):
        return self.initial_states
        #return tf.zeros( tuple([batch_size]) + tuple([self.state_size]), dtype=dtype)



class Model(tf.keras.Model):
    #https://stackoverflow.com/questions/57860614/specifying-the-batch-size-when-subclassing-keras-model   ##Thanks :)
    ### workarounds: batch_input_shape=batch_size and building..
    def __init__(self,stateful=True, params=[], dt=1e-4,
                true_parameters=[],
                initial_parameters=[], cov_in=np.zeros((2,2)),
                batch_size=(10), **kwargs):
        super(Model,self).__init__()
        self.recurrent_layer =tf.keras.layers.RNN(GRCell(units=5, params=params, dt=dt, true_parameters=true_parameters, initial_parameters=initial_parameters, cov_in = cov_in),
                                      return_sequences=True, stateful=True,  batch_input_shape=batch_size)
        self.total_loss = Metrica(name="LOSS")
        self.target_params_record = Metrica(name="PARAMS")
        self.gradient_history = Metrica(name="GRADS")
    def call(self, inputs):
        return self.recurrent_layer(inputs)

    def reset_states(self,states=None):
        self.recurrent_layer.states[0].assign(self.recurrent_layer.cell.get_initial_state())
        return self.recurrent_layer.cell.get_initial_state()

    @property
    def metrics(self):
        return [self.total_loss, self.target_params_record, self.gradient_history]

    @tf.function
    def train_step(self, data):
        inputs, times_dys = data
        dys = times_dys[:,:,1:] ###recall first entry is time, then signals
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            preds = self(inputs)
            diff = tf.squeeze(preds - dys)
            loss = tf.reduce_sum(tf.einsum('bj,bj->b',diff,diff))

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss.update_state(loss)
        self.target_params_record.update_state(self.trainable_variables[0])
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
        self.metric_variable = tf.convert_to_tensor(np.zeros((1,2)).astype(np.float32))

    def update_state(self, new_value):
        self.metric_variable = new_value

    def result(self):
        return self.metric_variable

    def reset_states(self):
        self.metric_variable = tf.convert_to_tensor(np.zeros((1,2)).astype(np.float32))
