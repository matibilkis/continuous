import tensorflow as tf
import numpy as np
from tensorflow.python.training.tracking.data_structures import NoDependency
from tensorflow.python.framework.tensor_shape import TensorShape
from numerics.integration.matrices import *


#The input of the recurrent_layer should be of the shape   shape = (batch_size,   time_steps,   features)
class RCell(tf.keras.layers.Layer):
    def __init__(self,
                units=5,### units = 5 [x,p, Vx, Vp, Cov(x,y)]
                params=[],
                dt= 1e-4,
                target_params=np.zeros(2).type(np.float32),
                **kwargs):

        self.units = units
        self.state_size = units   ### this means that the internal state is a "units"-dimensional vector

        self.xi, self.kappa, self.omega, self.eta = params
        self.dt = dt

        self.A_matrix, self.D_matrix, self.E_matrix, self.B_matrix = genoni_matrices(*params, type="32")
        self.XiCov = genoni_xi_cov(self.A_matrix,self.D_matrix,self.E_matrix,self.B_matrix,params).astype("float32")[tf.newaxis]
        self.cov_in = tf.convert_to_tensor(cov_in.astype(np.float32))[tf.newaxis]
        self.XiCovC = np.dot(self.XiCov,-np.sqrt(2)*self.B_matrix.T)


        super(Rcell, self).__init__()


    def call(self, inputs, states):
        inns = tf.squeeze(inputs)
        time, dy = inns[0], inns[1:][tf.newaxis]

        sts = states
        cov = self.cov_in
        #sts, cov = states

        XiCov = (self.E_matrix - tf.einsum('bij,jk->bik',cov,self.B_matrix))/np.sqrt(2)
        XiCovC = tf.matmul(XiCov,-np.sqrt(2)*self.B_matrix.T)

        output = tf.einsum('ij,bj->bi',-np.sqrt(2)*self.B_matrix.T, sts)*self.dt
        dx = tf.einsum('bij,bj->bi',self.A_matrix - XiCovC, sts)*self.dt + tf.einsum('bij,bj->bi', XiCov, dy) + 10*tf.cos(0.2*np.pi*time)*self.dt*self.x_signal ##  + params...
        x = sts + dx

        cov_dt = tf.einsum('ij,bjk->bik',self.A_matrix,cov) + tf.einsum('bij,jk->bik',cov, tf.transpose(self.A_matrix)) + self.D_matrix - 2*tf.einsum('bij,bjk->bik',XiCov, tf.transpose(XiCov, perm=[0,2,1]))
        new_cov = cov + cov_dt*self.dt

        new_states = [x, cov]
        return output, [new_states]


    def build(self):
        self.target_params = self.add_weight(shape=(1, 2),
                                      initializer='uniform',
                                      name='kernel')
        self.target_params[0].assign( self.target_params_true)
        self.built = True

        self.built = True

    def get_initial_state(self,inputs=None, batch_size=1, dtype=np.float32):
        return tf.random.uniform( tuple([batch_size]) + tuple([self.state_size]), dtype=dtype)




















# class Rcell(tf.keras.layers.Layer):
#     def __init__(self,
#                 state_size = NoDependency([2, TensorShape([2,2])]),
#                 params=None,
#                 true_target_params=[],
#                 dt=None):
#
#         self.state_size = state_size
#         super(Rcell, self).__init__()
#
#         self.xi, self.kappa, self.omega, self.eta = params
#         self.dt = dt
#
#         self.A_matrix, self.D_matrix, self.E_matrix, self.B_matrix = genoni_matrices(*params, type="32")
#         self.XiCov = genoni_xi_cov(self.A_matrix,self.D_matrix,self.E_matrix,self.B_matrix,params).astype("float32")
#         self.XiCovC = np.dot(self.XiCov,-np.sqrt(2)*self.B_matrix.T)
#         self.w=1### 2 otherwise
#
#         self.true_target_params = np.array(true_target_params).astype(np.float32)
#
#
#     def build(self, input_shape):
#         self.target_params = self.add_weight(shape=(self.w, self.w),
#                                       initializer='uniform',
#                                       name='kernel')
#         omega = self.omega
#         self.target_params[0].assign( self.true_target_params/10.)
#         self.built = True
#
#     def call(self, inputs, states):
#         dy = inputs
#         sts, cov = states
#
#         output = tf.einsum('ij,bj->bi',-np.sqrt(2)*self.B_matrix.T, sts)*self.dt
#
#         dx = tf.einsum('bij,bj->bi',self.A_matrix - self.XicovC, sts)*self.dt + tf.einsum('bij,bj->bi', self.XiCov, dy)  ##  + params...
#         x = sts + dx
#
#         RicattiCov = (self.E_matrix - tf.einsum('bij,jk->bik',cov,self.B_matrix)) #without the sqrt(2)
#         cov_dt = tf.einsum('ij,bjk->bik',self.A_matrix,cov) + tf.einsum('bij,jk->bik',cov, tf.transpose(self.A_matrix)) + self.D_matrix - tf.einsum('bij,bjk->bik',RicattiCov, tf.transpose(RicattiCov, perm=[0,2,1]))
#         new_cov = cov + cov_dt*self.dt
#
#         new_states = [x, new_cov]
#         return output, [new_states]
