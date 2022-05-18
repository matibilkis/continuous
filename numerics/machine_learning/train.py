import os
import sys
sys.path.insert(0, os.getcwd())

from numerics.utilities.misc import *
from numerics.integration.matrices import *
from numerics.machine_learning.misc import *
from numerics.machine_learning.models import *


import tensorflow as tf
from datetime import datetime
import argparse

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


#
# params, exp_path = def_params()
# total_time = 100.
# dt = 1e-4
# states_si, dys_si = load(itraj=1, exp_path=exp_path, total_time=total_time, dt=dt, ext_signal=1)
# #tfsignals = tf.convert_to_tensor(dys_si.astype(np.float32)[tf.newaxis])
# times = get_time(total_time,dt).astype(np.float32)
# dd = tf.unstack(dys_si.astype(np.float32),axis=1)
#
# tfsignals = tf.stack([times[:-1],dd[0], dd[1]])
# tfsignals = tf.transpose(tfsignals)[tf.newaxis]
#
# tfsignals = tfsignals[:,::1000,:]

st = datetime.now()
path = get_def_path()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--itraj", type=int, default=1)
parser.add_argument("--total_time", type=float,default=100.)
parser.add_argument("--dt",type=float, default=1e-3)
parser.add_argument("--trainid", type=int, default=0)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--learning_rate",type=float, default=1e-3)

args = parser.parse_args()


total_time = args.total_time
itraj = args.itraj
train_id = args.trainid
epochs = args.epochs
dt = args.dt
batch_size = args.batch_size
learning_rate = args.learning_rate

tf.random.set_seed(train_id)
np.random.seed(train_id)


params, exp_path = def_params()
xi, kappa, omega, eta = params

total_time,dt = total_time*kappa, kappa*dt

states,signals = load(itraj=1, exp_path=exp_path, total_time=total_time, dt=dt, ext_signal=1)
tfsignals = pre_process_data_for_ML(total_time, dt, signals)


save_dir = get_training_save_dir(exp_path, total_time, dt, itraj,train_id)
os.makedirs(save_dir, exist_ok=True)





### initialize parameters
f = 2*np.pi
initial_parameters = np.array([1e4, np.pi]).astype(np.float32)
true_parameters = np.array([1e4, 2*np.pi]).astype(np.float32)

A, D , E, B  = genoni_matrices(*params)
xicov, covss = genoni_xi_cov(A,D, E, B ,params, stat=True)



with open(save_dir+"training_details.txt", 'w') as f:
    f.write("BS: {}\nepochs: {}\n learning_rate: {}\n".format(batch_size, epochs, learning_rate))
f.close()


BS = 1
batch_shape = [BS, None, 3]
model = Model(params=params, dt=dt, initial_parameters=initial_parameters,
              true_parameters=true_parameters, initial_states = np.zeros((1,5)).astype(np.float32),
              cov_in=covss, batch_size=tuple([None,None,3]),
              save_dir = save_dir)
model.recurrent_layer.build(tf.TensorShape(batch_shape))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

history = model.craft_fit(tfsignals, batch_size=batch_size, epochs=epochs, early_stopping=1e-8)


with open(save_dir+"training_details.txt", 'w') as f:
    f.write("BS: {}\nepochs: {}\n learning_rate: {}".format(batch_size, epochs, learning_rate))
f.close()


with open(save_dir+"training_details.txt", 'w') as f:
    f.write("training time: "+str(datetime.now() - st))
f.close()


#
# rmod = GRNNmodel(params=params,
#                 dt=(2*np.pi)/(ppp*omega),
#                 total_time=times[-1],
#                 cov_in=tf.convert_to_tensor(covs[0].astype(np.float32)),
#                 train_path=train_path
#                 )
#
# rmod.compile(optimizer=optimizer)
# rmod(tfsignals[:,:3,:]) #just initialize model
# #
# with open(rmod.train_path+"training_details.txt", 'w') as f:
#     f.write(str(rmod.optimizer.get_config()))
# f.close()
#
# history = rmod.fit(x=tfsignals, y=tfsignals,
#                      epochs=epochs, callbacks = [CustomCallback(),
#                                                   tf.keras.callbacks.EarlyStopping(monitor='total_loss',
#                                                                                    min_delta=0, patience=200,
#                                                                                    verbose=0,
#                                                                                    mode='min')])
