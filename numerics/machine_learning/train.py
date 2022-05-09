import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
from tqdm import tqdm
from numerics.utilities.misc import *
import tensorflow as tf
import argparse
from numerics.machine_learning.RNN_models import *
from numerics.integration.matrices import *
from datetime import datetime

st = datetime.now()
defpath = get_def_path()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--itraj", type=int, default=1)
parser.add_argument("--dt",type=float, default=1e-3)
parser.add_argument("--total_time", type=float,default=100.)
parser.add_argument("--trainid", type=int, default=0)
parser.add_argument("--epochs", type=int, default=1000)
args = parser.parse_args()

total_time = args.total_time
path = defpath
itraj = args.itraj
train_id = args.trainid
epochs = args.epochs
dt = args.dt



params, exp_path = def_params()
xi, kappa, omega, eta = params
total_time,dt = total_time*kappa, kappa*dt

Hidden_states, signals = load(itraj=itraj, exp_path=exp_path, total_time=total_time, dt=dt, ext_signal=1)

A, D, E, B = genoni_matrices(*params, type="32")
times = get_time(total_time, dt)

total_time = times[-1]

train_path = get_path_config(total_time=total_time,
                            dt = dt,
                            itraj = itraj,
                            exp_path = exp_path)+ "ML/ID{}/".format(train_id)
os.makedirs(train_path, exist_ok=True)
tf.random.set_seed(train_id)
np.random.seed(train_id)


learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
tfsignals = tf.convert_to_tensor(signals)[tf.newaxis]



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
# with open(rmod.train_path+"training_details.txt", 'w') as f:
#     f.write(str(rmod.optimizer.get_config()))
#     f.write("training time: "+str(datetime.now() - st))
# f.close()
