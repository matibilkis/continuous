import numpy as np
from tqdm import tqdm
from misc import *
import tensorflow as tf
import argparse
from datetime import datetime
import os
tf.random.set_seed(1)

defpath = get_def_path()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--path", type=str, default=defpath)#"/data/uab-giq/scratch/matias/quantera/trajectories/")#'../sanity/data/'
parser.add_argument("--itraj", type=int, default=0)
parser.add_argument("--periods",type=int, default=40)
parser.add_argument("--ppp", type=int,default=500)
parser.add_argument("--trainid", type=int, default=0)
parser.add_argument("--oneparam", type=int, default=0) #0 is one param 1 is whole matrix


args = parser.parse_args()
path = args.path
itraj = args.itraj
periods = args.periods
ppp = args.ppp
train_id = args.trainid
oneparam = args.oneparam

optimizers = {0:tf.keras.optimizers.Adam, 1:tf.keras.optimizers.SGD}
lrs = [0.01, 0.1, 1, 10]
learning_rate = lrs[train_id%2]
optimizer = optimizers[train_id%2](lr=learning_rate)

st = datetime.now()

path = path+"{}periods/{}ppp/".format(periods,ppp)

means, covs, signals, coeffs = load_data(path, itraj=itraj)
tfsignals = tf.convert_to_tensor(signals)[tf.newaxis]
A,dt,C,D = coeffs
total_time = dt*ppp*periods

print(oneparam)
path+=["oneparam/","matrix/"][oneparam]
if oneparam == 0:
    from RNN_models_one_param import *
    rmod = GRNNmodel(coeffs = [C.astype(np.float32),D.astype(np.float32),dt, total_time], traj_details=[periods, ppp, train_id, path], cov_in=tf.convert_to_tensor(covs[0].astype(np.float32)))
else:
    
    from RNN_models import *
    rmod = GRNNmodel(coeffs = [C.astype(np.float32),D.astype(np.float32),dt, total_time], traj_details=[periods, ppp, train_id, path], cov_in=tf.convert_to_tensor(covs[0].astype(np.float32)))
    
rmod.compile(optimizer=optimizer)
rmod(tfsignals[:,:3,:]) #just initialize model

history = rmod.fit(x=tfsignals, y=tfsignals,
                     epochs=2000, callbacks = [CustomCallback(),
                                                  tf.keras.callbacks.EarlyStopping(monitor='total_loss',
                                                                                   min_delta=0, patience=100,
                                                                                   verbose=0,
                                                                                   mode='min')])
print("finished at "+str(datetime.now() - st))