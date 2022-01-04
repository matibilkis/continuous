import numpy as np
from tqdm import tqdm
from misc import *
import tensorflow as tf
from RNN_models import *
import argparse
import os
tf.random.set_seed(1)

defpath = get_def_path()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--path", type=str, default=defpath)#"/data/uab-giq/scratch/matias/quantera/trajectories/")#'../sanity/data/'
parser.add_argument("--itraj", type=int, default=0)
parser.add_argument("--periods", default=40)
parser.add_argument("--ppp", type=int,default=500)
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--trainid", type=int, default=0)


args = parser.parse_args()
path, itraj, epochs, periods, train_id, ppp = args.path, int(float(args.itraj)), int(float(args.epochs)), int(float(args.periods)), args.trainid, args.ppp

optimizers = {0:tf.keras.optimizers.Adam, 1:tf.keras.optimizers.SGD}
lrs = [0.001, 0.01, 0.1, 1]
optimizer = optimizers[train_id%2](lr=lrs[train_id%4])


path = path+"{}periods/{}ppp/".format(periods,ppp)

means, covs, signals, coeffs = load_data(path, itraj=itraj)
tfsignals = tf.convert_to_tensor(signals)[tf.newaxis]
A,dt,C,D = coeffs
total_time = dt*ppp*periods

rmod = GRNNmodel(coeffs = [C.astype(np.float32),D.astype(np.float32),dt, total_time], traj_details=[periods, ppp, itraj, get_def_path()], cov_in=tf.convert_to_tensor(cov_in.astype(np.float32)))

rmod.compile(optimizer=optimizer)
rmod(tfsignals[:,:3,:]) #just initialize model

history = rmod.fit(x=tfsignals, y=tfsignals,
                     epochs =epochs, callbacks = [CustomCallback(),
                                                  tf.keras.callbacks.EarlyStopping(monitor='total_loss',
                                                                                   min_delta=0, patience=100,
                                                                                   verbose=0,
                                                                                   mode='min')])
