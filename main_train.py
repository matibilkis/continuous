import numpy as np
from tqdm import tqdm
from misc import *
import tensorflow as tf
from RNN_models import *
import argparse
import os
defpath = get_def_path()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--path", type=str, default=defpath)#"/data/uab-giq/scratch/matias/quantera/trajectories/")#'../sanity/data/'
parser.add_argument("--itraj", type=int, default=0)
parser.add_argument("--periods", default=20)
parser.add_argument("--ppp", type=int,default=1000)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--trainid", type=int, default=0)


args = parser.parse_args()
path, itraj, epochs, periods, train_id, ppp = args.path, int(float(args.itraj)), int(float(args.epochs)), int(float(args.periods)), args.trainid, args.ppp

path = path+"{}periods/{}ppp/".format(periods,ppp)

means, covs, signals, coeffs = load_data(path, itraj=itraj)
tfsignals = tf.convert_to_tensor(signals)[tf.newaxis]
A,dt,C,D = coeffs
total_time = periods ##asumming freq = 1

lrs = {ind:k for ind,k in enumerate(np.logspace(-4,1,10))}

rmodel = GRNNmodel(coeffs = [C,D,dt, total_time], traj_details=[periods, ppp, train_id, path], cov_in=tf.convert_to_tensor(covs[0].astype(np.float32)), stateful=False)
rmodel.compile(optimizer=tf.keras.optimizers.SGD(lr=lrs[train_id]))
rmodel.recurrent_layer(tfsignals[:,:10,:], initial_state=rmodel.initial_state)


history = rmodel.fit(x=tfsignals, y=tfsignals,
                     epochs = 10**3, callbacks = [CustomCallback(),
                                                  tf.keras.callbacks.EarlyStopping(monitor='total_loss',
                                                                                   min_delta=0, patience=100,
                                                                                   verbose=0,
                                                                                   mode='min')])
