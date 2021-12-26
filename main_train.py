import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
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
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--trainid", type=int, default=0)


args = parser.parse_args()
path, itraj, epochs, periods, train_id = args.path, int(float(args.itraj)), int(float(args.epochs)), int(float(args.periods)), args.trainid

path = path+"{}periods/{}ppp/".format(periods,ppp)
train_path = path+"/training/train_id_{}/".format(train_id)
os.makedirs(train_path, exist_ok=True)


means, covs, signals, coeffs = load_data(path, itraj=itraj)
tfsignals = tf.convert_to_tensor(signals)[tf.newaxis]
A,dt,C,D = coeffs
total_time = 2*np.pi*periods ##asumming freq = 1


rmodel = GRNNmodel([C,dt, total_time], cov_in=tf.convert_to_tensor(covs[0].astype(np.float32)))
rmodel.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01))
rmodel.recurrent_layer(tfsignals, initial_state=rmodel.initial_state)
#rmodel.trainable_variables[0].assign(tf.convert_to_tensor(A.astype(np.float32)))

history = rmodel.fit(x=tfsignals, y=tfsignals,
                     epochs = 10**3, callbacks = [tf.keras.callbacks.EarlyStopping(monitor='total_loss',
                                                                                   min_delta=0, patience=500,
                                                                                   verbose=0,
                                                                                   mode='min')])

histories = rmodel.history.history
keys_histories = list(histories.keys())
for k,v, in histories.items():
    np.save(train_path+"{}".format(k), v, allow_pickle=True)
