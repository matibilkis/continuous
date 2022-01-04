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

length_series = [int(k) for k in np.logspace(2,np.log10(len(signals[:,0])), 16)][itraj]


rmodel = GRNNmodel(coeffs = [C,D,dt, total_time], traj_details=[periods, ppp, train_id, path], cov_in=tf.convert_to_tensor(covs[0].astype(np.float32)), stateful=False)
rmodel.compile(optimizer=tf.keras.optimizers.SGD(lr=lrs[train_id]))
rmodel.recurrent_layer(tfsignals[:,:10,:], initial_state=rmodel.initial_state)


parameters = np.arange(0,4*np.pi+np.pi/2,np.pi/2)

l={}
preds = {}
for th in tqdm(parameters):
    rmodel.trainable_variables[0].assign(tf.convert_to_tensor(np.array([[th]]).astype(np.float32)))
    dy = tfsignals[:,:length_series,:]
    tr = rmodel(dy)
    diff = (tr - dy)[0]
    l[th] = np.sum(tf.einsum('bj,bj->b',diff,diff))/2
    preds[th] = tr
loss_values = np.array(list(l.values()))/(dt*length_series)

landscape_path=model.train_path+"landscape/"
os.makedirs(landscape_path,exist_ok=True)

np.save(landscape_path+"loss_values", loss_values)
for k, p in preds.items():
    np.save(landscape_path+"preds{}".format(k), p)
