import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from misc import *
import tensorflow as tf
from RNN_models import *
import argparse
import os

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--path", type=str, default="/data/uab-giq/scratch/matias/quantera/trajectories/")
parser.add_argument("--itraj", default=0)
parser.add_argument("--epochs", type=int, default=100)

args = parser.parse_args()
path, itraj, epochs = args.path, int(float(args.itraj)), int(float(args.epochs))


means = np.load(path+"{}/means.npy".format(itraj), allow_pickle=True).astype(np.float32) ### this is \textbf{q}(t)
covs = np.load(path+"{}/covs.npy".format(itraj), allow_pickle=True).astype(np.float32) ## this is the \Sigma(t)
xicovs = np.load(path+"{}/xicovs.npy".format(itraj), allow_pickle=True).astype(np.float32) ## this is the \Chi(\Sigma) (evolution)
signals = np.load(path+"{}/signals.npy".format(itraj), allow_pickle=True).astype(np.float32) ##this is the dy's
A = np.load(path+"{}/A.npy".format(itraj), allow_pickle=True).astype(np.float32)
dt = np.load(path+"{}/dt.npy".format(itraj), allow_pickle=True)[0]
C = np.load(path+"{}/C.npy".format(itraj), allow_pickle=True).astype(np.float32)
D = np.load(path+"{}/D.npy".format(itraj), allow_pickle=True).astype(np.float32)

coeffs = [C, A, D , dt]

model = GaussianRecuModel(coeffs)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.05))
model(sliced_dataset(signals, xicovs,1))
initial_A = model.trainable_variables


### training ####
history_A, history_loss = [], []
for time_slice in [-1]:      #tqdm(range(10,len(signals),3000)):
    inputs = sliced_dataset(signals, xicovs,time_slice)
    histo = model.fit(x=inputs, y=inputs[1][tf.newaxis,:,:], epochs=epochs, verbose=1)
    for k,v in zip(histo.history["Coeffs_A"], histo.history["total_loss"]):
        history_A.append(k)
        history_loss.append(v)


#os.makedirs("/data/uab-giq/scratch/matias/quantera/trajectories/{}/".format(itraj), exist_ok=True)
np.save(path+"{}/A_history".format(itraj),np.array(history_A) )
np.save(path+"{}/loss_history".format(itraj),np.array(history_loss) )
