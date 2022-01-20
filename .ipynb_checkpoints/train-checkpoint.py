import numpy as np
from tqdm import tqdm
from misc import *
import tensorflow as tf
import argparse
from datetime import datetime
import os
from RNN_models import *

defpath = get_def_path()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--path", type=str, default=defpath)
parser.add_argument("--itraj", type=int, default=1)
parser.add_argument("--periods",type=int, default=50)
parser.add_argument("--ppp", type=int,default=500)
parser.add_argument("--trainid", type=int, default=0)

args = parser.parse_args()
path = args.path
itraj = args.itraj
periods = args.periods
ppp = args.ppp
train_id = args.trainid

learning_rate = 0.1
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

tf.random.set_seed(train_id)
np.random.seed(train_id)

st = datetime.now()
path = get_def_path()+"{}periods/{}ppp/".format(periods,ppp)
times = np.linspace(0,periods,ppp*periods)

truncation_times = [k for k in np.logspace(times[1000],np.log10(times[-1]), 10)]
index_series = np.argmin(np.abs(times - truncation_times[train_id]))

states, covs, signals, [A,dt,C,D], params = load_data(periods=periods, ppp=ppp, itraj=itraj,method="RK4")

times = times[:index_series]
states = states[:index_series]
covs = covs[:index_series]
signals = signals[:index_series]

tfsignals = tf.convert_to_tensor(signals)[tf.newaxis]
total_time = times[-1]


rmod = GRNNmodel(params=params, dt=dt, total_time=total_time, traj_details=[periods, ppp, train_id, itraj], cov_in=tf.convert_to_tensor(covs[0].astype(np.float32)))
rmod.compile(optimizer=optimizer)
rmod(tfsignals[:,:3,:]) #just initialize model

with open(rmod.train_path+"training_details.txt", 'w') as f:
    f.write(str(rmod.optimizer.get_config()))
f.close()

history = rmod.fit(x=tfsignals, y=tfsignals,
                     epochs=2000, callbacks = [CustomCallback(),
                                                  tf.keras.callbacks.EarlyStopping(monitor='total_loss',
                                                                                   min_delta=0, patience=500,
                                                                                   verbose=0,
                                                                                   mode='min')])
with open(rmod.train_path+"training_details.txt", 'w') as f:
    f.write(str(rmod.optimizer.get_config()))
    f.write("training time: "+str(datetime.now() - st))
f.close()
