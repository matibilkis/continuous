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
parser.add_argument("--itraj", type=int, default=0)
parser.add_argument("--periods",type=int, default=40)
parser.add_argument("--ppp", type=int,default=500)
parser.add_argument("--trainid", type=int, default=0)

args = parser.parse_args()
path = args.path
itraj = args.itraj
periods = args.periods
ppp = args.ppp
train_id = args.trainid

#optimizer = {0:tf.keras.optimizers.Adam, 1:tf.keras.optimizers.SGD}

learning_rate = 0.01
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
#optimizer = optimizers[train_id%2](lr=learning_rate)
tf.random.set_seed(train_id)
np.random.seed(train_id)

st = datetime.now()
path = path+"{}periods/{}ppp/".format(periods,ppp)


means, covs, signals, coeffs = load_data(path, itraj=itraj)
tfsignals = tf.convert_to_tensor(signals)[tf.newaxis]
A,dt,C,D = coeffs
total_time = dt*ppp*periods


rmod = GRNNmodel(coeffs = [C.astype(np.float32),D.astype(np.float32),dt, total_time], traj_details=[periods, ppp, train_id, itraj], cov_in=tf.convert_to_tensor(covs[0].astype(np.float32)))
    
rmod.compile(optimizer=optimizer)
rmod(tfsignals[:,:3,:]) #just initialize model

with open(rmod.train_path+"training_details.txt", 'w') as f:
    f.write(str(rmod.optimizer.get_config()))
    f.write("training time: "+str(datetime.now() - st))
f.close()


history = rmod.fit(x=tfsignals, y=tfsignals,
                     epochs=2000, callbacks = [CustomCallback(),
                                                  tf.keras.callbacks.EarlyStopping(monitor='total_loss',
                                                                                   min_delta=0, patience=100,
                                                                                   verbose=0,
                                                                                   mode='min')])
with open(rmod.train_path+"training_details.txt", 'w') as f:
    f.write(str(rmod.optimizer.get_config()))
    f.write("training time: "+str(datetime.now() - st))
f.close()
