import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
from tqdm import tqdm
from numerics.utilities.misc import *#get_def_path, ct, s_to_cov, convert_solution, get_path_config
import tensorflow as tf
import argparse
from datetime import datetime
from numerics.machine_learning.RNN_models import *

defpath = get_def_path()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--ppp", type=int)
parser.add_argument("--periods", type=int)
parser.add_argument("--path", type=str, default=defpath) #
parser.add_argument("--itraj", type=int, default=1)
parser.add_argument("--method", type=str, default="rossler")
parser.add_argument("--params", type=str, default="") #[eta, gamma, kappa, omega, n]
parser.add_argument("--trainid", type=int, default=0)
parser.add_argument("--rppp", type=int, default=1)
parser.add_argument("--epochs", type=int, default=1000)

args = parser.parse_args()

periods = args.periods
ppp = args.ppp
path = args.path
itraj = args.itraj
rppp_reference = rppp = args.rppp
method = args.method
params = args.params
train_id = args.trainid
epochs = args.epochs


params, exp_path = check_params(params)
states, covs, signals, params, times = load_data(ppp=ppp, periods=periods, method=method, itraj=itraj, exp_path=exp_path, rppp=rppp_reference, fisher=False)
[eta, gamma, kappa, omega, n] = params
[C, A, D , Lambda] = build_matrix_from_params(params)
[C, A, D , Lambda] = [C.astype(np.float32), A.astype(np.float32), D.astype(np.float32) , Lambda.astype(np.float32)]
total_time = times[-1]


st = datetime.now()


print(len(times))
indices = [int(k) for k in np.logspace(1, np.log10(len(times)), 10)]
index_series = indices[train_id]

times = times[:index_series]
states = states[:index_series]
covs = covs[:index_series]
signals = signals[:index_series]

tfsignals = tf.convert_to_tensor(signals)[tf.newaxis]
total_time = times[-1]

train_path = get_path_config(periods = periods, ppp= ppp, rppp=rppp, method=method, itraj=itraj, exp_path=exp_path) +"training/train_id_{}/".format(train_id)
os.makedirs(train_path, exist_ok=True)

tf.random.set_seed(train_id)
np.random.seed(train_id)


learning_rate = float(omega/50)
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
tfsignals = tf.convert_to_tensor(signals)[tf.newaxis]




rmod = GRNNmodel(params=params,
                dt=(2*np.pi)/(ppp*omega),
                total_time=times[-1],
                cov_in=tf.convert_to_tensor(covs[0].astype(np.float32)),
                train_path=train_path
                )

rmod.compile(optimizer=optimizer)
rmod(tfsignals[:,:3,:]) #just initialize model
#
with open(rmod.train_path+"training_details.txt", 'w') as f:
    f.write(str(rmod.optimizer.get_config()))
f.close()

history = rmod.fit(x=tfsignals, y=tfsignals,
                     epochs=epochs, callbacks = [CustomCallback(),
                                                  tf.keras.callbacks.EarlyStopping(monitor='total_loss',
                                                                                   min_delta=0, patience=200,
                                                                                   verbose=0,
                                                                                   mode='min')])
with open(rmod.train_path+"training_details.txt", 'w') as f:
    f.write(str(rmod.optimizer.get_config()))
    f.write("training time: "+str(datetime.now() - st))
f.close()
