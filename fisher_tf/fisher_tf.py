import os
import numpy as np
from misc import *
import argparse
from RNN_models import *
import tensorflow as tf
from tqdm import tqdm
#####

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

method = "rossler"
ppp = 1000
periods = 10
seed = args.seed
itraj = seed
rppp = rppp_reference = 1
train_id = 0


params = give_def_params()
params, exp_path = check_params(params)
[eta, gamma, kappa, omega, n] = params

states, covs, signals, params, times = load_data(ppp=ppp, periods=periods, method=method, itraj=itraj, exp_path=exp_path, rppp=rppp_reference)

[eta, gamma, kappa, omega, n] = params
[C, A, D , Lambda] = build_matrix_from_params(params)
[C, A, D , Lambda] = [C.astype(np.float32), A.astype(np.float32), D.astype(np.float32) , Lambda.astype(np.float32)]
total_time = times[-1]

train_path = get_path_config(periods = periods, ppp= ppp, rppp=rppp, method=method, itraj=itraj, exp_path=exp_path) +"training/train_id_{}/".format(train_id)

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
rmod(tfsignals[:,:3,:]) #just initialize


rmod.trainable_variables[0].assign( np.array([[omega]]).astype(np.float32))

def train_step(model, data):
    inputs, dys = data
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        preds = tf.squeeze(model(inputs))
        #cx2dt = tf.einsum('bj,bj->b',preds,preds)
    grads = tape.gradient(preds, model.trainable_variables)
    grads = np.squeeze((grads[0]**2).numpy())
    return grads

grads = []
#len(signals)

for k in tqdm(range(2,len(signals))): ### begin from 2, eggining from 1 retrieves an error, seems like it's not enough
    grads.append(np.squeeze(train_step(rmod, [tfsignals[:,:k,:], tfsignals[:,:k,:]])))
grads = np.array(grads)
fisher_tf_path = get_path_config(periods = periods, ppp= ppp, rppp=rppp, method=method, itraj=itraj, exp_path=exp_path) +"fisher_tf/"
os.makedirs(fisher_tf_path,exist_ok=True)
print("grads for the fisher!,  info saved in"+fisher_tf_path)
np.save(fisher_tf_path+"fisher_tf", grads)
