import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numerics.utilities.misc import *

def get_training_save_dir(exp_path, total_time, dt, itraj,train_id):
    save_dir = get_path_config(exp_path=exp_path,total_time = total_time, dt = dt, itraj=itraj, ext_signal=1)+"training_{}/".format(train_id)
    return save_dir

def pre_process_data_for_ML(total_time, dt, signals):

    times = get_time(total_time,dt).astype(np.float32)
    dd = tf.unstack(signals.astype(np.float32),axis=1)

    tfsignals = tf.stack([times[:-1],dd[0], dd[1]])
    tfsignals = tf.transpose(tfsignals)[tf.newaxis]
    return tfsignals





def plot_history(logs=None, from_loss=False, data=None, preds=None, signals=None, **kwargs):
    true_parameters = kwargs.get("true_parameters",[10., 2*np.pi])

    plt.figure(figsize=(20,5))

    if preds is None:

        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)
    else:
        ax1 = plt.subplot(141)
        ax2 = plt.subplot(142)
        ax3 = plt.subplot(143)
        ax4 = plt.subplot(144)

    if from_loss == True:

        history_loss = np.squeeze([logs[k]["LOSS"] for k in range(len(logs))])
        params=np.squeeze([logs[k]["PARAMS"] for k in range(len(logs))])
        grads = np.squeeze([logs[k]["LOSS"] for k in range(len(logs))])
    else:
        history_loss, params, grads = data

    ax1.set_title("LOSS")
    ax1.plot(history_loss)
    ax1.set_yscale("log")
    ax1.set_xlabel("GRADIENT STEP")

    ax2.set_title("PARAMS")
    ax2.plot(params[:,0],label="RNN -p0")
    ax2.plot(params[:,1],label="RNN -p1")

    ax2.plot(np.ones(len(params))*true_parameters[0], '--',label="true")
    ax2.plot(np.ones(len(params))*true_parameters[1], '--',label="true")

    ax2.set_yscale("log")
    ax2.set_xlabel("GRADIENT STEP")
    ax2.legend()

    ax3.set_title("GRADS")
    ax3.plot(grads)
    ax3.loglog()


    if preds is not None:
        ax4.set_title("PREDICTIONS")
        ax4.plot(np.squeeze(preds)[:,0], '--',color="red", alpha=0.5,label="RNN")
        ax4.plot(signals[0], color="blue", label="true", alpha=0.5)
        ax4.legend()
