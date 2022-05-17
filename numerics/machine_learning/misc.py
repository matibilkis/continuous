import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numerics.utilities.misc import *



def pre_process_data_for_ML(total_time, dt, signals):

    times = get_time(total_time,dt).astype(np.float32)
    dd = tf.unstack(signals.astype(np.float32),axis=1)

    tfsignals = tf.stack([times[:-1],dd[0], dd[1]])
    tfsignals = tf.transpose(tfsignals)[tf.newaxis]
    return tfsignals





def plot_history(logs, preds=None, signals=None):
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

    history_loss = np.squeeze([logs[k]["LOSS"] for k in range(len(logs))])
    ax1.set_title("LOSS")
    ax1.plot(history_loss)
    ax1.loglog()
    ax1.set_xlabel("GRADIENT STEP")

    params=np.squeeze([logs[k]["PARAMS"] for k in range(len(logs))])
    ax2.set_title("PARAMS")
    ax2.plot(params[:,0],label="RNN")
    ax2.plot(np.ones(len(params))*true_parameters[0], '--',label="true")
    ax2.set_xlabel("GRADIENT STEP")
    ax2.legend()

    grads = np.squeeze([logs[k]["LOSS"] for k in range(len(logs))])
    ax3.set_title("GRADS")
    ax3.plot(grads)
    ax3.loglog()


    if preds is not None:
        ax4.set_title("PREDICTIONS")
        ax4.plot(np.squeeze(preds)[:,0], '--',color="red", alpha=0.5,label="RNN")
        ax4.plot(signals[0], color="blue", label="true", alpha=0.5)
        ax4.legend()
