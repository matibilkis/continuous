import numpy as np
import matplotlib.pyplot as plt 

def display_histories(fig,histories):

    axarr=fig.add_subplot(2,1,1)
    ax = fig.axes[-1]
    for k in list(histories.keys()):
        ax.plot(histories[k][1], label=k)
    ax.set_yscale('log')
    ax.legend()
    ax.set_title("Loss evolution")

    axarr=fig.add_subplot(2,1,2)
    ax = fig.axes[-1]
    #for k in range(1,9):
    for k in list(histories.keys()):
        ax.plot(np.abs(np.reshape(histories[k][2], [-1,histories[k][2].shape[1]])), label=k)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title("Grads evolution")
    ax.legend()
    return fig