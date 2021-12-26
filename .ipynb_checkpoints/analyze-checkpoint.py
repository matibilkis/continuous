import numpy as np
import matplotlib.pyplot as plt 

def display_histories(fig,histories):

    axarr=fig.add_subplot(2,1,1)
    ax = fig.axes[-1]
    for k in range(1,9):
        ax.plot(histories[k][1], label=k)
    ax.set_yscale('log')
    ax.legend()
    ax.set_title("Loss evolution")

    axarr=fig.add_subplot(2,1,2)
    ax = fig.axes[-1]
    #for k in range(1,9):
    ax.plot(np.abs(np.reshape(histories[1][2], [-1,4])), label=k)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title("Grads evolution")
    ax.legend()
    return fig