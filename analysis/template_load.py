%load_ext autoreload
%autoreload 2

import os
from numerics.utilities.misc import *
import matplotlib.pyplot as plt


def load(itraj = 1,total_time = 50., dt = 1e-3, exp_path=""):
    pp = get_path_config(total_time=total_time, dt=dt, itraj=itraj, exp_path = exp_path)
    states = np.load(pp+"states.npy")
    dys = np.load(pp+"dys.npy")
    return states, dys


params, exp_path = def_params()

ss=[]
dyy=[]
for k in range(1,6):
    states, dys = load(itraj=k, exp_path=exp_path, total_time=4., dt=1e-3)
    ss.append(states[:,0])
    dyy.append(dys[:,0])
