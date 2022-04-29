import os 
import sys
import argparse
sys.path.insert(0, os.getcwd())
print(os.getcwd())
from numerics.integration.steps import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numerics.utilities.misc import *
import time
from scipy.special import erf
import pickle


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--mode", type=str, default="damping")
args = parser.parse_args()
mode = args.mode


def load_liks(itraj, mode="damping", dtt=1e-6, total_time_in=6.):
    pars = give_def_params_discrimination(flip=0, mode = mode)
    params, exp_path = check_params_discrimination(pars)
    [gamma1, omega1, n1, eta1, kappa1], [gamma0, omega0, n0, eta0, kappa0] = params
    logliks =load_data_discrimination_liks(itraj=itraj, total_time = total_time_in, dt=dtt, exp_path = exp_path)
    l1  = logliks[:,0] - logliks[:,1]
    
    pars = give_def_params_discrimination(flip=1, mode = mode)
    params, exp_path = check_params_discrimination(pars)
    [gamma1, omega1, n1, eta1, kappa1], [gamma0, omega0, n0, eta0, kappa0] = params
    logliks =load_data_discrimination_liks(itraj=itraj, total_time = total_time_in, dt=dtt, exp_path = exp_path)
    l0  = logliks[:,1] - logliks[:,0]
      
    return l0, l1


mode = "damping"
#if mode == "damping":
dtt = 1e-6
total_time = 6.
#elif mode == "frequencies":
 #   dtt = 1e-6
 #   total_time = 50.#
#else:
#    print("che, undefined", mode)
pars = give_def_params_discrimination(flip=0, mode = mode)
params, exp_path = check_params_discrimination(pars)
[gamma1, omega1, n1, eta1, kappa1], [gamma0, omega0, n0, eta0, kappa0] = params

total_time, dt = get_total_time_dt(params, dt=dtt, total_time=total_time)
times = np.arange(0, total_time+ dt, dt)


indis = np.logspace(1,np.log10(len(times)-1), 1000)
indis = [int(k) for k in indis]
timind = [times[ind] for ind in indis]


Ntraj = 2000
boundsB= np.arange(-4,4.1,.1)


stop = {}
stop["_0"] = {i:[] for i in range(1,Ntraj)}
stop["_1"] = {i:[] for i in range(1,Ntraj)}


er=[]
deter = {}
deter["h0/h1"] = {indb:[0]*len(indis) for indb in range(len(boundsB))}
deter["h1/h0"] = {indb:[0]*len(indis) for indb in range(len(boundsB))}

l0,l1 = np.zeros((2,len(times)+1))

deter_data_h0_h1 = np.zeros((Ntraj, len(boundsB), len(indis)))#{indb:{ind:[] for ind in indis} for indb in range(len(boundsB))}#["h1/h0"] = {indb:[0]*len(indis) for indb in range(len(boundsB))}
deter_data_h1_h0 = np.zeros((Ntraj, len(boundsB), len(indis)))##{indb:{ind:[] for ind in indis} for indb in range(len(boundsB))}#["h1/h0"] = {indb:[0]*len(indis) for indb in range(len(boundsB))}




n=1
ers = []
for itraj in tqdm(range(1,Ntraj)):

    log_lik_ratio, log_lik_ratio_swap = load_liks(itraj=itraj, mode=mode, dtt=dt, total_time_in=total_time)
    for indb,b in enumerate(boundsB):
        if b>=0:
            stop["_1"][itraj].append(get_stop_time(log_lik_ratio, b, times))
            stop["_0"][itraj].append(get_stop_time(log_lik_ratio_swap, b,times))

        for ind, time_index in enumerate(indis):
            deter_1 = log_lik_ratio[time_index] < b
            deter_0 = log_lik_ratio_swap[time_index] > b


            deter_data_h0_h1[itraj][indb][ind] = deter_1
            deter_data_h1_h0[itraj][indb][ind] = deter_0

            deter["h0/h1"][indb][ind] +=  (int(deter_1) - deter["h0/h1"][indb][ind] )/n
            deter["h1/h0"][indb][ind] += (int(deter_0) - deter["h1/h0"][indb][ind])/n
    l1= l1 + log_lik_ratio
    l0 = l0 +log_lik_ratio_swap
    n+=1
l0/=(Ntraj - len(ers))
l1/=(Ntraj - len(ers))



bpos = boundsB[boundsB >=0]
bneg = boundsB[boundsB <0]

stops0 = [[] for k in range(len(bpos))]
stops1 = [[] for k in range(len(bpos))]

values1 = list(stop["_1"].values())
values0 = list(stop["_0"].values())
for k,val in enumerate(values1):
    if len(val)>2:
        for indb in range(len(val)):
            if ~np.isnan([values1[k][indb]])[0] == True:
                stops1[indb].append(np.squeeze(values1[k][indb]))
        
for k,val in enumerate(values0):
    if len(val)!=0:
        for indb in range(len(val)):
            if ~np.isnan([values0[k][indb]])[0] == True:
                stops0[indb].append(np.squeeze(values0[k][indb]))

                
                
cons1, cons0 = [], []
anals1, anals0 = [], []
timbin0, timbin1 = [], []
for indb, b in enumerate(boundsB[int(len(boundsB)/2):]):
    counts1, bins1 = np.histogram(stops1[indb], 50, normed=True)
    counts0, bins0 = np.histogram(stops0[indb], 50, normed=True)
    
    timms1 = np.linspace(0,np.max(bins1), 100)
    timms0 = np.linspace(0,np.max(bins0), 100)
    
    timbins1 = .5*(bins1[1:] + bins1[:-1])
    timbins0 = .5*(bins0[1:] + bins0[:-1])
    
    cons1.append(counts1)
    cons0.append(counts0)
    
    timbin1.append(timbins1)
    timbin0.append(timbins0)
    
    gp1 =[prob(t, b, kappa0, kappa1, eta0 , eta1, n0, n1, gamma0, gamma1)[0] for t in timms1]
    anals1.append(gp1)
    
    gp0 = [prob(t, b, kappa1, kappa0, eta1 , eta0, n1, n0, gamma1, gamma0)[0] for t in timms0]
    anals0.append(gp0)
    
    
    
path = get_def_path()+"results_stopping_time/paper_{}/".format(mode)
path_data = get_def_path()+"results_stopping_time/paper_{}/".format(mode)
os.makedirs(path,exist_ok=True)
os.makedirs(path_data,exist_ok=True)



with open(path_data+"stop.pickle","wb") as f:
    pickle.dump(stop, f, protocol=pickle.HIGHEST_PROTOCOL)
    

with open(path_data+"deter.pickle","wb") as f:
    pickle.dump(deter, f, protocol=pickle.HIGHEST_PROTOCOL)
    
np.save(path_data+"timbin", timbin1)
np.save(path_data+"cons", cons1)
np.save(path_data+"timbin", timbin0)
np.save(path_data+"cons", cons0)
np.save(path_data+"gp0", gp0)
np.save(path_data+"gp1", gp1)
np.save(path_data+"deth1h0", deter_data_h1_h0)
np.save(path_data+"deth0h1", deter_data_h0_h1)
np.save(path_data+"anals0",anals0)
np.save(path_data+"anals1",anals1)

np.save(path_data+"l0",l0)
np.save(path_data+"l1",l1)