import sys
import os 
import argparse
sys.path.insert(0, os.getcwd())

from numerics.integration.steps import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numerics.utilities.misc import *
import time
from scipy.special import erf
import pickle
import matplotlib


plotdir = "figuras/"
os.makedirs(plotdir, exist_ok=True)

mode = "frequencies"
pars = give_def_params_discrimination(flip=0, mode = mode)
params, exp_path = check_params_discrimination(pars)
[gamma1, omega1, n1, eta1, kappa1], [gamma0, omega0, n0, eta0, kappa0] = params

dtt = 1e-6
total_time = 50.
total_time, dt = get_total_time_dt(params, dt=dtt, total_time=total_time)
times = np.arange(0, total_time+ dt, dt)


indis = np.logspace(1,np.log10(len(times)-1), 1000)
indis = [int(k) for k in indis]
timind = [times[ind] for ind in indis]


Ntraj = 2000
boundsB= np.arange(-4,4.1,.1)


path = get_def_path()+"results_stopping_time/paper_{}/".format(mode)
path_data = get_def_path()+"results_stopping_time/paper_{}/".format(mode)


timbin1 = np.load(path_data+"timbin.npy")
cons1 = np.load(path_data+"cons.npy")#, cons1)
timbin0 = np.load(path_data+"timbin.npy")#, timbin0)
cons0 = np.load(path_data+"cons.npy")#, cons0)
gp0 = np.load(path_data+"gp0.npy")#, gp0)
gp1 = np.load(path_data+"gp1.npy")#, gp1)
deter_data_h1_h0 = np.load(path_data+"deth1h0.npy")#, deter_data_h1_h0)
deteR_data_h0_h1 = np.load(path_data+"deth0h1.npy")#, deter_data_h0_h1)
anals0 = np.load(path_data+"anals0.npy")
anals1 = np.load(path_data+"anals1.npy")#,anals1)
#l0 = np.load(path_data+"l0.npy")
#l1 = np.load(path_data+"l1.npy")


with open(path_data+"stop.pickle","rb") as f:
    stop = pickle.load( f)#, protocol=pickle.HIGHEST_PROTOCOL)

with open(path_data+"deter.pickle","rb") as f:
    deter = pickle.load(f)#, protocol=pickle.HIGHEST_PROTOCOL)
    
alphas = list(deter["h0/h1"].values())
betas = list(deter["h1/h0"].values())

alphas = np.stack(alphas)
betas = np.stack(betas)



print("plotting HISTOGRAM FREQUENCY")

LS, TS = 60, 40
plt.figure(figsize=(20,20))
ax = plt.subplot(111)
indb = -5
ax.bar(timbin1[indb], cons1[indb], width=timbin1[indb][1]-timbin1[indb][0], color="red", alpha=0.75, edgecolor="black",)#, label="simulations")
ax.set_xlabel(r'$\tau$',size=LS)
ax.set_ylabel(r'$P(\tau)$', size=LS)
ax.tick_params(axis='both', which='major', labelsize=TS)
plt.savefig(plotdir+"freq_histogram.pdf")


print("plotting alphas/betas error")




maps = plt.get_cmap("cividis")

plt.figure(figsize=(20,20))
#plt.suptitle("some numerics on " + r'$P(H_1|H_0)$', size=20)
ax=plt.subplot(111)
indboundsplot = list(range(0,len(boundsB)))[0:38:10]
boundsplot = [boundsB[int(k)] for k in indboundsplot]
colors = maps(np.linspace(0,1,len(boundsplot)))
c=-1
l=-250
for k, b in zip(indboundsplot, boundsplot):
    c+=1
    ax.scatter((1/betas[k,:])[:l],timind[:l],  color=colors[c], s=200, alpha=0.5)
    ax.plot((1/betas[k,:])[:l],timind[:l],  color=colors[c], linewidth=2, alpha=0.5, label="b={}".format(np.round(b,2)))
    
ax.legend(prop={"size":LS})
ax.set_xscale("log")
ax.set_xlabel(r'$\log 1/\epsilon$',size=LS)
ax.tick_params(axis='both', which='major', labelsize=TS)
ax.set_ylabel("t",size=LS)
plt.savefig(plotdir+"freq_beta.pdf")



maps = plt.get_cmap("cividis")
plt.figure(figsize=(20,20))

bpos = boundsB[boundsB>=0]
bneg = boundsB[boundsB<0]

#plt.suptitle("some numerics on " + r'$P(H_1|H_0)$', size=20)
ax=plt.subplot(111)
indboundsplot = list(range(0,len(boundsB)))[len(bneg)+10:len(boundsB):10]
boundsplot = [boundsB[int(k)] for k in indboundsplot]
colors = maps(np.linspace(0,1,len(boundsplot)))
c=-1
l=-250
for k, b in zip(indboundsplot, boundsplot):
    c+=1
    ax.scatter((1/alphas[k,:])[:l],timind[:l],  color=colors[c], s=200, alpha=0.5)
    ax.plot((1/alphas[k,:])[:l],timind[:l],  color=colors[c], linewidth=2, alpha=0.5, label="b={}".format(np.round(b,2)))
    
ax.legend(prop={"size":LS})
ax.set_xscale("log")
ax.tick_params(axis='both', which='major', labelsize=TS)
ax.set_xlabel(r'$\log 1/\epsilon$',size=LS)
ax.set_ylabel("t",size=LS)
plt.savefig(plotdir+"freq_alpha.pdf")




print("comparing freq det stoch")



bpos = boundsB[boundsB>=0]
bneg = boundsB[boundsB<0]

avg_err_alpha = lambda o: (1-np.exp(-abs(o)))/(np.exp(abs(o)) - np.exp(-abs(o)))
avg_err_beta = lambda o :(1-np.exp(-abs(o)))/(np.exp(abs(o)) - np.exp(-abs(o)))

errs = np.array([avg_err_alpha(b) for b in boundsB]) #
tot_err = 0.5*(alphas+betas)#0.5*(alphas + betas)
times_to_errs = [timind[np.argmin(np.abs(tot_err[indb,:] - errs[indb]))] for indb in range(len(bpos))]



stops0 = [[] for k in range(len(bpos))]
stops1 = [[] for k in range(len(bpos))]

values1 = list(stop["_1"].values())
values0 = list(stop["_0"].values())
for k,val in enumerate(values1):
    if len(val)!=0:
        for indb in range(len(val)):
            if ~np.isnan([values1[k][indb]])[0] == True:
                stops1[indb].append(np.squeeze(values1[k][indb]))
        
for k,val in enumerate(values0):
    if len(val)!=0:
        for indb in range(len(val)):
            if ~np.isnan([values0[k][indb]])[0] == True:
                stops0[indb].append(np.squeeze(values0[k][indb]))


avg_times1 = np.array([np.mean(k) for k in stops1])
avg_times0 = np.array([np.mean(k) for k in stops0])

std_times1 = np.array([np.std(k) for k in stops1])
std_times0 = np.array([np.std(k) for k in stops0])
avg_times = 0.5*(avg_times0 + avg_times1)

std_times = np.sqrt(std_times1**2   + std_times0**2)#0.5*(np.array(avg_times0) + np.array(avg_times1) )
stoch = avg_times
stoch_std = std_times


avg_err_alpha = lambda o: (1-np.exp(-o))/(np.exp(o) - np.exp(-o))
errs = [avg_err_alpha(b) for b in bpos]
times_alpha_to_errB = [timind[np.argmin(np.abs(alphas[indb+len(bneg),:]+betas[len(bneg)-indb+1,:] - errs[indb]))] for indb in range(len(bpos))]


fig = plt.figure(figsize=(25,25))
ax = plt.subplot(111)
lw=10
ax.scatter(errs, times_alpha_to_errB,color="red", alpha=0.8,s=200, label="deterministic")
ax.scatter(errs, stoch ,color="blue",s=200,  alpha=0.8,label="stochastic")
#ax.fill_between(errs, stoch - stoch_std/2, stoch + stoch_std/2, alpha=0.5, color="blue")
ax.set_xscale("log")
ax.set_xticks([np.round(k,2) for k in np.linspace(min(errs),max(errs),4)])
ax.set_xlabel(r'$P_e$', size=LS)
ax.tick_params(axis='both', which='major', labelsize=TS)
ax.set_ylabel("t", size=LS)
ax.legend(prop={"size":LS})
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.savefig(plotdir+"freq_adaptive_vs_det.pdf")





print("DAMPING")



mode = "damping"
pars = give_def_params_discrimination(flip=0, mode = mode)
params, exp_path = check_params_discrimination(pars)
[gamma1, omega1, n1, eta1, kappa1], [gamma0, omega0, n0, eta0, kappa0] = params

dtt = 1e-6
total_time = 6.
total_time, dt = get_total_time_dt(params, dt=dtt, total_time=total_time)
times = np.arange(0, total_time+ dt, dt)


indis = np.logspace(1,np.log10(len(times)-1), 1000)
indis = [int(k) for k in indis]
timind = [times[ind] for ind in indis]


Ntraj = 2000
boundsB= np.arange(-4,4.1,.1)


path = get_def_path()+"results_stopping_time/paper_{}/".format(mode)
path_data = get_def_path()+"results_stopping_time/paper_{}/".format(mode)


timbin1 = np.load(path_data+"timbin.npy")
cons1 = np.load(path_data+"cons.npy")#, cons1)
timbin0 = np.load(path_data+"timbin.npy")#, timbin0)
cons0 = np.load(path_data+"cons.npy")#, cons0)
gp0 = np.load(path_data+"gp0.npy")#, gp0)
gp1 = np.load(path_data+"gp1.npy")#, gp1)
deter_data_h1_h0 = np.load(path_data+"deth1h0.npy")#, deter_data_h1_h0)
deteR_data_h0_h1 = np.load(path_data+"deth0h1.npy")#, deter_data_h0_h1)
anals0 = np.load(path_data+"anals0.npy")
anals1 = np.load(path_data+"anals1.npy")#,anals1)
l0 = np.load(path_data+"l0.npy")
l1 = np.load(path_data+"l1.npy")

ll1={}
ll0={}
for k in range(1,5):
    ll1[k], ll0[k] = load_liks(itraj=k, mode=mode, dtt=dt, total_time_in=total_time)
    
    
colors1 = plt.get_cmap("Reds")
colors0 = plt.get_cmap("Blues")
Ntraj = 4
Ltraj=3
c1 = colors1(np.linspace(0,1,Ntraj))[::-1]
c0 = colors0(np.linspace(0,1,Ntraj))[::-1]

jump = 100
plt.figure(figsize=(20,20))
ax=plt.subplot(111)
ax.plot(times[::jump],l0[:-1][::jump], color="red", alpha=0.8, linewidth=10, label=r'$\langle \ell_{|1} \rangle$')
ax.plot(times[::jump],l1[:-1][::jump], color="blue", alpha=0.8, linewidth=10, label=r'$\langle \ell_{|0} \rangle$')
for ind,k in enumerate(range(1,5)):
    ax.plot(times[::jump],ll1[k][:-1][::jump], color=c1[ind], linewidth=Ltraj, alpha=0.6)
    ax.plot(times[::jump],ll0[k][:-1][::jump], color=c0[ind], linewidth=Ltraj, alpha=0.6)   
ax.plot(times[::jump], np.zeros(len(times))[::jump], '--', color="black")
ax.set_xlabel("t", size=LS)
ax.set_ylabel(r'$\ell$', size=LS)
ax.tick_params(axis='both', which='major', labelsize=TS)
ax.legend(prop={"size":LS})
plt.savefig(plotdir+"liks_damping.pdf")




with open(path_data+"stop.pickle","rb") as f:
    stop = pickle.load( f)#, protocol=pickle.HIGHEST_PROTOCOL)

with open(path_data+"deter.pickle","rb") as f:
    deter = pickle.load(f)#, protocol=pickle.HIGHEST_PROTOCOL)
    
alphas = list(deter["h0/h1"].values())
betas = list(deter["h1/h0"].values())

alphas = np.stack(alphas)
betas = np.stack(betas)


bpos = boundsB[boundsB>=0]
bneg = boundsB[boundsB<0]

avg_err_alpha = lambda o: (1-np.exp(-abs(o)))/(np.exp(abs(o)) - np.exp(-abs(o)))
avg_err_beta = lambda o :(1-np.exp(-abs(o)))/(np.exp(abs(o)) - np.exp(-abs(o)))

errs = np.array([avg_err_alpha(b) for b in boundsB]) #
tot_err = 0.5*(alphas+betas)#0.5*(alphas + betas)
times_to_errs = [timind[np.argmin(np.abs(tot_err[indb,:] - errs[indb]))] for indb in range(len(bpos))]


stops0 = [[] for k in range(len(bpos))]
stops1 = [[] for k in range(len(bpos))]

values1 = list(stop["_1"].values())
values0 = list(stop["_0"].values())
for k,val in enumerate(values1):
    if len(val)!=0:
        for indb in range(len(val)):
            if ~np.isnan([values1[k][indb]])[0] == True:
                stops1[indb].append(np.squeeze(values1[k][indb]))
        
for k,val in enumerate(values0):
    if len(val)!=0:
        for indb in range(len(val)):
            if ~np.isnan([values0[k][indb]])[0] == True:
                stops0[indb].append(np.squeeze(values0[k][indb]))


avg_times1 = np.array([np.mean(k) for k in stops1])
avg_times0 = np.array([np.mean(k) for k in stops0])

std_times1 = np.array([np.std(k) for k in stops1])
std_times0 = np.array([np.std(k) for k in stops0])
avg_times = 0.5*(avg_times0 + avg_times1)

std_times = np.sqrt(std_times1**2   + std_times0**2)#0.5*(np.array(avg_times0) + np.array(avg_times1) )
stoch = avg_times
stoch_std = std_times


print("DAMPING CHISTOGRAM")
plt.figure(figsize=(20,20))
ax = plt.subplot(111)
indb = -5
ax.bar(timbin1[indb], cons1[indb], width=timbin1[indb][1]-timbin1[indb][0], color="red", alpha=0.75, edgecolor="black",label="simulations")
ax.plot(np.linspace(0,np.max(timbin1[indb]), 100), anals1[indb], linewidth=8, color="black", alpha=0.8, label="analytical")
ax.set_xlabel(r'$\tau$',size=LS)
ax.set_ylabel(r'$P(\tau)$', size=LS)
ax.legend(prop={"size":LS})
ax.tick_params(axis='both', which='major', labelsize=TS)
plt.savefig(plotdir+"damp_compa.pdf")





avg_err_alpha = lambda o: (1-np.exp(-o))/(np.exp(o) - np.exp(-o))
errs = [avg_err_alpha(b) for b in bpos]
times_alpha_to_errB = [timind[np.argmin(np.abs(alphas[indb+len(bneg),:]+betas[len(bneg)-indb+1,:] - errs[indb]))] for indb in range(len(bpos))]


fig = plt.figure(figsize=(20,20))
ax = plt.subplot(111)
lw=10
LS = 30
ax.scatter(errs, times_alpha_to_errB, color="red", s=200,label="deterministic")
ax.scatter(errs, stoch, color="blue", s=200, label="stochastic")
#ax.fill_between(errs, stoch - stoch_std/2, stoch + stoch_std/2, alpha=0.5, color="blue")
ax.set_xscale("log")
ax.set_xticks([np.round(k,2) for k in np.linspace(min(errs),max(errs),4)])
ax.set_xlabel(r'$P_e$', size=LS)
ax.tick_params(axis='both', which='major', labelsize=TS)
ax.set_ylabel("t", size=LS)
ax.legend(prop={"size":20})
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.savefig(plotdir+"damp_comparison.pdf")