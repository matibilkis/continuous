import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from misc import *
import argparse
import os
from integrate import *

defpath = get_def_path()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--ppp", type=int, default=500) ###points per period
parser.add_argument("--periods", type=int, default=5)
parser.add_argument("--path", type=str, default=defpath) #
parser.add_argument("--itraj", type=int, default=1)

args = parser.parse_args()

periods = args.periods
ppp = args.ppp
path = args.path
itraj = args.itraj
method = "RK4"

path = path+"{}periods/{}ppp/".format(periods,ppp)
### INTEGRATE TRAJ
generate_traj_RK4(ppp=ppp, periods = periods, itraj=itraj, path = path, seed=itraj)

states, covs, signals, [A,dt,C,D], params = load_data(periods=periods, ppp=ppp, itraj=itraj,method="RK4")
eta, gamma, Lambda, omega, n = params

symplectic = np.array([[0,1],[-1,0]])
#e = np.pi/10
#parameters = np.arange(0,4*np.pi + e,e)
ep=0.05
parameters = np.arange(0,1+ep,ep)

preds = {t:[] for t in range(len(parameters))}
simulated_states = {t:[[states[0], covs[0]]] for t in range(len(parameters))}


give_pred = lambda st: np.dot(C,st)*dt
xi = lambda cov,D: np.dot(cov, ct(C)) + ct(D)

def evolve_simu_state(simu_st, simu_a, simu_d, dy):
    x, cov = simu_st
    XiCov = xi(cov, simu_d)
    dx = np.dot(simu_a - np.dot(XiCov,C), x)*dt + np.dot(XiCov, dy)  #evolution update (according to what you measure)
    dcov = (np.dot(simu_a,cov) + np.dot(cov, ct(simu_a)) + simu_d - np.dot(XiCov, ct(XiCov)))*dt  #covariance update
    return [x + dx, cov + dcov]


for dy in tqdm(signals):
    for i in range(len(parameters)):
        preds[i].append(give_pred(simulated_states[i][-1][0]))
        simu_a =  symplectic + np.diag([-0.5*parameters[i]])
        simu_d = np.diag([(parameters[i]*(n+0.5)) + Lambda]*2)
        simulated_states[i].append(evolve_simu_state(simulated_states[i][-1], simu_a, simu_d, dy))
path_landscape=get_def_path()+"{}periods/{}ppp/{}/cost_landscape/gamma1_".format(periods,ppp,itraj)
for i,k in simulated_states.items():
    np.save(path_landscape+str(i),np.array(k), allow_pickle=True)

print("done")
landscape = {}
cut_series = [int(k) for k in np.logspace(2,np.log10(len(signals)),10)]
for length_series in tqdm(cut_series):
    losses = []
    for i in range(len(parameters)):
        losses.append(np.sum(np.square(np.array(preds[i])[:length_series] - signals[:length_series]))/(2*dt*length_series))
    landscape[length_series] = losses

os.makedirs(path_landscape,exist_ok=True)
np.save(path_landscape+method,list(landscape.values()), allow_pickle=True)

print("landscapes done")
plt.figure(figsize=(20,7))
colors = plt.get_cmap("rainbow")
ax = plt.subplot2grid((1,1),(0,0))
for ind,p in enumerate(landscape.values()):
    plt.plot(parameters,p, color=colors(np.linspace(0,1,len(landscape)))[ind], marker='.', label="{}".format(int(np.round(cut_series[ind]+1,0))),linewidth=3)
plt.legend()
plt.xlabel(r'$\tilde{\omega}$',size=30)
plt.ylabel(r'$C(\omega, \tilde{\omega})$',size=30)
plt.savefig(path_landscape+"{}.pdf".format(method))
plt.savefig(path_landscape+"{}.png".format(method))
