import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from misc import *
import argparse
import os
import pickle

defpath = get_def_path()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--ppp", type=int) ###points per period
parser.add_argument("--periods", type=int)
parser.add_argument("--path", type=str, default=defpath) #
parser.add_argument("--itraj", type=int, default=1)
parser.add_argument("--rppp", type=int, default=1)
parser.add_argument("--method", type=str, default="rossler")

args = parser.parse_args()

periods = args.periods
ppp = args.ppp
path = args.path
itraj = args.itraj
rppp = args.rppp
method = args.method
states, covs, signals, params, times = load_data(ppp=ppp, periods=periods, method=method, itraj=itraj, path=get_def_path() + "rppp{}/".format(rppp))

[eta, gamma, kappa, omega, n] = params
[C, A, D , Lambda] = build_matrix_from_params(params)

xi = lambda cov,Lambda: np.dot(cov, ct(C)) + ct(Lambda)

def evolve_simu_state(x,cov, dy, simu_A, internal_step):
    XiCov = xi(cov, Lambda)
    dx = np.dot(simu_A-np.dot(XiCov,C),x)*internal_step  + np.dot(XiCov,dy)
    dcov = (np.dot(simu_A,cov) + np.dot(cov, ct(simu_A)) + D - np.dot(XiCov.T, XiCov))*internal_step
    return [x + dx, cov + dcov]

simu_states, simu_covs = {}, {}

omegas = list(set([omega] + list(np.linspace(0, 2*omega, 10))))

dt = 1/ppp
cuts_final_time = np.unique(np.concatenate([(10**k)*np.arange(1,11,1) for k in range(1,5)]))
cuts_final_time = cuts_final_time[:(np.argmin(np.abs(cuts_final_time - len(times)))+1)] -1 #the -1 is for pyhton oindexing
loss = np.zeros((len(omegas), len(cuts_final_time)))

for ind_simu_omega, simu_omega in tqdm(enumerate(omegas)):
    simu_A = np.array([[-.5*gamma, simu_omega], [-simu_omega, -0.5*gamma]])

    simu_states[simu_omega] = [states[0]]
    simu_covs[simu_omega] = [covs[0]]

    for ind,dy in enumerate(tqdm(signals)):
        simu = evolve_simu_state(simu_states[simu_omega][-1], simu_covs[simu_omega][-1], dy, simu_A,  dt)
        simu_states[simu_omega].append(simu[0])
        simu_covs[simu_omega].append(simu[1])

    for indcut, cut in enumerate(cuts_final_time):
        loss[ind_simu_omega, indcut] = np.sum(np.square(signals[:cut] - np.einsum('ij,bj->bi',C,simu_states[simu_omega][:-1][:cut])*dt))/(2*times[cut])

path_landscape=get_def_path()+"{}periods/{}ppp/{}/cost_landscape/{}".format(periods,ppp,itraj,method)

os.makedirs(path_landscape,exist_ok=True)
np.save(path_landscape+"losses",loss)
np.save(path_landscape+"omegas",omegas)
np.save(path_landscape+"cuts",cuts_final_time)

with open(path_landscape+"simu_states.pickle","wb") as f:
    pickle.dump(simu_states,f, protocol=pickle.HIGHEST_PROTOCOL)
