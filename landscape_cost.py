import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from misc import *
import argparse
import os
import pickle
import ast

defpath = get_def_path()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--ppp", type=int)
parser.add_argument("--periods", type=int)
parser.add_argument("--path", type=str, default=defpath) #
parser.add_argument("--itraj", type=int, default=1)
parser.add_argument("--rppp", type=int, default=1)
parser.add_argument("--method", type=str, default="rossler")
parser.add_argument("--params", type=str, default="") #[eta, gamma, kappa, omega, n]

args = parser.parse_args()


periods = args.periods
ppp = args.ppp
path = args.path
itraj = args.itraj
rppp = args.rppp
method = args.method
params = args.params

rppp_reference = 1

## load "real" trajectory
params, exp_path = check_params(params)
states, covs, signals, params, times = load_data(ppp=ppp, periods=periods, method=method, itraj=itraj, exp_path=exp_path, rppp=rppp_reference)
[eta, gamma, kappa, omega, n] = params
[C, A, D , Lambda] = build_matrix_from_params(params)

## compute matrix
C_rank = np.linalg.matrix_rank(C)
xi = lambda cov,Lambda: np.dot(cov, C.T) + Lambda.T

def evolve_simu_state(x,cov, dy, simu_A, internal_step):
    XiCov = xi(cov, Lambda)
    dx = np.dot(simu_A-np.dot(XiCov,C),x)*internal_step  + np.dot(XiCov,dy)
    dcov = (np.dot(simu_A,cov) + np.dot(cov, simu_A.T) + D - np.dot(XiCov, XiCov.T))*internal_step
    return [x + dx, cov + dcov]

simu_states, simu_covs = {}, {}
#omegas = list(set([omega] + list(np.linspace(0, 2*omega, 10))))
omegas = np.array([omega]) + np.linspace(-omega/10, omega/10, 5) ## even number required so we have omega !!

Period = 2*np.pi/omega
dt = (Period/ppp)*rppp_reference ### this is because you might increase the dt as well! (1 for now, which is rppp_reference). NOTE: if you want to increase rppp you should also integrate the signals in time!

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
        loss[ind_simu_omega, indcut] = np.sum(np.square(signals[:cut] - np.einsum('ij,bj->bi',C,simu_states[simu_omega][:-1][:cut])*dt))/(C_rank*times[cut])

path_landscape= get_path_config(periods = periods, ppp= ppp, rppp=rppp, method=method, itraj=itraj, exp_path=exp_path)+"landscape/"

os.makedirs(path_landscape,exist_ok=True)
np.save(path_landscape+"losses",loss)
np.save(path_landscape+"omegas",omegas)
np.save(path_landscape+"cuts",cuts_final_time)

with open(path_landscape+"simu_states.pickle","wb") as f:
    pickle.dump(simu_states,f, protocol=pickle.HIGHEST_PROTOCOL)
