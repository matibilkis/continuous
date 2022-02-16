import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from misc import *
import argparse
import os
import pickle
import ast
from numba import jit


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

[C, A, D , Lambda] = [C.astype(np.float32), A.astype(np.float32), D.astype(np.float32) , Lambda.astype(np.float32)]


simu_states, simu_covs = {}, {}
omegas = np.array([omega]) + np.linspace(-omega/10, omega/10, 5) ## even number required so we have omega !!

Period = 2*np.pi/omega

global dt

dt = (Period/ppp)*rppp_reference ### this is because you might increase the dt as well! (1 for now, which is rppp_reference). NOTE: if you want to increase rppp you should also integrate the signals in time!

cuts_final_time = np.unique(np.concatenate([(10**k)*np.arange(1,11,1) for k in range(1,5)]))
cuts_final_time = cuts_final_time[:(np.argmin(np.abs(cuts_final_time - len(times)))+1)] -1 #the -1 is for pyhton oindexing

loss = np.zeros((len(omegas), len(cuts_final_time)))



@jit(nopython=True)
def integrate_with_omega(simu_omega,ind_simu_omega, signals, simu_A, simu_states_omega, simu_covs_omega):
    for ind,dy in enumerate(signals):

        x = simu_states_omega[-1].astype(np.float32)
        cov = simu_covs_omega[-1].astype(np.float32)

        XiCov = np.dot(cov, C.T) + Lambda.T
        dx = np.dot(simu_A-np.dot(XiCov,C),x)*dt  + np.dot(XiCov,dy)
        dcov = (np.dot(simu_A,cov) + np.dot(cov, simu_A.T) + D - np.dot(XiCov, XiCov.T))*dt

        simu_states_omega.append( (x + dx).astype(np.float32))
        simu_covs_omega.append( (cov + dcov).astype(np.float32))

    return simu_states_omega, simu_covs_omega


for ind_simu_omega, simu_omega in tqdm(enumerate(omegas)):
    simu_A = np.array([[-.5*gamma, simu_omega], [-simu_omega, -0.5*gamma]]).astype(np.float32)
    simu_states = [states[0].astype(np.float32)]
    simu_covs = [covs[0].astype(np.float32)]

    simu_states, simu_covs = integrate_with_omega(simu_omega, ind_simu_omega, signals, simu_A, simu_states, simu_covs)

    for indcut, cut in enumerate(cuts_final_time):
        loss[ind_simu_omega, indcut] = np.sum(np.square(signals[:cut] - np.einsum('ij,bj->bi',C,simu_states[:-1][:cut])*dt))
        loss[ind_simu_omega, indcut] =  (loss[ind_simu_omega, indcut] - times[cut])/(2*C[0,0]*dt**(3/2))  ### assuming we have homodyne, otherwise we should take inverse of C...!



# for ind_simu_omega, simu_omega in tqdm(enumerate(omegas)):
#     simu_A = np.array([[-.5*gamma, simu_omega], [-simu_omega, -0.5*gamma]]).astype(np.float32)
#     simu_states[simu_omega] = [states[0].astype(np.float32)]
#     simu_covs[simu_omega] = [covs[0].astype(np.float32)]
#
#     simu_states[simu_omega], simu_covs[simu_omega] = integrate_with_omega(simu_omega, ind_simu_omega, signals, simu_A, simu_states[simu_omega], simu_covs[simu_omega])
#
#     for indcut, cut in enumerate(cuts_final_time):
#         loss[ind_simu_omega, indcut] = np.sum(np.square(signals[:cut] - np.einsum('ij,bj->bi',C,simu_states[simu_omega][:-1][:cut])*dt))
#         loss[ind_simu_omega, indcut] =  (loss[ind_simu_omega, indcut] - times[cut])/(2*C[0,0]*dt**(3/2))  ### assuming we have homodyne, otherwise we should take inverse of C...!

path_landscape= get_path_config(periods = periods, ppp= ppp, rppp=rppp, method=method, itraj=itraj, exp_path=exp_path)+"landscape/"

os.makedirs(path_landscape,exist_ok=True)
np.save(path_landscape+"losses",loss)
np.save(path_landscape+"omegas",omegas)
np.save(path_landscape+"cuts",cuts_final_time)

# os.makedirs(path_landscape+"simu_states/",exist_ok=True)
# for k, v in simu_states.items():
#     np.save(path_landscape+"simu_states/{}".format(k), list(v))
