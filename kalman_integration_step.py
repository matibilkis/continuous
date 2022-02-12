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
parser.add_argument("--ppp", type=int) ###points per period
parser.add_argument("--periods", type=int)
parser.add_argument("--itraj", type=int, default=1)
parser.add_argument("--rppp", type=int, default=1)
parser.add_argument("--method", type=str, default="rossler")
parser.add_argument("--euler_rppp", type=int, default=1)
parser.add_argument("--params", type=str, default="") #[eta, gamma, kappa, omega, n]



args = parser.parse_args()

periods = args.periods
ppp = args.ppp
itraj = args.itraj
rppp = args.rppp
method = args.method
euler_rppp = args.euler_rppp
params = args.params

params, exp_path = check_params(params)


states, covs, signals, params, times = load_data(ppp=ppp, periods=periods, method=method, itraj=itraj, exp_path = exp_path)
[eta, gamma, kappa, omega, n] = params
[C, A, D , Lambda] = build_matrix_from_params(params)

xi = lambda cov,Lambda: np.dot(cov, C.T) + Lambda.T

def evolve_simu_state(x,cov, dy, simu_A, internal_step):
    XiCov = xi(cov, Lambda)
    dx = np.dot(simu_A-np.dot(XiCov,C),x)*internal_step  + np.dot(XiCov,dy)
    dcov = (np.dot(simu_A,cov) + np.dot(cov, ct(simu_A)) + D - np.dot(XiCov, XiCov.T))*internal_step
    return [x + dx, cov + dcov]

simu_states, simu_covs = {}, {}

omegas = list(set([omega] + list(np.linspace(0, 2*omega, 10))))

remainder = (len(times)%euler_rppp)
if remainder > 0:
    tspan = times[:-remainder] #this is so we can split evenly
    signals_jump = signals[:-remainder]
else:
    tspan = times
    signals_jump = signals

signals_jump = np.stack([np.sum(signals_jump[k:(k+euler_rppp)], axis=0)  for k in range(int(len(signals_jump)/euler_rppp)) ])

Period = 2*np.pi/omega
dt = (Period/ppp)*euler_rppp

for ind_simu_omega, simu_omega in tqdm(enumerate(omegas)):
    simu_A = np.array([[-.5*gamma, simu_omega], [-simu_omega, -0.5*gamma]])
    simu_states[simu_omega] = [states[0]]
    simu_covs[simu_omega] = [covs[0]]

    for ind,dy in enumerate(tqdm(signals_jump)):
        simu = evolve_simu_state(simu_states[simu_omega][-1], simu_covs[simu_omega][-1], dy, simu_A,  dt)
        simu_states[simu_omega].append(simu[0])
        simu_covs[simu_omega].append(simu[1])

path_kalman_dt = get_path_config(periods = periods, ppp= ppp, rppp=rppp, method=method, itraj=itraj, exp_path=exp_path)+"stroboscopic_euler_rppp{}/".format(euler_rppp)

os.makedirs(path_kalman_dt,exist_ok=True)
os.makedirs(path_kalman_dt+"states/",exist_ok=True)

for ind_simu_omega, simu_omega in enumerate(omegas):
    np.save(path_kalman_dt+"states/states{}".format(ind_simu_omega),np.array(simu_states[simu_omega]))
    np.save(path_kalman_dt+"states/covs{}".format(ind_simu_omega),np.array(simu_covs[simu_omega]))

np.save(path_kalman_dt+"omegas",omegas)
