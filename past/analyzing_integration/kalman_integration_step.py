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
[C, A, D , Lambda] = [C.astype(np.float32), A.astype(np.float32), D.astype(np.float32) , Lambda.astype(np.float32)]

@jit(nopython=True)
def integrate_euler( signals, simu_A, simu_states, simu_covs):
    for ind,dy in enumerate(signals):

        x = simu_states[-1].astype(np.float32)
        cov = simu_covs[-1].astype(np.float32)

        XiCov = np.dot(cov, C.T) + Lambda.T
        dx = np.dot(simu_A-np.dot(XiCov,C),x)*dt  + np.dot(XiCov,dy)
        dcov = (np.dot(simu_A,cov) + np.dot(cov, simu_A.T) + D - np.dot(XiCov, XiCov.T))*dt

        simu_states.append( (x + dx).astype(np.float32))
        simu_covs.append( (cov + dcov).astype(np.float32))

    return simu_states, simu_covs


remainder = (len(times)%euler_rppp)
if remainder > 0:
    tspan = times[:-remainder] #this is so we can split evenly
    signals_jump = signals[:-remainder]
else:
    tspan = times
    signals_jump = signals

signals_jump = np.stack([np.sum(signals_jump[k:(k+euler_rppp)], axis=0)  for k in range(int(len(signals_jump)/euler_rppp)) ])

### compute the right dt (i.e. the one used for integration, and then multiply ir by the corresponding factor)
global dt
Period = 2*np.pi/omega
dt = (Period/ppp)*euler_rppp


simu_omega = omega
simu_A = np.array([[-.5*gamma, simu_omega], [-simu_omega, -0.5*gamma]]).astype(np.float32)
simu_states = [states[0].astype(np.float32)]
simu_covs = [covs[0].astype(np.float32)]

print("Euler integrating the signals!")
simu_states, simu_covs = integrate_euler(signals_jump, simu_A, simu_states, simu_covs)


path_kalman_dt = get_path_config(periods = periods, ppp= ppp, rppp=rppp, method=method, itraj=itraj, exp_path=exp_path)+"stroboscopic_euler_rppp{}/".format(euler_rppp)

os.makedirs(path_kalman_dt,exist_ok=True)

sqrt_mse = np.sqrt(np.mean( ( np.array(simu_states) - states[::euler_rppp])**2 ))

np.save(path_kalman_dt+"sqrt_mse",[sqrt_mse])
# os.makedirs(path_kalman_dt+"states/",exist_ok=True)
# np.save(path_kalman_dt+"states/states",np.array(simu_states))
# np.save(path_kalman_dt+"states/covs",np.array(simu_covs))
