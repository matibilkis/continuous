import os
import sys
sys.path.insert(0, os.getcwd())
from numerics.utilities.misc import *
from numerics.integration.steps import Ikpw, RosslerStep
from numerics.integration.matrices import *
import numpy as np
from tqdm import tqdm
import argparse
import ast
from numba import jit


##### Notation#
###   C = -Sqrt[2] B^T
###   \Gamma =  - E^T/Sqrt[2]
###   Xi(\Cov) = Cov C^T - \Gamma^T = (- Cov B + E)/Sqrt[2]
##   Carefull that D changes also, so the ricatti equation is slightly different (factors \sqrt{2})

def IntegrationLoop(S_hidden_in, times, dt):
    """
    dy = f(y,t) *dt + G(y,t) *dW
    """
    ## robbler method
    N = len(times)
    d = len(S_hidden_in)
    m = len(S_hidden_in)
    _,I=Ikpw(dW,dt)

    S_hidden = np.zeros((N, d))
    S_hidden[0] = S_hidden_in
    dys = []

    for ind, t in enumerate(tqdm(times[:-1])):
        S_hidden[ind+1] = RosslerStep(t, S_hidden[ind], dW[ind], I[ind,:,:], dt, Fhidden, Ghidden, d, m) #update hidden state (w/ Robler method)
        dy = -np.sqrt(2)*np.dot(B.T,S_hidden[ind])*dt + proj_C.dot(dW[ind]) ## measurement outcome, pinv in case you homodyne
        dys.append(dy)
    return S_hidden, dys

@jit(nopython=True)
def Fhidden(s, t, dt):
    """
    """
    return np.dot(A,s) + ext_fun(Ext_signal_params, t)*np.array([1.,0.]).astype(np.float32)

@jit(nopython=True)
def Ghidden():
    return XiCov

def integrate(params=[], total_time=10, dt=1e-6, itraj=1, ext_signal=1, exp_path="",**kwargs):
    """
    h1 is the hypothesis i use to get the data. (with al the coefficients gamma1...)
    """
    global  A, C, E, B, CovSS, XiCov, dW, Ext_signal_params, proj_C

    [xi, kappa, omega, eta] = params
    Np = 100
    if ext_signal == 1:
        Ext_signal_params = np.array([1e1,2*np.pi/(100/Np)])
    else:
        Ext_signal_params = np.array([0.,0.])


    ### stationary state for the covariance

    A, D, E, B = genoni_matrices(*params)
    proj_C = np.linalg.pinv(B/np.sum(B))
    XiCov = genoni_xi_cov(A,D,E,B, params)

    times = np.arange(0,total_time+dt,dt)

    #### generate trajectory of noises
    np.random.seed(itraj)
    dW = np.sqrt(dt)*np.random.randn(len(times),2)
    s0_hidden = np.array([0.,0.])

    Xs, dys = IntegrationLoop(s0_hidden,  times, dt)
    path = get_path_config(total_time=total_time, dt=dt, itraj=itraj, exp_path=exp_path, ext_signal=ext_signal)

    os.makedirs(path, exist_ok=True)
    np.save(path+"dys",np.array(dys ))
    np.save(path+"states",np.array(Xs))
    #np.save(path+"times",timind)
    #if save_all == 1:
    #     np.save(path+"covs1",np.array(covs1 ))
    #     np.save(path+"params",params)
    #     np.save(path+"states1",np.array(states1 ))
    #     #np.save(path+"states1",np.array(states1 ))
    #     np.save(path+"states0",np.array(states0 ))
    #     np.save(path+"covs0",np.array(covs0 ))
    # print("traj saved in {}\n save_all {}".format(path, save_all))
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--itraj", type=int, default=1)
    parser.add_argument("--dt",type=float, default=1e-3)
    parser.add_argument("--total_time", type=float,default=4)
    parser.add_argument("--ext_signal", type=int, default=1)
    args = parser.parse_args()

    itraj = args.itraj ###this determines the seed
    total_time = args.total_time
    dt = args.dt
    ext_signal = args.ext_signal

    params, exp_path = def_params()
    xi, kappa, omega, eta = params

    total_time,dt = total_time*kappa, kappa*dt

    global ext_fun
    ext_fun = external_function(mode="np")

    integrate(total_time = total_time,
                dt = dt,
                itraj=itraj,
                exp_path = exp_path,
                params = params,
                ext_signal=ext_signal)
