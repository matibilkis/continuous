import os
import sys
sys.path.insert(0, os.getcwd())
from numerics.utilities.misc import *
from numerics.integration.steps import Ikpw, RosslerStep
import numpy as np
from tqdm import tqdm
import argparse
import ast
from numba import jit


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
        dy = -np.sqrt(2)*np.dot(B.T,S_hidden[ind])*dt + dW[ind] ## measurement outcome
        dys.append(dy)
    print(len(dys), len(S_hidden))
    return S_hidden, dys

@jit(nopython=True)
def Fhidden(s, t, dt):
    """
    """
    return np.dot(A,s)

@jit(nopython=True)
def Ghidden():
    return XiCov

def integrate(params=[], total_time=10, dt=1e-6, itraj=1, exp_path="",**kwargs):
    """
    h1 is the hypothesis i use to get the data. (with al the coefficients gamma1...)
    """
    global  A, C, E, B, CovSS, XiCov, dW

    [xi, kappa, omega, eta] = params

    def give_matrices(xi, kappa, omega, eta):
        A = np.array([[-(xi + .5*kappa), omega], [-omega, xi - 0.5*kappa]])
        D = kappa*np.eye(2)
        E = B = -np.sqrt(eta*kappa)*np.array([[1.,0.],[0.,0.]])
        return A, D, E, B


    ### stationary state for the covariance
    def stat(xi, kappa, omega, eta):
        vx = (kappa*(2*eta -1) - 2*xi + np.sqrt(kappa**2 - 4*xi*kappa*(2*eta -1) + 4*xi**2))/(2*eta*kappa)
        vp = kappa/(kappa - 2*xi)
        return vx, vp

    A, D, E, B = give_matrices(*params)
    vx, vp = stat(*params)

    CovSS = np.diag([vx, vp])
    XiCov = (E - np.dot(CovSS, B))/np.sqrt(2)

    times = np.arange(0,total_time+dt,dt)

    #### generate trajectory of noises
    np.random.seed(itraj)
    dW = np.sqrt(dt)*np.random.randn(len(times),2)
    s0_hidden = np.array([0.,0.])

    Xs, dys = IntegrationLoop(s0_hidden,  times, dt)
    path = get_path_config(total_time=total_time, dt=dt, itraj=itraj, exp_path=exp_path)

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

    args = parser.parse_args()

    itraj = args.itraj ###this determines the seed
    total_time = args.total_time
    dt = args.dt

    params, exp_path = def_params()
    xi, kappa, omega, eta = params

    total_time,dt = total_time*kappa, kappa*dt

    integrate(total_time = total_time,
                dt = dt,
                itraj=itraj,
                exp_path = exp_path,
                params = params)
