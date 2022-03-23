import os
import sys
sys.path.insert(0, os.getcwd())
from numerics.utilities.misc import *
from numerics.integration.steps import Ikpw, RosslerStep
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from datetime import datetime
import ast
from numba import jit


def Euler(ff, G, y0, times, dt):
    N = len(times)+1
    y = np.zeros((N, len(y0)))

    y[0] = y0
    for ind, t in enumerate(tqdm(times)):
        y[ind+1] = y[ind] + ff(y[ind], t)*dt #+ np.dot(G(y[ind], t), dW[ind,:])
    return y

@jit(nopython=True)
def give_ders(varx, varp, covxp, gamma, omega, kappa,n):
    return [2*covxp*omega - 4*eta*kappa*varx**2 - gamma*varx + gamma*(n + 0.5) + kappa,
     -4*covxp**2*eta*kappa - 2*covxp*omega - gamma*varp + gamma*(n + 0.5) + kappa,
     -4*covxp*eta*kappa*varx - covxp*gamma + omega*varp - omega*varx]

def RosslerSRI2(f, G, y0, times, dt):
    """
    dy = f(y,t) *dt + G(y,t) *dW
    borrowed from sdeint - python
    """
    N = len(times)+1
    d = len(y0)
    m = len(y0)

    _,I=Ikpw(dW,dt)

    y = np.zeros((N, d))
    y[0] = y0
    Gn = np.zeros((d, m), dtype=y.dtype)

    for ind, t in enumerate(tqdm(times)):
        y[ind+1] = RosslerStep(t,y[ind], dW[ind,:], I[ind,:,:], dt, f,G, d, m)
    return y

@jit(nopython=True)
def Fs(s,t, coeffs=None, params=None, dt=None):
    """
    """
    vx, vp,cvxp = s[:3]
    varx_dot, varp_dot, covxy_dot = give_ders(vx, vp, cvxp, gamma, omega, kappa, n)
    return np.array([varx_dot, varp_dot, covxy_dot])

@jit(nopython=True)
def Gs(s,t, coeffs=None, params=None):
    #cov = s_to_cov(s)
    wieners = np.zeros((s.shape[0], s.shape[0]))
    return wieners


def integrate(periods, ppp, method="rossler", itraj=1, exp_path="",**kwargs):

    """
    everything is in the get_def_path()

    note that if you sweep params exp_path needs to specified
    """
    global A, A1, D, D1,  gamma, gamma1, omega, omega1, C,  Lambda, eta, n, n1, kappa, dW, proj_C

    n = kwargs.get("n",2.0)
    n1 = kwargs.get("n1",20.0)

    eta = kwargs.get("eta",1) #efficiency
    kappa = kwargs.get("kappa",1)

    gamma = kwargs.get("gamma",0.3)
    gamma1 = kwargs.get("gamma1",1.0)

    omega = kwargs.get("omega",2*np.pi)
    omega1 = kwargs.get("omega1",4*np.pi)

    gamma, gamma1, omega, omega1, n, n1, eta, kappa = give_def_params_discrimination()

    print("discriminating with eta {}, kappa {}, n {} ".format(eta, kappa,n))
    print("omega {} and  omega1 {} ".format(omega, omega1))
    print("gamma {} and  gamma1 {} ".format(gamma, gamma1))

    #print("integrating with parameters: \n eta {} \nkappa {}\ngamma {} \nomega {}\nn {}\n".format(eta,kappa,gamma,omega,n))

    l0, l10 = 0.,0.
    x0, p0, yx0, yp0 = 0., 0., 0.,0.
    x10, p10 = 0., 0.

    suc = n + 0.5 + kappa/gamma
    sst = (gamma/(8*eta*kappa))*(np.sqrt(1 + 16*eta*kappa*suc/gamma ) -1 )

    suc1 = n1 + 0.5 + kappa/gamma1
    sst1 = (gamma1/(8*eta*kappa))*(np.sqrt(1 + 16*eta*kappa*suc1/gamma1 ) -1 )

    varx0, varp0, covxy0 = sst ,suc ,0.
    varx10, varp10, covxy10 = sst1 ,suc1 ,0.

    #s0 = np.array([x0, p0, yx0, yp0, varx0, varp0, covxy0, x10, p10, varx10 , varp10 , covxy10, l0, l10])

    s0 = np.array([varx0, varp0, covxy0])
    C = np.array([[np.sqrt(4*eta*kappa),0],[0,0]]) #homodyne
    proj_C = C/np.sum(C)#np.linalg.pinv(C) this is not because i don't multiply by C
    Lambda = np.zeros((2,2))

    A = np.array([[-gamma/2, omega],[-omega, -gamma/2]])
    A1 = np.array([[-gamma1/2, omega1],[-omega1, -gamma1/2]])
    D = np.diag([gamma*(n+0.5) + kappa]*2)
    D1 = np.diag([gamma1*(n1+0.5) + kappa]*2)


    # Period = 2*np.pi/omega
    # dt = Period/ppp
    # times = np.arange(0.,Period*periods,dt)
    dt = 1/ppp
    times = np.arange(0,periods+dt,dt)
    params = [gamma, gamma1, omega, omega1, eta, kappa, n, n1]

    #### generate long trajectory of noises
    np.random.seed(itraj)
    dW = np.zeros((len(times), 3)) #I have ppp*periods points.

    # solution = RosslerSRI2(Fs, Gs, s0, times, dt)
    solution = Euler(Fs,Gs, s0, times, dt)
    #states, signals, covs, states1, covs1, l0, l1 = convert_solution_discrimination(solution)
    path = get_path_config(periods = periods, ppp= ppp, method=method, itraj=itraj, exp_path=exp_path)

    os.makedirs(path, exist_ok=True)
    np.save(path+"solution", np.array(solution))
    np.save(path+"A", np.array(A))
    np.save(path+"C", np.array(C))
    np.save(path+"D", np.array(D))

    # np.save(path+"noises",np.array(dW))
    # np.save(path+"times",np.array(times ))
    # np.save(path+"states",np.array(states ))
    # np.save(path+"covs",np.array(covs ))
    # np.save(path+"signals",np.array(signals ))
    # np.save(path+"params",params)
    # np.save(path+"states1",np.array(states1 ))
    # np.save(path+"covs1",np.array(covs1 ))
    # np.save(path+"loglik0",np.array(l0 ))
    # np.save(path+"loglik1",np.array(l1 ))
    print("traj saved in", path)
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--itraj", type=int, default=1)
    parser.add_argument("--periods",type=int, default=50)
    parser.add_argument("--ppp", type=int,default=100)
    parser.add_argument("--method", type=str,default="rossler")
    parser.add_argument("--rppp", type=int,default=1)
    parser.add_argument("--params", type=str, default="") #[gamma, gamma1, omega, omega1, eta, kappa, n]
    args = parser.parse_args()

    itraj = args.itraj ###this determines the seed
    periods = args.periods
    ppp = args.ppp
    method = args.method
    rppp = args.rppp
    params = args.params

    params, exp_path = check_params_discrimination(params)
    gamma, gamma1, omega, omega1, n, n1, eta, kappa = params

    integrate(periods, ppp, method=method, itraj=itraj, exp_path = exp_path ,rppp = rppp,
                eta=eta, kappa = kappa,  gamma = gamma, n = n, omega = omega, gamma1 = gamma1, omega1 = omega1, n1=n1)
