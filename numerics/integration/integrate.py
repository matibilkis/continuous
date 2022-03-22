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

@jit(nopython=True)
def give_ders(varx, varp, covxp, gamma, omega, kappa,n):
    return [2*covxp*omega - 2*eta*kappa*varx**2 - gamma*varx + gamma*(n + 0.5),
                -2*covxp*eta*kappa*varx - covxp*gamma + omega*varp - omega*varx,
                 -2*covxp**2*eta*kappa - 2*covxp*omega - gamma*varp + gamma*(n + 0.5)]

def RosslerSRI2(f, G, y0, times, dt):
    """
    dy = f(y,t) *dt + G(y,t) *dW
    borrowed from sdeint - python
    """
    N = len(times)+1
    d = len(y0)
    m = len(y0)

    _,I=Ikpw(noises,dt)

    y = np.zeros((N, d))
    y[0] = y0
    Gn = np.zeros((d, m), dtype=y.dtype)

    for ind, t in enumerate(tqdm(times)):
        y[ind+1] = RosslerStep(t,y[ind], noises[ind,:], I[ind,:,:], dt, f,G, d, m)
    return y

@jit(nopython=True)
def Fs(s,t, coeffs=None, params=None, dt=None):
    """
    """
    x = s[0:2]
    xdot = np.dot(A,x)

    y = s[2:4]
    ydot = np.dot(C,x)

    varx, varp,covxp = s[4:7]
    varx_dot, varp_dot, covxy_dot = give_ders(varx, varp, covxp, gamma, omega, kappa, n)

    # [2*covxp*omega - 2*eta*kappa*varx**2 - gamma*varx + gamma*(n + 0.5),
    #             -2*covxp*eta*kappa*varx - covxp*gamma + omega*varp - omega*varx,
    #              -2*covxp**2*eta*kappa - 2*covxp*omega - gamma*varp + gamma*(n + 0.5)]

###########################3
    x1 = s[7:9]
    x1dot = np.dot(A1,x1)
    varx1, varp1,covxp1 = s[9:12]

    varx1_dot, varp1_dot, covxy1_dot = give_ders(varx1, varp1, covxp1, gamma1, omega1, kappa, n)

#######################3
    l = s[12]
    v1 = np.dot(C,x)
    l_dot = - np.dot(v1,v1)/2 + np.dot(np.dot(C,x), y)

    l1 = s[13]
    v2 = np.dot(C,x1)
    l1_dot = - np.dot(v2,v2)/2 + np.dot(np.dot(C,x1), y)

    return np.array([xdot[0], xdot[1], ydot[0],  ydot[1], varx_dot, varp_dot, covxy_dot, x1dot[0], x1dot[1], varx1_dot, varp1_dot, covxy1_dot, l_dot, l1_dot])

@jit(nopython=True)
def Gs(s,t, coeffs=None, params=None):
    #cov = s_to_cov(s)
    varx, varp,covxy = s[4:7]
    varx1, varp1,covxy1 = s[9:12]

    cov = np.array([[varx, covxy], [covxy, varp]])
    XiCov = np.dot(cov, C.T) + Lambda.T

    cov1 = np.array([[varx1, covxy1], [covxy1, varp1]])
    XiCov1 = np.dot(cov1, C.T) + Lambda.T


    wieners = np.zeros((s.shape[0], s.shape[0]))
    wieners[:2,:2]  = XiCov
    wieners[2:4,2:4] = proj_C
    wieners[7:9,7:9] = np.dot(C, XiCov1)
    return wieners


def integrate(periods, ppp, method="rossler", itraj=1, exp_path="",**kwargs):

    """
    everything is in the get_def_path()

    note that if you sweep params exp_path needs to specified
    """
    global A, A1, D, D1,  gamma, gamma1, omega, omega1, C,  Lambda, eta, n, kappa, noises, proj_C

    n = kwargs.get("n",2.0)

    eta = kwargs.get("eta",1) #efficiency
    kappa = kwargs.get("kappa",1)

    gamma = kwargs.get("gamma",0.3)
    gamma1 = kwargs.get("gamma1",1.0)

    omega = kwargs.get("omega",2*np.pi)
    omega1 = kwargs.get("omega1",4*np.pi)

    print("discriminating with eta {}, kappa {}, n {} ".format(eta, kappa,n))
    print("omega {} and  omega1 {} ".format(omega, omega1))
    print("gamma {} and  gamma1 {} ".format(gamma, gamma1))

    #print("integrating with parameters: \n eta {} \nkappa {}\ngamma {} \nomega {}\nn {}\n".format(eta,kappa,gamma,omega,n))

    l0, l10 = 0.,0.
    x0, p0, yx0, yp0 = 1., 0., 0.,0.
    x10, p10 = 1., 0.
    varx0, varp0, covxy0 = .1 ,.1 ,1e-3
    varx10, varp10, covxy10 = .1 ,.1 ,1e-3

    s0 = np.array([x0, p0, yx0, yp0, varx0, varp0, covxy0, x10, p10, varx10 , varp10 , covxy10, l0, l10])

    C = np.array([[np.sqrt(2*eta*kappa),0],[0,0]]) #homodyne
    proj_C = C/np.sum(C)
    Lambda = np.zeros((2,2))

    A = np.array([[-gamma/2, omega],[-omega, -gamma/2]])
    A1 = np.array([[-gamma1/2, omega1],[-omega1, -gamma1/2]])
    D = np.diag([(gamma*(n+0.5))]*2)
    D1 = np.diag([(gamma1*(n+0.5))]*2)


    # Period = 2*np.pi/omega
    # dt = Period/ppp
    # times = np.arange(0.,Period*periods,dt)
    dt = 1/ppp
    times = np.linspace(0,10, ppp)
    params = [gamma, gamma1, omega, omega1, eta, kappa, n]

    #### generate long trajectory of noises
    np.random.seed(itraj)
    dW = np.zeros((len(times), 14)) #I have ppp*periods points.
    for ind,t in enumerate(times):
        w0 = np.random.normal()*np.sqrt(dt)
        w1 = np.random.normal()*np.sqrt(dt)
        dW[ind,:] = np.array([w0, w1, w0, w1 , 0.,0.,0., w0, w1, 0., 0., 0., 0. , 0.])  ## x0, x1,  y0, y1, varx, covxp, varp, u_th0, u_th1, var_uth0, covuth, varputh

    noises = np.zeros((len(times),dW.shape[1]))

    solution = RosslerSRI2(Fs, Gs, s0, times, dt)

    states, signals, covs, states1, covs1, l0, l1 = convert_solution_discrimination(solution)
    path = get_path_config(periods = periods, ppp= ppp, method=method, itraj=itraj, exp_path=exp_path)

    os.makedirs(path, exist_ok=True)
    np.save(path+"times",np.array(times ))
    np.save(path+"states",np.array(states ))
    np.save(path+"covs",np.array(covs ))
    np.save(path+"signals",np.array(signals ))
    np.save(path+"params",params)
    np.save(path+"states1",np.array(states1 ))
    np.save(path+"covs1",np.array(covs1 ))
    np.save(path+"loglik0",np.array(l0 ))
    np.save(path+"loglik1",np.array(l1 ))
    print("traj saved in", path)
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--itraj", type=int, default=1)
    parser.add_argument("--periods",type=int, default=5)
    parser.add_argument("--ppp", type=int,default=1000)
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
    [gamma, gamma1, omega, omega1, eta, kappa, n]= params

    integrate(periods, ppp, method=method, itraj=itraj, exp_path = exp_path ,rppp = rppp,
                eta=eta, kappa = kappa,  gamma = gamma, n = n, omega = omega, gamma1 = gamma1, omega1 = omega1)
