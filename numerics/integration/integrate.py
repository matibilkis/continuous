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
        y[ind+1] = y[ind] + ff(y[ind], t)*dt + np.dot(G(y[ind], t), dW[ind,:])
    return y


@jit(nopython=True)
def give_ders(vx, vp, cvxp, gamma_val, omega_val, kappa_val,n_val, eta_val):
    return [2*cvxp*omega_val - 4*eta_val*kappa_val*vx**2 - gamma_val*vx + gamma_val*(n_val + 0.5) + kappa_val,
     -4*cvxp**2*eta_val*kappa_val - 2*cvxp*omega_val - gamma_val*vp + gamma_val*(n_val + 0.5) + kappa_val,
     -4*cvxp*eta_val*kappa_val*vx - cvxp*gamma_val + omega_val*vp - omega_val*vx]

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
    x = s[0:2]
    xdot = np.dot(A0,x)
    y = s[2:4]
    ydot = np.dot(C0,x)
    vx, vp,cvxp = s[4:7]
    varx_dot, varp_dot, covxy_dot = give_ders(vx, vp, cvxp, gamma0, omega0, kappa0, n0, eta0)

###########################3
    x1 = s[7:9]
    x1dot = np.dot(A1,x1)
    varx1, varp1,covxp1 = s[9:12]
    varx1_dot, varp1_dot, covxy1_dot = give_ders(varx1, varp1, covxp1, gamma1, omega1, kappa1, n1, eta1)
#######################3
    l = s[12]
    v1 = np.dot(C0,x)
    l_dot =  np.dot(v1,v1)/2 #+ np.dot(np.dot(C,x), y)

    l1 = s[13]
    v2 = np.dot(C1,x1)
    l1_dot = -np.dot(v2,v2)/2 + np.dot(np.dot(C1,x1), np.dot(C0,x))

    return np.array([xdot[0], xdot[1], ydot[0],  ydot[1], varx_dot, varp_dot, covxy_dot, x1dot[0], x1dot[1], varx1_dot, varp1_dot, covxy1_dot, l_dot, l1_dot])


@jit(nopython=True)
def Gs(s,t, coeffs=None, params=None):
    varx, varp,covxy = s[4:7]
    varx1, varp1,covxy1 = s[9:12]

    cov0 = np.array([[varx, covxy], [covxy, varp]])
    XiCov0 = np.dot(cov0, C0.T)

    cov1 = np.array([[varx1, covxy1], [covxy1, varp1]])
    XiCov1 = np.dot(cov1, C1.T)

    wieners = np.zeros((s.shape[0], s.shape[0]))
    wieners[:2,:2]  = XiCov0
    wieners[2:4,2:4] = proj_C
    wieners[7:9,7:9] = XiCov1

    wieners[12] = np.dot(C0, s[:2])[0] ###this will only work for homodyning
    wieners[13] = np.dot(C1, s[:2])[0]###this will only work for homodyning

    return wieners

def integrate(periods, ppp, method="rossler", itraj=1, exp_path="",**kwargs):

    """
    h0 is the hypothesis i use to get the data.
    """
    global A0, A1, D0, D1, C0, C1,proj_C, gamma0, gamma1, omega0, omega1 , eta0, eta1, n0, n1, kappa0, kappa1, dW
    # n0 = kwargs.get("n0",2.0)
    # n1 = kwargs.get("n1",20.0)
    #
    # eta0 = kwargs.get("eta0",1) #efficiency
    # eta1 = kwargs.get("eta1",1) #efficiency
    #
    # kappa0 = kwargs.get("kappa0",1)
    # kappa1 = kwargs.get("kappa1",1)
    #
    # gamma0 = kwargs.get("gamma0",0.3)
    # gamma1 = kwargs.get("gamma1",1.0)
    #
    # omega0 = kwargs.get("omega0",2*np.pi)
    # omega1 = kwargs.get("omega1",4*np.pi)
    params0 = [gamma0, omega0, n0, eta0, kappa0]
    params1 = [gamma1, omega1, n1, eta1, kappa1]
    # [gamma1, omega1, n1, eta1, kappa1], [gamma0, omega0, n0, eta0, kappa0] = give_def_params_discrimination()
    print("params0: {}\nparams1 {}".format(params0, params1))
    #print("integrating with parameters: \n eta {} \nkappa {}\ngamma {} \nomega {}\nn {}\n".format(eta,kappa,gamma,omega,n))


    def give_matrices(gamma, omega, n, eta, kappa):
        A = np.array([[-gamma/2, omega],[-omega, -gamma/2]])
        C = np.array([[np.sqrt(4*eta*kappa),0],[0,0]]) #homodyne
        D = np.diag([gamma*(n+0.5) + kappa]*2)
        return A, C, D

    A0, C0, D0 = give_matrices(gamma0, omega0, n0, eta0, kappa0)
    A1, C1, D1 = give_matrices(gamma1, omega1, n1, eta1, kappa1)
    proj_C = C0/np.sum(C0)#np.linalg.pinv(C) this is not because i don't multiply by C


    l0, l10 = 1.,1.
    x0, p0, yx0, yp0 = 0., 0., 0.,0.
    x10, p10 = 0., 0.

    def stat(gamma, omega, n, eta, kappa):
        suc = n + 0.5 + kappa/gamma
        sst = (gamma/(8*eta*kappa))*(np.sqrt(1 + 16*eta*kappa*suc/gamma ) -1 )
        return suc, sst

    suc0, sst0 = stat(gamma0, omega0, n0, eta0, kappa0)
    suc1, sst1 = stat(gamma1, omega1, n1, eta1, kappa1)

    varx0, varp0, covxy0 = sst0 ,suc0 ,0.
    varx10, varp10, covxy10 = sst1 ,suc1 ,0.
    s0 = np.array([x0, p0, yx0, yp0, varx0, varp0, covxy0, x10, p10, varx10 , varp10 , covxy10, l0, l10])

    dt = 1/ppp
    times = np.arange(0,periods+dt,dt)
    params = [params0,params1]

    #### generate long trajectory of noises
    np.random.seed(itraj)
    dW = np.zeros((len(times), 14)) #I have ppp*periods points.
    for ind,t in enumerate(times):
        w0 = np.random.normal()*np.sqrt(dt)
        w1 = np.random.normal()*np.sqrt(dt)
        dW[ind,:] = np.array([w0, w1, w0, w1 , 0.,0.,0., w0, w1, 0., 0., 0., w0 , w0])  ## x0, x1,  y0, y1, varx, covxp, varp, u_th0, u_th1, var_uth0, covuth, varputh

    if method.lower() == "euler":
        print("euler")
        solution = Euler(Fs, Gs, s0, times, dt)
    elif method.lower() == "rossler":
        print("rossler")
        solution = RosslerSRI2(Fs, Gs, s0, times, dt)
    else:
        raise NameError("asasda")
    states0, signals0, covs0, states1, covs1, l0, l1 = convert_solution_discrimination(solution)
    path = get_path_config(periods = periods, ppp= ppp, method=method, itraj=itraj, exp_path=exp_path)

    os.makedirs(path, exist_ok=True)
    np.save(path+"times",np.array(times ))
    np.save(path+"states0",np.array(states0 ))
    np.save(path+"covs0",np.array(covs0 ))
    np.save(path+"signals0",np.array(signals0 ))
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
    parser.add_argument("--periods",type=int, default=2)
    parser.add_argument("--ppp", type=int,default=100000)
    parser.add_argument("--method", type=str,default="rossler")
    parser.add_argument("--rppp", type=int,default=1)
    parser.add_argument("--h1true", type=int, default=0)
    args = parser.parse_args()

    itraj = args.itraj ###this determines the seed
    periods = args.periods
    ppp = args.ppp
    method = args.method
    rppp = args.rppp
    h1 = args.h1true

    params = give_def_params_discrimination(h1true = h1)
    print(params)
    params, exp_path = check_params_discrimination(params)
    [gamma0, omega0, n0, eta0, kappa0], [gamma1, omega1, n1, eta1, kappa1] = params

    integrate(periods, ppp, method=method, itraj=itraj, exp_path = exp_path ,rppp = rppp,
                        eta0=eta0,
                        kappa0 = kappa0,
                        gamma0 = gamma0,
                        n0 = n0,
                        omega0 = omega0, ####
                        eta1=eta1,
                        kappa1 = kappa1,
                        gamma1 = gamma1,
                        n1 = n1,
                        omega1 = omega1,
                        )


###
