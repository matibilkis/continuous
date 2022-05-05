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


def IntegrationLoop(y0_hidden, y0_exp, times, dt):
    """
    dy = f(y,t) *dt + G(y,t) *dW
    """
    N = len(times)+1
    d = m = len(y0_hidden)
    _,I=Ikpw(dW,dt)

    yhidden = np.zeros((N, d))
    yexper = np.zeros((N, len(y0_exp)))

    yhidden[0] = y0_hidden
    yexper[0] = y0_exp
    dys = []

    for ind, t in enumerate(tqdm(times)):
        yhidden[ind+1] = RosslerStep(t, yhidden[ind], dW[ind,:], I[ind,:,:], dt, Fhidden, Ghidden, d, m)
        ## measurement outcome
        x1 = yhidden[ind][:2]
        dy = -np.sqrt(2)*np.dot(B1.T,x1)*dt + dW[ind,:2]
        dys.append(dy)

        yexper[ind+1] = EulerUpdate_x0_logliks(x1, dy, yexper[ind], dt)
    return yhidden, yexper, dys

def EulerUpdate_x0_logliks(x1,dy,s, dt):
    """
    this function updates the value of {x0,cov0} (wrong hypothesis) by using the dy
    also updates the log likelihoods l1 and l0
    """
    ### x1 is the hidden state i use to simulate the data
    x0 = s[:2]
    dx0 = np.dot(A0 + np.sqrt(2)*np.dot(XiCov0, B0.T), x0)*dt + np.dot(XiCov0, dy)

    l0, l1 = s[2:]
    u0 = -np.dot(B0.T,x0)*np.sqrt(2)
    u1 = -np.dot(B1.T,x1)*np.sqrt(2)
    dl0 = -dt*np.dot(u0,u0)/2 + np.dot(u0, dy)
    dl1 = -dt*np.dot(u1,u1)/2 + np.dot(u1, dy)
    return [(x0 + dx0)[0], (x0 + dx0)[1], l0 + dl0, l1+dl1 ]

@jit(nopython=True)
def Fhidden(s, t, dt):
    """
    """
    x1 = s[0:2]
    return np.dot(A1,x1)

@jit(nopython=True)
def Ghidden():
    return XiCov1

def integrate(total_time=10, dt=1e-6, itraj=1, exp_path="",**kwargs):
    """
    h1 is the hypothesis i use to get the data. (with al the coefficients gamma1...)
    """
    global A0, C0, E0, B0, A1, C1, E1, B1, CovSS0, CovSS1, XiCov1,XiCov0, dW

    params1 = [xi1, kappa1, omega1, eta1]
    params0 = [xi0, kappa0, omega0, eta0]

    # print("Hypothesis 1 (used to simulate) with params: {}\n Null hypothesis H0 has params {}\n\n".format(params1,params0))

    def give_matrices(xi, kappa, omega, eta):
        A = np.array([[-(xi + .5*kappa), omega], [-omega, xi - 0.5*kappa]])
        D = kappa*np.eye(2)
        E = B = -np.sqrt(eta*kappa)*np.array([[1.,0.],[0.,0.]])
        return A, D, E, B

    A1, D1, E1, B1 = give_matrices(*params1)#gamma1, omega1, n1, eta1, kappa1)
    A0, D0, E0, B0  = give_matrices(*params0)#gamma0, omega0, n0, eta0, kappa0)

    lin0, lin1 = 0., 0.
    x1in ,p1in, x0in, p0in, dyxin, dypin = np.zeros(6)

    ### stationary state for the covariance
    def stat(xi, kappa, omega, eta):
        vx = (kappa*(2*eta -1) - 2*xi + np.sqrt(kappa**2 - 4*xi*kappa*(2*eta -1) + 4*xi**2))/(2*eta*kappa)
        vp = kappa/(kappa - 2*xi)
        return vx, vp

    vx1, vp1  = stat(*params1)#gamma1, omega1, n1, eta1, kappa1)
    vx0, vp0  = stat(*params0)

    CovSS1 = np.diag([vx1, vp1])
    CovSS0 = np.diag([vx0, vp0])

    XiCov1 = (E1 - np.dot(CovSS1, B1))/np.sqrt(2)
    XiCov0 = (E0 - np.dot(CovSS0, B0))/np.sqrt(2)


    s0_hidden = np.array([x1in, p1in])
    s0_exper = np.array([x0in, p0in, lin0, lin1])

    times = np.arange(0,total_time+dt,dt)
    params = [params1,params0]

    #### generate long trajectory of noises
    np.random.seed(itraj)
    dW = np.sqrt(dt)*np.random.randn(len(times),2)

    yhidden, yexper, dys = IntegrationLoop(s0_hidden, s0_exper,  times, dt)
    states1 = yhidden[:,0:2]
    states0 = yexper[:,:2]
    liks = yexper[:,2:]

    path = get_path_config(total_time=total_time, dt=dt, itraj=itraj, exp_path=exp_path)
    os.makedirs(path, exist_ok=True)

    #indis = np.logspace(1,np.log10(len(times)-1), 1000)
    #indis = [int(k) for k in indis]
    #timind = [times[ind] for ind in indis]
    #logliks_short =  np.array([liks[ii] for ii in indis])
    np.save(path+"logliks",liks)#_short)
    np.save(path+"states1",np.array(states1 ))
    np.save(path+"states0",np.array(states0 ))
    np.save(path+"dys",np.array(dys ))


    #np.save(path+"times",timind)

    #
    #if save_all == 1:
    #     np.save(path+"covs1",np.array(covs1 ))

    #
        #p.save(path+"times",np.array(times ))
    #     np.save(path+"params",params)
    #     np.save(path+"dys",np.array(dys ))
    #
    #     #np.save(path+"states1",np.array(states1 ))
    #     np.save(path+"covs0",np.array(covs0 ))

    # print("traj saved in {}\n save_all {}".format(path, save_all))
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--itraj", type=int, default=2)
    parser.add_argument("--dt",type=float, default=1e-6)
    parser.add_argument("--ppp",type=float, default=1e5)

    parser.add_argument("--total_time", type=float,default=4)
    parser.add_argument("--flip_params", type=int, default=0)
    parser.add_argument("--mode", type=str, default="frequencies")
    parser.add_argument("--save_all", type=int, default=0)

    args = parser.parse_args()

    itraj = args.itraj ###this determines the seed
    total_time = args.total_time
    dt = args.dt
    ppp = args.ppp
    flip_params = args.flip_params
    save_all = args.save_all

    params = give_def_params_discrimination_squeezing(flip = flip_params)
    params, exp_path = check_params_discrimination(params)
    [xi1, kappa1, omega1, eta1], [xi0, kappa0, omega0, eta0] = params

    total_time,dt = 100*kappa1, kappa1*1e-3


    integrate(total_time = total_time,
                        dt = dt,
                        itraj=itraj,
                        exp_path = exp_path,
                        save_all = save_all,
                        eta0=eta0,
                        eta1=eta1,
                        xi1 = xi1,
                        xi0 = xi0,
                        kappa0 = kappa0,
                        kappa1 = kappa1,
                        omega0 = omega0, ####
                        omega1 = omega1
                        )


###
