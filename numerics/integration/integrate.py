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


def IntegrationLoop(y0_hidden, y0_covhidden, y0_exp, times, dt):
    """
    dy = f(y,t) *dt + G(y,t) *dW
    """
    N = len(times)+1
    d = len(y0_hidden)
    m = len(y0_hidden)
    _,I=Ikpw(dW,dt)

    shidden = np.zeros((N, d))

    shidden[0] = y0_hidden
    dys = []

    for ind, t in enumerate(times):
        shidden[ind+1] = RosslerStep(t, shidden[ind], dW[ind,:], I[ind,:,:], dt, Fhidden, Ghidden, d, m, XiCov)

        ## measurement outcome
        dy = np.dot(C1,shidden[ind][:2])*dt + dW[ind,:2]
        dys.append(dy)

    return yhidden, ycovhidden, yexper, dys

@jit(nopython=True)
def Fhidden(s, t, dt):
    """
    """
    x1 = s[:2]
    x1_dot = np.dot(A1,x1)
    return x1_dot


def integrate(total_time=10, dt=1e-6, itraj=1, exp_path="",**kwargs):
    """
    h1 is the hypothesis i use to get the data. (with al the coefficients gamma1...)
    """
    global A, C, D, XiCov, gamma, eta, n, kappa, dW

    params = [gamma, omega, n, eta, kappa]

    def give_matrices(gamma, omega, n, eta, kappa):
        A = np.array([[-gamma/2, omega],[-omega, -gamma/2]])
        C = np.sqrt(4*eta*kappa)*np.array([[1.,0.],[0.,1.]]) #homodyne
        D = np.diag([gamma*(n+0.5) + kappa]*2)
        return A, C, D

    A,C,D = give_matrices(gamma, omega, n, eta, kappa)
    def stat(gamma, omega, n, eta, kappa):
        suc = n + 0.5 + kappa/gamma
        sst = (gamma/(8*eta*kappa))*(np.sqrt(1 + 16*eta*kappa*suc/gamma ) -1 )
        return suc, sst

    S_uc, S_st = stat(gamma, omega, n , eta, kappa)

    XiCov = np.dot(np.eye(2)*S_st, C.T)
    times = np.arange(0,total_time+dt,dt)


    #### generate long trajectory of noises
    np.random.seed(itraj)
    dW = np.sqrt(dt)*np.random.randn(len(times),2)


    yhidden, dys = IntegrationLoop(s0_hidden, s0cov_hidden, s0_exper,  times, dt)

    path = get_path_config(total_time=total_time, dt=dt, itraj=itraj, exp_path=exp_path)

    os.makedirs(path, exist_ok=True)


    #indis = np.logspace(1,np.log10(len(times)-1), 1000)
    #indis = [int(k) for k in indis]
    #timind = [times[ind] for ind in indis]
    #logliks_short =  np.array([liks[ii] for ii in indis])
    np.save(path+"logliks",liks)#_short)
    #np.save(path+"times",timind)

    #
    #if save_all == 1:
    #     np.save(path+"covs1",np.array(covs1 ))

    #
        #p.save(path+"times",np.array(times ))
    #     np.save(path+"params",params)
    #     np.save(path+"states1",np.array(states1 ))
    #     np.save(path+"dys",np.array(dys ))
    #
    #     #np.save(path+"states1",np.array(states1 ))
    #     np.save(path+"states0",np.array(states0 ))
    #     np.save(path+"covs0",np.array(covs0 ))

    # print("traj saved in {}\n save_all {}".format(path, save_all))
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--itraj", type=int, default=1)
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

    params = give_def_params_discrimination(flip = flip_params, mode=args.mode)
    params, exp_path = check_params_discrimination(params)
    [gamma1, omega1, n1, eta1, kappa1], [gamma0, omega0, n0, eta0, kappa0] = params

    total_time, dt = get_total_time_dt(params, dt=dt, total_time=total_time, ppp=ppp)

    integrate(total_time = total_time,
                        dt = dt,
                        itraj=itraj,
                        exp_path = exp_path,
                        save_all = save_all,
                        eta0=eta0,
                        kappa0 = kappa0,
                        gamma0 = gamma0,
                        n0 = n0,
                        omega0 = omega0, ####
                        eta1=eta1,
                        kappa1 = kappa1,
                        gamma1 = gamma1,
                        n1 = n1,
                        omega1 = omega1
                        )


###
