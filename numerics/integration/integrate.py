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

def IntegrationLoop(y0_exp, y0_hidden_in, times, dt):
    """
    dy = f(y,t) *dt + G(y,t) *dW
    """
    ## robbler method
    N = len(times)
    d = len(y0_hidden_in)
    m = len(y0_hidden_in)
    _,I=Ikpw(dW,dt)

    yhidden = np.zeros((N+1, d))
    yexper = np.zeros((N+1, len(y0_exp)))

    yhidden[0] = y0_hidden_in
    yexper[0] = y0_exp
    dys = []


    for ind, t in enumerate(times):
        yhidden[ind+1] = RosslerStep(t, yhidden[ind], dW[ind,:], I[ind,:,:], dt, Fhidden, Ghidden, d, m)

        x1 = yhidden[ind+1][:2]
        dy = -np.sqrt(2)*np.dot(B1,x1)*dt + np.dot(proj_C, dW[ind,:2])
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

    dx0 = np.dot(A0 - XiCov0, x0)*dt + np.dot(XiCov0, dy)

    l0, l1 = s[2:]
    u0 = -np.sqrt(2)*np.dot(B0.T,x0)
    u1 = -np.sqrt(2)*np.dot(B1.T,x1)
    dl0 = -dt*np.dot(u0,u0)/2 + np.dot(u0, dy)
    dl1 = -dt*np.dot(u1,u1)/2 + np.dot(u1, dy)
    return [(x0 + dx0)[0], (x0 + dx0)[1],  l0 + dl0, l1+dl1 ]

@jit(nopython=True)
def Fhidden(s, t, dt):
    """
    """
    x1 = s[0:2]
    x1_dot = np.dot(A1,x1)
    return np.array([x1_dot[0], x1_dot[1]])

@jit(nopython=True)
def Ghidden():
    return XiCov1


def integrate(params=[], total_time=10, dt=1e-6, itraj=1, ext_signal=1, exp_path="",**kwargs):
    """
    h1 is the hypothesis i use to get the data. (with al the coefficients gamma1...)
    """
    global A0, A1, D0, D1, C0, C1, proj_C, kappa0, kappa1, omega0, omega1 , eta0, eta1, dW, sprev, XiCov1, XiCov0, B1, B0

    [xi1, kappa1, omega1, eta1], [xi0, kappa0, omega0, eta0] = params


    ### stationary state for the covariance

    A1, D1, E1, B1 = genoni_matrices(*[xi1, kappa1, omega1, eta1])
    A0, D0, E0, B0 = genoni_matrices(*[xi1, kappa1, omega1, eta1])

    ##### Notation#
    ###   C = -Sqrt[2] B^T
    ###   \Gamma =  - E^T/Sqrt[2]
    ###   Xi(\Cov) = Cov C^T - \Gamma^T = (- Cov B + E)/Sqrt[2]
    ##   Carefull that D changes also, so the ricatti equation is slightly different (factors \sqrt{2})

    proj_C = np.linalg.pinv(B1/np.sum(B1))
    XiCov1 = genoni_xi_cov(A1,D1,E1,B1, [xi1, kappa1, omega1, eta1])
    XiCov0 = genoni_xi_cov(A0,D0,E0,B0, [xi0, kappa0, omega0, eta0])


    lin0, lin1 = 0., 0.
    x1in ,p1in, x0in, p0in, dyxin, dypin = np.zeros(6)



    times = np.arange(0,total_time+dt,dt)

    #### generate trajectory of noises
    np.random.seed(itraj)
    dW = np.sqrt(dt)*np.random.randn(len(times),2)

    s0_hidden = np.array([0.,0.])
    s0_exper = np.array([x0in, p0in, lin0, lin1])

    yhidden, yexper, dys  = IntegrationLoop(s0_exper, s0_hidden,  times, dt)

    states1 = yhidden[:,0:2]
    states0 = yexper[:,0:2]
    liks = yexper[:,2:]

    path = get_path_config(total_time=total_time, dt=dt, itraj=itraj, exp_path=exp_path)


    os.makedirs(path, exist_ok=True)

    indis = np.logspace(1,np.log10(len(times)-1), int(1e5))
    indis = [int(k) for k in indis]
    timind = [times[ind] for ind in indis]
    logliks_short =  np.array([liks[ii] for ii in indis])


    np.save(path+"logliks",logliks_short)#_short)
    np.save(path+"states1",np.array(np.array([states1[ii] for ii in indis]) ))
    np.save(path+"states0",np.array(np.array([states0[ii] for ii in indis]) ))

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

    total_time, dt = get_total_time_dt(params, dt=dt, total_time=total_time, ppp=ppp)

    integrate(total_time = total_time,
                dt = dt,
                itraj=itraj,
                exp_path = exp_path,
                save_all = save_all,
                params = params
                )
