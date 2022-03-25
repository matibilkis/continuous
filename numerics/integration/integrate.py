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
def give_ders(vx, vp, cvxp, gamma_val, omega_val, kappa_val,n_val, eta_val):
    return  [-4*cvxp**2*eta_val*kappa_val + 2*cvxp*omega_val - gamma_val*vx + gamma_val*(n_val + 0.5) + kappa_val - 4*vx**2*eta_val*kappa_val,
     -4*cvxp**2*eta_val*kappa_val - 2*cvxp*omega_val - gamma_val*vp + gamma_val*(n_val + 0.5) + kappa_val - 4*vp**2*eta_val*kappa_val,
     -cvxp*gamma_val - 4*cvxp*vp*eta_val*kappa_val - 4*cvxp*vx*eta_val*kappa_val + omega_val*vp - omega_val*vx]


def Integrate_hybrid(f_hidden, G_gidden, y0_hidden, y0_exp, times, dt):
    """
    dy = f(y,t) *dt + G(y,t) *dW
    """
    N = len(times)+1
    d = len(y0_hidden)
    m = len(y0_hidden)
    _,I=Ikpw(dW,dt)

    yhidden = np.zeros((N, d))
    yexper = np.zeros((N, len(y0_exp)))

    yhidden[0] = y0_hidden
    yexper[0] = y0_exp

    dys = []
    # Gn = np.zeros((d, m), dtype=yhidden.dtype)

    for ind, t in enumerate(tqdm(times)):
        yhidden[ind+1] = RosslerStep(t, yhidden[ind], dW[ind,:], I[ind,:,:], dt, f_hidden, G_gidden, d, m) ### this updates the hidden state (and the covariance)

        x1 = yhidden[ind][:2]
        dy = np.dot(C1,x1)*dt + dW[ind,:2]
        dys.append(dy)

        ####   euler update, given signal ####
        s = yexper[ind]
        x0 = s[:2]
        vx, vp,cvxp = s[2:5]
        cov = np.array([[vx, cvxp], [cvxp, vp]])
        xicovC0 = np.dot(np.dot(cov,C0.T),C0)

        dx0 = np.dot(A0 - xicovC0, x0)*dt + np.dot(np.dot(cov,C0.T), dy)
        varx_dot, varp_dot, covxy_dot = give_ders(vx, vp, cvxp, gamma0, omega0, kappa0, n0, eta0)
        dvx, dvp, dcvxp = dt*varx_dot, dt*varp_dot, dt*covxy_dot

        l0, l1 = s[5:7]
        u0 = np.dot(C0,x0)
        u1 = np.dot(C1,x1)
        dl0 = -dt*np.dot(u0,u0)/2 + np.dot(u0, dy)
        dl1 = -dt*np.dot(u1,u1)/2 + np.dot(u1, dy)

        yexper[ind+1] = [(x0 + dx0)[0], (x0 + dx0)[1],  vx + dvx, vp + dvp , cvxp + dcvxp, l0 + dl0, l1+dl1 ]
    return yhidden, yexper, dys

@jit(nopython=True)
def Fhidden(s, t, dt):
    """
    """
    x1 = s[0:2]
    x1_dot = np.dot(A1,x1)
    vx, vp,cvxp = s[2:5]
    varx_dot, varp_dot, covxy_dot = give_ders(vx, vp, cvxp, gamma1, omega1, kappa1, n1, eta1)
    return np.array([x1_dot[0], x1_dot[1],varx_dot, varp_dot, covxy_dot])#, x1dot[0], x1dot[1], varx1_dot, varp1_dot, covxy1_dot, l_dot, l1_dot])

@jit(nopython=True)
def Ghidden(s,t, coeffs=None, params=None):
    wieners = np.zeros((s.shape[0], s.shape[0]))
    varx, varp,covxy = s[2:5]
    cov = np.array([[varx, covxy], [covxy, varp]])
    XiCov = np.dot(cov, C1.T)
    wieners[:2,:2]  = XiCov
    return wieners

# @jit(nopython=True)
# def Fexperi(s, t, x1, dy):
#     x0 = s[:2]
#     vx, vp,cvxp = s[2:5]
#     cov = np.array([[vx, cvxp], [cvxp, vp]])
#     xicovC0 = np.dot(np.dot(cov,C0.T),C0)
#     x_dot0 = np.dot(A0 - xicovC0, x0) + np.dot(np.dot(cov,C0.T), dy/dt)
#     varx_dot, varp_dot, covxy_dot = give_ders(vx, vp, cvxp, gamma0, omega0, kappa0, n0, eta0)
#
#     u0 = np.dot(C0,x0)
#     u1 = np.dot(C1,x1)
#
#     dl0 = -np.dot(u0,u0)/2 + np.dot(u0, dy)/dt
#     dl1 = -np.dot(u1,u1)/2 + np.dot(u1, dy)/dt
#
#     return np.array([x_dot0[0], x_dot0[1], varx_dot, varp_dot, covxy_dot, dl0, dl1 ])

def integrate(total_time=10, dt=1e-6, itraj=1, exp_path="",**kwargs):
    """
    h1 is the hypothesis i use to get the data. (with al the coefficients gamma1...)
    """
    global A0, A1, D0, D1, C0, C1,proj_C, gamma0, gamma1, omega0, omega1 , eta0, eta1, n0, n1, kappa0, kappa1, dW, sprev

    params1 = [gamma1, omega1, n1, eta1, kappa1]
    params0 = [gamma0, omega0, n0, eta0, kappa0]

    print("params0: {}\nparams1 {}".format(params1,params0))

    def give_matrices(gamma, omega, n, eta, kappa):
        A = np.array([[-gamma/2, omega],[-omega, -gamma/2]])
        C = np.sqrt(4*eta*kappa)*np.array([[1.,0.],[0.,1.]]) #homodyne
        D = np.diag([gamma*(n+0.5) + kappa]*2)
        return A, C, D

    A1, C1, D1 = give_matrices(gamma1, omega1, n1, eta1, kappa1)
    A0, C0, D0 = give_matrices(gamma0, omega0, n0, eta0, kappa0)

    lin0, lin1 = 0., 0.
    x1in ,p1in, x0in, p0in, dyxin, dypin = np.zeros(6)

    ### stationary state for the covariance
    def stat(gamma, omega, n, eta, kappa):
        suc = n + 0.5 + kappa/gamma
        sst = (gamma/(8*eta*kappa))*(np.sqrt(1 + 16*eta*kappa*suc/gamma ) -1 )
        return suc, sst

    suc1, sst1 = stat(gamma1, omega1, n1, eta1, kappa1)
    suc0, sst0 = stat(gamma0, omega0, n0, eta0, kappa0)
    varx10, varp10, covxy10 = sst1 ,suc1 ,0.
    varx0, varp0, covxy0 = sst0 ,suc0 ,0.

    # s0_hidden = np.array([x1in, p1in, dyxin, dypin, varx10, varp10, covxy10])
    # s0_exper = np.array([x0in, p0in, varx0 , varp0 , covxy0, lin0, lin1])

    s0_hidden = np.array([x1in, p1in, varx10, varp10, covxy10])
    s0_exper = np.array([x0in, p0in, varx0 , varp0 , covxy0, lin0, lin1])

    times = np.arange(0,total_time+dt,dt)
    params = [params1,params0]

    #### generate long trajectory of noises
    np.random.seed(itraj)
    dW = np.zeros((len(times), 5))
    for ind,t in enumerate(times):
        w0 = np.random.normal()*np.sqrt(dt)
        w1 = np.random.normal()*np.sqrt(dt)
        dW[ind,:2] = np.array([w0, w1])  ## x0, x1,  y0, y1, varx, covxp, varp, u_th0, u_th1, var_uth0, covuth, varputh

    yhidden, yexper, dys = Integrate_hybrid(Fhidden, Ghidden, s0_hidden, s0_exper,  times, dt)
    states1 = yhidden[:,0:2]
    #dys = yhidden[:,2:4]
    covs1 = yhidden[:,2:5]

    states0 = yexper[:,0:2]
    covs0 = yexper[:,2:5]
    liks = yexper[:,5:]

    path = get_path_config_bis(total_time=total_time, dt=dt, method="hybrid", itraj=itraj, exp_path=exp_path)

    os.makedirs(path, exist_ok=True)
    np.save(path+"times",np.array(times ))
    np.save(path+"params",params)

    np.save(path+"states1",np.array(states1 ))
    np.save(path+"covs1",np.array(covs1 ))
    np.save(path+"states0",np.array(states0 ))
    np.save(path+"covs0",np.array(covs0 ))

    np.save(path+"dys",np.array(dys ))
    np.save(path+"logliks",liks)

    print("traj saved in", path)
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--itraj", type=int, default=1)
    parser.add_argument("--dt",type=float, default=1e-6)
    parser.add_argument("--total_time", type=float,default=0.4)
    parser.add_argument("--h1true", type=int, default=0)
    args = parser.parse_args()

    itraj = args.itraj ###this determines the seed
    total_time = args.total_time
    dt = args.dt
    h1 = args.h1true

    params = give_def_params_discrimination(flip = h1)
    print(params)
    params, exp_path = check_params_discrimination(params)
    [gamma1, omega1, n1, eta1, kappa1], [gamma0, omega0, n0, eta0, kappa0] = params

    integrate(total_time = total_time, dt = dt,
            itraj=itraj, exp_path = exp_path,
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
