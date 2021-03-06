import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utilities.misc import *
import argparse
from datetime import datetime
import os
from integration.steps import RK4_step, Ikpw, RosslerStep
import ast
from numba import jit


def Euler(ff, G, y0, times, dt,**kwargs):
    exp = kwargs.get("exp",False)
    N = len(times)+1
    y = np.zeros((N, len(y0)))

    if exp is True:
        dtexp = 1.
    else:
        dtexp = dt

    y[0] = y0
    for ind, t in enumerate(tqdm(times)):
        y[ind+1] = y[ind] + ff(y[ind], t, dt=dt)*dtexp + np.dot(G(y[ind], t), noises[ind,:])
    return y



def Fs_exp(s,t, coeffs=None, params=None, dt=1.):
    """
    dt demanded for exp
    """
    x = s[0:2]
    ExpA = np.array([[np.cos(omega*dt), np.sin(omega*dt)], [-np.sin(omega*dt), np.cos(omega*dt)]])
    xdot = np.dot(ExpA-np.eye(2), x)  #evolution update (according to what you measure)

    y = s[2:4]
    ydot = np.dot(C,x)
    varx, varp,covxp = s[4:7]
    varx_dot, covxp_dot, varp_dot = ders(varx, varp, covxp)
    return np.array([xdot[0], xdot[1], ydot[0],  ydot[1], varx_dot, varp_dot, covxp_dot])


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

    varx_dot, covxp_dot, varp_dot = ders_cov(varx, varp, covxp)
    
    u_th = s[7:9]
    u_th_dot = np.dot(C, np.dot(A_th, x) + np.dot(np.dot(A, C_inv), u_th))

    varx_th, varp_th, covxp_th = s[9:12]

    varx_th_dot, covxp_th_dot, varp_th_dot = ders_cov_th(varx, varp, covxp, varx_th, varp_th, covxp_th)
    
    
    return np.array([xdot[0], xdot[1], ydot[0],  ydot[1], varx_dot, varp_dot, covxp_dot, u_th_dot[0], u_th_dot[1] , varx_th_dot, varp_th_dot, covxp_th_dot])

@jit(nopython=True)
def Gs(s,t, coeffs=None, params=None):
    #cov = s_to_cov(s)
    varx, varp,covxy = s[4:7]
    cov = np.array([[varx, covxy], [covxy, varp]])

    XiCov = np.dot(cov, C.T) + Lambda.T
    varx_th, varp_th, covxy_th = s[9:12]

    cov_th = np.array([[varx_th, covxy_th], [covxy_th, varp_th]])
    XiCov_th = np.dot(cov_th, C.T)

    wieners = np.zeros((s.shape[0], s.shape[0]))
    wieners[:2,:2]  = XiCov
    wieners[2:4,2:4] = proj_C
    wieners[7:9,7:9] = np.dot(C, XiCov_th)
    return wieners


def integrate(periods, ppp, method="rossler", itraj=1, exp_path="",**kwargs):

    """
    everything is in the get_def_path()

    note that if you sweep params exp_path needs to specified
    """
    global A, C, D, Lambda, eta, gamma, omega, n, kappa, rppp, ders, noises, proj_C, A_th, C_inv, ders_cov, ders_cov_th

    eta = kwargs.get("eta",1) #efficiency
    kappa = kwargs.get("kappa",1)
    gamma = kwargs.get("gamma",0.3)
    omega = kwargs.get("omega",2*np.pi)
    n = kwargs.get("n",2.0)

    print("integrating with parameters: \n eta {} \nkappa {}\ngamma {} \nomega {}\nn {}\n".format(eta,kappa,gamma,omega,n))
    rppp = kwargs.get("rppp",1)

    x0 = 1.
    p0 = 0.
    yx0 = 0.
    yp0 = 0.

    varx0, varp0, covxy0 = .3 ,2.4 ,1e-3
    varx0_th, varp0_th, covxy0_th = 0., 0., 0.

    ux_th, up_th = 0.,0.

    s0 = np.array([x0, p0, yx0, yp0, varx0, varp0, covxy0, ux_th, up_th, varx0_th, varp0_th, covxy0_th])

    C = np.array([[np.sqrt(2*eta*kappa),0],[0,0]]) #homodyne
    C_inv = np.linalg.pinv(C)
    proj_C = C/np.sum(C)
    A = np.array([[-gamma/2, omega],[-omega, -gamma/2]])
    A_th = np.array([[0,1],[-1,0]])
    D = np.diag([(gamma*(n+0.5))]*2)
    Lambda = np.zeros((2,2))

    ### obtained w/scipy, see utilities
    
    ders_cov = lambda varx, varp, covxp: [2*covxp*omega - 2*eta*kappa*varx**2 - gamma*varx + gamma*(n + 0.5),
                -2*covxp*eta*kappa*varx - covxp*gamma + omega*varp - omega*varx,
                 -2*covxp**2*eta*kappa - 2*covxp*omega - gamma*varp + gamma*(n + 0.5)]
    
    
    ders_cov_th = lambda varx, varp, covxp, varx_th, varp_th, covxp_th: [2*covxp + 2*covxp_th*omega - 4*eta*kappa*varx*varx_th - gamma*varx_th,
 -2*covxp*eta*kappa*varx_th - 2*covxp_th*eta*kappa*varx - covxp_th*gamma + omega*varp_th - omega*varx_th + varp - varx,
 -4*covxp*covxp_th*eta*kappa - 2*covxp - 2*covxp_th*omega - gamma*varp_th]
    
    
    Period = 2*np.pi/omega
    dt = Period/ppp
    times = np.arange(0.,Period*periods,dt)

    coeffs = [C, A, D , Lambda, dt, rppp]
    params = [eta, gamma, kappa, omega, n]

    ### what we integrate
    remainder = (len(times)%rppp)
    if remainder > 0:
        tspan = times[:-remainder] #this is so we can split evenly
    else:
        tspan = times

    #### generate long trajectory of noises
    np.random.seed(itraj)
    
    dW = np.zeros((len(tspan), 12)) #I have ppp*periods points.
    for ind,t in enumerate(tspan):
        w0 = np.random.normal()*np.sqrt(dt)
        w1 = np.random.normal()*np.sqrt(dt)
        dW[ind,:] = np.array([w0, w1, w0, w1 , 0.,0.,0., w0, w1, 0., 0., 0.])  ## x0, x1,  y0, y1, varx, covxp, varp, u_th0, u_th1, var_uth0, covuth, varputh

    integration_times = tspan[::rppp] #jump tspan with step rppp
    noises = np.zeros((len(integration_times),dW.shape[1]))
    for sl in range(rppp):
        noises+=dW[sl::rppp,:]
    integration_step = rppp*dt

    if method.lower()=="rossler":
        print("integrating with rossler")
        solution = RosslerSRI2(Fs, Gs, s0, integration_times, integration_step)
    elif method.lower()=="euler":
        print("integrating with euler")
        solution = Euler(Fs, Gs, s0, integration_times, integration_step)
    elif method.lower()=="expeuler": #this blows-up when the non-physical scenario...
        print("integrating with Exp-euler")
        solution = Euler(Fs_exp, Gs, s0, integration_times, integration_step, exp=True)
    elif method.lower()=="rk4":
        print("integrating with RK4 is deprecatred due to weak convergence.")

    states, signals, covs, u_th, covs_th = convert_solution(solution)
    path = get_path_config(periods = periods, ppp= ppp, rppp=rppp, method=method, itraj=itraj, exp_path=exp_path)

    os.makedirs(path, exist_ok=True)
    np.save(path+"times",np.array(integration_times ))
    np.save(path+"states",np.array(states ))
    np.save(path+"covs",np.array(covs ))
    np.save(path+"signals",np.array(signals ))
    np.save(path+"params",params)
    np.save(path+"u_th",np.array(u_th))
    np.save(path+"covs_th",np.array(covs_th))

    print("traj saved in", path)
    return times, states, signals, covs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--itraj", type=int, default=1)
    parser.add_argument("--periods",type=int, default=50)
    parser.add_argument("--ppp", type=int,default=500)
    parser.add_argument("--method", type=str,default="rossler")
    parser.add_argument("--rppp", type=int,default=1)
    parser.add_argument("--params", type=str, default="") #[eta, gamma, kappa, omega, n]
    args = parser.parse_args()

    itraj = args.itraj ###this determines the seed
    periods = args.periods
    ppp = args.ppp
    method = args.method
    rppp = args.rppp
    params = args.params

    params, exp_path = check_params(params)
    [eta, gamma, kappa, omega, n] = params

    integrate(periods, ppp, method=method, itraj=itraj, exp_path = exp_path ,rppp = rppp,
                eta=eta, kappa = kappa,  gamma = gamma, n = n, omega = omega)
