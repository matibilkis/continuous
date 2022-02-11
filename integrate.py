import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from misc import get_def_path, ct, s_to_cov, convert_solution
import argparse
from datetime import datetime
import os
from steps import RK4_step, Ikpw, RosslerStep



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


def Fs(s,t, coeffs=None, params=None, dt=None):
    """
    """
    x = s[0:2]
    xdot = np.dot(A,x)

    y = s[2:4]
    ydot = np.dot(C,x)

    varx, varp,covxp = s[4:]
    varx_dot, covxp_dot, varp_dot = ders(varx, varp, covxp)
    return np.array([xdot[0], xdot[1], ydot[0],  ydot[1], varx_dot, varp_dot, covxp_dot])


def Fs_exp(s,t, coeffs=None, params=None, dt=1.):
    """
    dt demanded for exp
    """
    x = s[0:2]
    ExpA = np.array([[np.cos(omega*dt), np.sin(omega*dt)], [-np.sin(omega*dt), np.cos(omega*dt)]])
    xdot = np.dot(ExpA-np.eye(2), x)  #evolution update (according to what you measure)

    y = s[2:4]
    ydot = np.dot(C,x)
    varx, varp,covxp = s[4:]
    varx_dot, covxp_dot, varp_dot = ders(varx, varp, covxp)
    return np.array([xdot[0], xdot[1], ydot[0],  ydot[1], varx_dot, varp_dot, covxp_dot])

def Gs(s,t, coeffs=None, params=None):
    cov = s_to_cov(s)
    XiCov = np.dot(cov, C.T) + Lambda.T
    wieners = np.zeros((s.shape[0], s.shape[0]))
    wieners[:2,:2]  = XiCov
    wieners[2:4,2:4] = np.eye(2)
    return wieners


def integrate(periods, ppp, method="rossler", itraj=1, path="",**kwargs):

    global A, C, D, Lambda, eta, gamma, omega, n, kappa, rppp, ders, noises

    eta = kwargs.get("eta",1) #efficiency
    kappa = kwargs.get("kappa",1) 
    gamma = kwargs.get("gamma",0.3)
    omega = kwargs.get("omega",2*np.pi)
    n = kwargs.get("n",2.0)
    rppp = kwargs.get("rppp",1)

    x0 = 1.
    p0 = 0.
    yx0 = 0.
    yp0 = 0.
    varx0 = 1.
    varp0 = 1.
    covxy0 = 0.
    s0 = np.array([x0, p0, yx0, yp0, varx0, varp0,covxy0])

    C = np.array([[np.sqrt(2*eta*kappa),0],[0,0]]) #homodyne
    A = np.array([[-gamma/2, omega],[-omega, -gamma/2]])
    D = np.diag([(gamma*(n+0.5))]*2)
    Lambda = np.zeros((2,2))

    ders = lambda varx, varp, covxp: [-2*covxp**2*eta*kappa + 2*covxp*omega - gamma*varx + gamma*(n + 0.5) - 2*varx**2*eta*kappa,-covxp*gamma + omega*varp - omega*varx, -2*covxp*omega - gamma*varp + gamma*(n + 0.5)]

    dt = 1/ppp
    times = np.arange(0.,periods,dt)

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
    dW = np.zeros((len(tspan), 7)) #I have ppp*periods points.
    for ind,t in enumerate(tspan):
        w0 = np.random.normal()*np.sqrt(dt)
        w1 = np.random.normal()*np.sqrt(dt)
        dW[ind,:] = np.array([w0, w1, w0, w1 , 0.,0.,0.])

    integration_times = tspan[::rppp] #jump tspan with step rppp
    noises = np.zeros((len(integration_times),7))
    for sl in range(rppp):
        #print(dW[sl::rppp,:].shape, sl, len(times), remainder, len(integration_times), rppp, noises.shape, dW[::rppp, :].shape)
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
    states, signals, covs = convert_solution(solution)

    if path == "":
        path = get_def_path()
    path+="rppp{}/".format(rppp)
    path = path + "{}periods/{}ppp/{}/{}/".format(periods,ppp,method, itraj)

    os.makedirs(path, exist_ok=True)
    np.save(path+"times",np.array(integration_times ))
    np.save(path+"states",np.array(states ))
    np.save(path+"covs",np.array(covs ))
    np.save(path+"signals",np.array(signals ))
    np.save(path+"params",params)

    return times, states, signals, covs

if __name__ == "__main__":

    defpath = get_def_path()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--path", type=str, default=defpath)
    parser.add_argument("--itraj", type=int, default=1)
    parser.add_argument("--periods",type=int, default=50)
    parser.add_argument("--ppp", type=int,default=500)
    parser.add_argument("--method", type=str,default="rossler")
    parser.add_argument("--rppp", type=int,default=1)

    args = parser.parse_args()
    path = args.path
    itraj = args.itraj
    periods = args.periods
    ppp = args.ppp
    method = args.method
    rppp = args.rppp

    integrate(periods, ppp, method=method, itraj=1, path="",rppp = rppp)
