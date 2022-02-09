import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from misc import get_def_path, ct, s_to_cov, convert_solution
import argparse
from datetime import datetime
import os
from steps import RK4_step, Ikpw, RosslerStep


def RK4(y0,tspan,dt):
    def f(t,s,parameters=None):
        x = s[0:2]
        xdot = np.dot(A,x)
        y = s[2:4]
        ydot = np.dot(C,x)
        varx, varp,covxp = s[4:]

        ders = [(-2*covxp**2*eta*kappa + 2*covxp*omega - gamma*varx + gamma*(n + 0.5) - 2*varx**2*eta*kappa,
      -covxp*gamma + omega*varp - omega*varx,
      -2*covxp*omega - gamma*varp + gamma*(n + 0.5))]

        varx_dot, covxp_dot, varp_dot = ders

        return np.array([xdot[0], xdot[1], ydot[0],  ydot[1], varx_dot, varp_dot, covxp_dot])

    def g(t,s,parameters=None):
        cov = s_to_cov(s)
        XiCov = np.dot(cov, C.T) + Lambda.T
        ww = np.zeros(s.shape[0])
        noises = np.array([np.random.normal(), np.random.normal()])
        ww[:2] = np.dot(XiCov, noises)
        ww[2:4] = noises
        return ww*kill_noise

    ss = np.zeros((len(tspan),len(y0)))
    ss[0] = y0
    for ind, t in enumerate(tqdm(tspan[:-1])):
        ss[ind+1] = RK4_step(ss[ind], t, dt, f, g, parameters=None)
    return ss

def Euler(ff, G, y0, tspan, dt,**kwargs):
    exp = kwargs.get("exp",False)
    N = len(tspan)
    y = np.zeros((N, len(y0)))
    y[0] = y0
    if exp is True:
        dtexp = 1.
    else:
        dtexp = dt

    for ind,t in enumerate(tqdm(tspan[:-1])):
        w0 = np.random.normal()*np.sqrt(dt)
        w1 = np.random.normal()*np.sqrt(dt)
        dW = np.array([w0, w1, w0, w1 , 0.,0.,0.])
        y[ind+1] = y[ind] + ff(y[ind], t, exp=exp, dt=dt)*dtexp + np.dot(G(y[ind], t), dW)
    return y

def RosslerSRI2(f, G, y0, times, dt):
    """
    dy = f(y,t) *dt + G(y,t) *dW
    borrowed from sdeint - python
    """
    #(d, m, f, G, y0, tspan, dW, IJ) = _check_args(f, G, y0, tspan, dW, IJ)
    N = len(times)
    d = len(y0)
    m = len(y0)

    #dW = np.random.normal(0,np.sqrt(dt), (N-1, m))
    dW = np.zeros((len(times), 7))
    for k in range(len(times)):
        w0 = np.random.normal()*np.sqrt(dt)
        w1 = np.random.normal()*np.sqrt(dt)
        dW[k,:] = np.array([w0, w1, w0, w1 , 0.,0.,0.])

    #_, I = Ikpw(dW, dt)
    _,I=Ikpw(dW,dt)
    # allocate space for result
    y = np.zeros((N, d))
    y[0] = y0;
    Gn = np.zeros((d, m), dtype=y.dtype)

    for ind, t in enumerate(tqdm(times[:-1])):
        y[ind+1] = RosslerStep(t,y[ind], dW[ind,:], I[ind,:,:], dt, f,G, d, m)
    return y


def Fs(s,t, exp=False, coeffs=None, params=None, dt=None):
    """
    maybe best is to have two functions...  flag on method
    """
    x = s[0:2]
    if exp is True:
        ExpA = np.array([[np.cos(omega*dt), np.sin(omega*dt)], [-np.sin(omega*dt), np.cos(omega*dt)]])
        xdot = np.dot(ExpA-np.eye(2), x)  #evolution update (according to what you measure)
    else:
        xdot = np.dot(A,x)

    y = s[2:4]
    ydot = np.dot(C,x)

    varx, varp,covxp = s[4:]

    ders = [-2*covxp**2*eta*kappa + 2*covxp*omega - gamma*varx + gamma*(n + 0.5) - 2*varx**2*eta*kappa,-covxp*gamma + omega*varp - omega*varx, -2*covxp*omega - gamma*varp + gamma*(n + 0.5)]

    varx_dot, covxp_dot, varp_dot = ders


    return np.array([xdot[0], xdot[1], ydot[0],  ydot[1], varx_dot, varp_dot, covxp_dot])



def Gs(s,t, coeffs=None, params=None):
    cov = s_to_cov(s)
    XiCov = np.dot(cov, C.T) + Lambda.T
    wieners = np.zeros((s.shape[0], s.shape[0]))
    wieners[:2,:2]  = XiCov
    wieners[2:4,2:4] = np.eye(2)
    return wieners*kill_noise


def integrate(periods, ppp, reduce_ppp = 1, method="rossler", itraj=1, path="",**kwargs):

    global A, C, D, Lambda, eta, gamma, omega, n, kappa, kill_noise, r_ppp

    kill_noise = 1.
    eta = kwargs.get("eta",1) #efficiency
    kappa = kwargs.get("kappa",1) #efficiency
    gamma = kwargs.get("gamma",0.3) # damping (related both to D and to A)
    omega = kwargs.get("omega",0)#2*np.pi) #rate of measurement
    n = kwargs.get("n",2.0)
    r_ppp = reduce_ppp

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

    dt = 1/ppp
    times = np.arange(0.,periods+dt,dt)
    coeffs = [C, A, D , Lambda, dt]
    params = [eta, gamma, kappa, omega, n]

    np.random.seed(itraj)
    if method=="rossler":
        print("integrating with rossler")
        solution = RosslerSRI2(Fs, Gs, s0, times, dt)
    elif method=="euler":
        print("integrating with euler")
        solution = Euler(Fs, Gs, s0, times, dt)
    elif method=="Expeuler": #this blows-up when the non-physical scenario...
        print("integrating with Exp-euler")
        solution = Euler(Fs, Gs, s0, times, dt,exp=True)
    elif method=="RK4":
        print("integrating with RK4")
        solution = RK4(s0,times,dt)
    states, signals, covs = convert_solution(solution)

    if path == "":
        path = get_def_path()
    path+="rppp{}/".format(r_ppp)
    path = path + "{}periods/{}ppp/{}/{}/".format(periods,ppp,method, itraj)

    os.makedirs(path, exist_ok=True)
    np.save(path+"times",np.array(times ))
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

    args = parser.parse_args()
    path = args.path
    itraj = args.itraj
    periods = args.periods
    ppp = args.ppp
    method = args.method

    integrate(periods, ppp, method=method, itraj=1, path="")
