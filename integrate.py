import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from misc import get_def_path, ct
import argparse
from datetime import datetime
import os

def RK4_step(x,t,dt, fv,gv, parameters):
    ###https://people.math.sc.edu/Burkardt/cpp_src/stochastic_rk/stochastic_rk.cpp
    #Runge-Kutta Algorithm for the Numerical Integration
    #of Stochastic Differential Equations
    ##
    a21 =   0.66667754298442
    a31 =   0.63493935027993
    a32 =   0.00342761715422#D+00
    a41 = - 2.32428921184321#D+00
    a42 =   2.69723745129487#D+00
    a43 =   0.29093673271592#D+00
    a51 =   0.25001351164789#D+00
    a52 =   0.67428574806272#D+00
    a53 = - 0.00831795169360#D+00
    a54 =   0.08401868181222#D+00

    q1 = 3.99956364361748#D+00
    q2 = 1.64524970733585#D+00
    q3 = 1.59330355118722#D+00
    q4 = 0.26330006501868#D+00

    t1 = t
    x1 = x
    k1 = dt*fv( t1, x1, parameters ) + np.sqrt(dt*q1)*gv( t1, x1, parameters)

    t2 = t1 + (a21 * dt)
    x2 = x1 + (a21 * k1)
    k2 = dt * fv( t2, x2, parameters) + np.sqrt(dt*q2)*gv( t2, x2, parameters)

    t3 = t1 + (a31 * dt)  + (a32 * dt)
    x3 = x1 + (a31 * k1) + (a32 * k2)
    k3 = dt * fv( t3 , x3, parameters) + np.sqrt(dt*q3)*gv( t3, x3, parameters)

    t4 = t1 + (a41 * dt)  + (a42 * dt)  + (a43 * dt)
    x4 = x1 + (a41 * k1) + (a42 * k2) + (a43 * k3)
    k4 = dt * fv( t4, x4, parameters) + np.sqrt(dt*q4)* gv( t4, x4, parameters)

    xstar = x1 + (a51 * k1) + (a52 * k2) + (a53 * k3) + (a54 * k4)
    return xstar

def RK4(y0,tspan,dt):
    def f(t,s,parameters=None):
        x = s[0:2]
        xdot = np.dot(A,x)
        y = s[2:4]
        ydot = np.dot(C,x)
        varx, varp,covxp = s[4:]
        varx_dot = ((0.5 + n)*gamma) - (varx*gamma) + Lambda - (4*eta*Lambda*covxp**2)  - ((0.5+n)*gamma  + Lambda + (2*varx*np.sqrt(eta*Lambda)))**2 + (2*covxp*omega)
        varp_dot = ((0.5 + n)*gamma) - (varp*gamma) + Lambda - (4*eta*Lambda*covxp**2) -  ((0.5+n)*gamma + Lambda + (2*varp*np.sqrt(eta*Lambda)))**2 - (2*covxp*omega)
        covxp_dot = covxp*(-(4*eta*varp) - (4*varx*eta) - (4*np.sqrt(eta*Lambda))  ) + covxp*gamma*(-1 -2*np.sqrt(eta*Lambda) - (4*n*np.sqrt(eta*Lambda))) + (varp*omega - varx*omega)
        return np.array([xdot[0], xdot[1], ydot[0],  ydot[1], varx_dot, varp_dot, covxp_dot])

    def s_to_cov(s,begin_cov=4):
        varx, varp,covxy = s[begin_cov:]
        cov = np.array([[varx, covxy], [covxy, varp]])
        return cov

    def g(t,s,parameters=None):
        cov = s_to_cov(s)
        XiCov = np.dot(cov, C.T) + D.T
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
        dint = 1.
    else:
        dint= dt

    for ind,t in enumerate(tqdm(tspan[:-1])):
        #dW = np.random.normal(0,np.sqrt(dt), (7))
        w0 = np.random.normal()*np.sqrt(dt)
        w1 = np.random.normal()*np.sqrt(dt)
        dW = np.array([w0, w1, w0, w1 , 0.,0.,0.])
        y[ind+1] = y[ind] + ff(y[ind], t, exp=exp, dt=dt)*dint + np.dot(G(y[ind], t), dW)
    return y

def dot(a, b):
    return np.einsum('ijk,ikl->ijl', a, b)

def Aterm(N, h, m, k, dW):
    """kth term in the sum of Wiktorsson2001 equation (2.2)"""
    sqrt2h = np.sqrt(2.0/h)
    Xk = np.random.normal(0.0, 1.0, (N, m, 1))
    Yk = np.random.normal(0.0, 1.0, (N, m, 1))
    term1 = dot(Xk, (Yk + sqrt2h*dW).transpose((0, 2, 1)))
    term2 = dot(Yk + sqrt2h*dW, Xk.transpose((0, 2, 1)))
    return (term1 - term2)/k


def Ikpw(dW, h, n=5):
    """matrix I approximating repeated Ito integrals for each of N time
    intervals, based on the method of Kloeden, Platen and Wright (1992).
    Args:
      dW (array of shape (N, m)): giving m independent Weiner increments for
        each time step N. (You can make this array using sdeint.deltaW())
      h (float): the time step size
      n (int, optional): how many terms to take in the series expansion
    Returns:
      (A, I) where
        A: array of shape (N, m, m) giving the Levy areas that were used.
        I: array of shape (N, m, m) giving an m x m matrix of repeated Ito
        integral values for each of the N time intervals.
    """
    N = dW.shape[0]
    m = dW.shape[1]
    if dW.ndim < 3:
        dW = dW.reshape((N, -1, 1)) # change to array of shape (N, m, 1)
    if dW.shape[2] != 1 or dW.ndim > 3:
        raise(ValueError)
    A = Aterm(N, h, m, 1, dW)
    for k in range(2, n+1):
        A += Aterm(N, h, m, k, dW)
    A = (h/(2.0*np.pi))*A
    I = 0.5*(dot(dW, dW.transpose((0,2,1))) - np.diag(h*np.ones(m))) + A
    dW = dW.reshape((N, -1)) # change back to shape (N, m)
    return (A, I)


def RosslerSRI2(f, G, y0, times, dt):
    """Implements the Roessler2010 order 1.0 strong Stochastic Runge-Kutta

      IJmethod (callable): which function to use to generate repeated
        integrals. N.B. for an Ito equation, must use an Ito version here
        (either Ikpw or Iwik). For a Stratonovich equation, must use a
        Stratonovich version here (Jkpw or Jwik).
      dW: optional array of shape (len(tspan)-1, d).
      IJ: optional array of shape (len(tspan)-1, m, m).
        Optional arguments dW and IJ are for advanced use, if you want to
        use a specific realization of the d independent Wiener processes and
        their multiple integrals at each time step. If not provided, suitable
        values will be generated randomly.
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
        Yn = y[ind] # shape (d,)
        Ik = dW[ind,:] # shape (m,)

        Iij = I[ind,:,:] # shape (m, m)
        fnh = f(Yn, t)*dt # shape (d,)

        Gn = G(Yn, t)
        sum1 = np.dot(Gn, Iij)/np.sqrt(dt) # shape (d, m)

        H20 = Yn + fnh # shape (d,)
        H20b = np.reshape(H20, (d, 1))
        H2 = H20b + sum1 # shape (d, m)

        H30 = Yn
        H3 = H20b - sum1
        fn1h = f(H20, times[ind+1])*dt
        Yn1 = Yn + 0.5*(fnh + fn1h) + np.dot(Gn, Ik)
        for k in range(0, m):
            Yn1 += 0.5*np.sqrt(dt)*(G(H2[:,k], times[ind+1])[:,k] - G(H3[:,k], times[ind+1])[:,k])
        y[ind+1] = Yn1
    return y


def Fs(s,t, coeffs=None, params=None,exp=False, dt=1.):
    x = s[0:2]
    if exp == True:
        ExpA = np.array([[np.cos(omega*dt), np.sin(omega*dt)], [-np.sin(omega*dt), np.cos(omega*dt)]])
        if unphysical is not True:
            ExpA*=np.exp(-gamma*dt/2)
        xdot = np.dot(ExpA-np.eye(2), x)  #evolution update (according to what you measure)
    else:
        xdot = np.dot(A,x)

    y = s[2:4]
    ydot = np.dot(C,x)

    varx, varp,covxp = s[4:]

    if unphysical is True:
        varx_dot = ((0.5+n)*gamma)  + Lambda   +( 4*eta*Lambdacovxp**2) + ( (0.5+n)*gamma  + Lambda + 2*varx*np.sqrt(Lambda*eta)) **2 + (2*covxp*omega)
        varp_dot = ((0.5+n)*gamma)  + Lambda   +( 4*eta*Lambdacovxp**2) + ( (0.5+n)*gamma  + Lambda + 2*varp*np.sqrt(Lambda*eta)) **2 - (2*covxp*omega)
        covxp_dot =  (4*covxp*((varp*eta*Lambda) + (varx*eta*Lambda)  + np.sqrt(eta*Lambda)*((0.5+n)*gamma + Lambda)  ))   + omega*(varp -varx)

    else:

        varx_dot = ((0.5 + n)*gamma) - (varx*gamma) + Lambda - (4*eta*Lambda*covxp**2)  - ((0.5+n)*gamma  + Lambda + (2*varx*np.sqrt(eta*Lambda)))**2 + (2*covxp*omega)
        varp_dot = ((0.5 + n)*gamma) - (varp*gamma) + Lambda - (4*eta*Lambda*covxp**2) -  ((0.5+n)*gamma + Lambda + (2*varp*np.sqrt(eta*Lambda)))**2 - (2*covxp*omega)
        #covxp_dot = covxp*(-(4*eta*varp) - (4*varx*eta) - (4*np.sqrt(eta*Lambda))  ) + covxp*gamma*(-1 -2*np.sqrt(eta*Lambda) - (4*n*np.sqrt(eta*Lambda))) + (varp*omega - varx*omega)
        covxp_dot = Lambda*covxp*4*((varp*eta)  + (varx*eta)  + np.sqrt(eta*Lambda) )   + (covxp*gamma*(-1 + 2*np.sqrt(eta*Lambda) + 4*np.sqrt(eta*Lambda)*n)) + omega*(varp - varx)

    return np.array([xdot[0], xdot[1], ydot[0],  ydot[1], varx_dot, varp_dot, covxp_dot])


def s_to_cov(s,begin_cov=4):
    varx, varp,covxy = s[begin_cov:]
    cov = np.array([[varx, covxy], [covxy, varp]])
    return cov


def Gs(s,t, coeffs=None, params=None):
    cov = s_to_cov(s)
    XiCov = np.dot(cov, C.T) + D.T
    wieners = np.zeros((s.shape[0], s.shape[0]))
    wieners[:2,:2]  = XiCov
    wieners[2:4,2:4] = np.eye(2)
    return wieners*kill_noise

def convert_solution(ss):
    states = ss[:,0:2]

    signals = ss[:,2:4]
    signals = signals[1:] - signals[:-1]

    covss = ss[:,-3:]
    covs = [s_to_cov(s,begin_cov=0) for s in covss]
    return states, signals, covs


def integrate(periods, ppp, method="rossler", itraj=1, path="",**kwargs):

    global A, C, D, eta, gamma, Lambda, omega, n, kill_noise, unphysical

    kill_noise = 1.
    eta = kwargs.get("eta",1) #efficiency
    gamma = kwargs.get("gamma",0.3) # damping (related both to D and to A)
    Lambda = kwargs.get("Lambda",0.8) #rate of measurement
    omega = kwargs.get("omega",2*np.pi) #rate of measurement
    n = kwargs.get("n",10.0)

    unphysical = kwargs.get("unphysical",False)

    x0 = 1.
    p0 = 0.
    yx0 = 0.
    yp0 = 0.
    varx0 = 1.
    varp0 = 1.
    covxy0 = 0.
    s0 = np.array([x0, p0, yx0, yp0, varx0, varp0,covxy0])

### to change unphy
    if unphysical is True:
        A = np.array([[0., omega], [-omega, 0.]])
    else:
        A = np.array([[-.5*gamma, omega], [-omega, -0.5*gamma]])

    D = np.diag([(gamma*(n+0.5)) + Lambda]*2)
    C = np.diag([np.sqrt(4*eta*Lambda)]*2)

    dt = 1/ppp
    times = np.arange(0.,periods+dt,dt)

    coeffs = [C, A, D , dt]
    params = [eta, gamma, Lambda, omega, n]

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
    path = path + "{}periods/{}ppp/{}/{}/".format(periods,ppp,method, itraj)
    if unphysical is True:
        path+="unphysical_"

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
