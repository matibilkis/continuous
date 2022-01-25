import os
import numpy as np
from misc import ct
from tqdm import tqdm
from scipy.integrate import solve_ivp

def vector_to_matrix(v):
    return np.array([[v[0], v[1]],[v[2], v[3]]])
def matrix_to_vector(v):
    return np.array([v[0,0], v[0,1], v[1,0], v[1,1]])

def RK4(x,t,dt, fv,gv, parameters):
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



def generate_traj_RK4(ppp=500, periods = 40, itraj=0, path = ".", seed=0, **kwargs):

    eta = kwargs.get("eta",1) #efficiency
    gamma = kwargs.get("gamma",0.3) # damping (related both to D and to A)
    Lambda = kwargs.get("Lambda",0.8) #rate of measurement
    omega = kwargs.get("omega",2*np.pi) #rate of measurement
    n = kwargs.get("n",10.0)

    periods = int(periods*2*np.pi/omega)
    dt = 1/ppp

    times = np.arange(0,periods+ dt, dt)

    A = np.array([[-.5*gamma, omega], [-omega, -0.5*gamma]])
    D = np.diag([(gamma*(n+0.5)) + Lambda]*2)
    C = np.diag([np.sqrt(4*eta*Lambda)]*2)

    cov_in = np.eye(2)

    xi = lambda cov: np.dot(cov, ct(C)) + ct(D)
    print("integrating ricatti eq... covariance...")
    def dcovdt(t,cov):
        cov= vector_to_matrix(cov)
        XiCov = xi(cov)
        ev_cov = np.dot(A,cov) + np.dot(cov, ct(A)) + D - np.dot(XiCov, ct(XiCov))
        return matrix_to_vector(ev_cov)

    integrate_cov = solve_ivp(dcovdt, y0=matrix_to_vector(cov_in), t_span=(0,times[-1]), t_eval=times, max_step = dt, atol=1, rtol=1)
    covs = np.reshape(integrate_cov.y.T, (len(times),2,2))

    np.random.seed(seed)

    def f(t,x,parameters=None):
        return np.dot(A, x)

    xi = lambda cov: np.dot(cov, ct(C)) + ct(D)
    def g(t,x,parameters=None):
        gg = np.dot(xi(covs[parameters]),[np.random.normal(), np.random.normal()])
        return gg

    states = np.zeros((len(times),2))
    states[0] = np.array([0.,0.])
    print("integrating states")
    for ind, t in enumerate(tqdm(times[:-1])):
        states[ind+1] = RK4(states[ind], t, dt, f, g, parameters=ind)

    ## compute the signals from differences (possible error here, maybe numerical errors in the inverse propagate)... ?
    diffs = states[1:]-states[:-1]
    invsXiCov = np.array([np.linalg.inv(xi(cov)) for cov in covs[:-1]])
    CAx = np.einsum('ij,bj->bi',C-A,states[:-1])*dt
    signals = np.einsum('bij,bj->bi',invsXiCov,CAx + diffs)

    coeffs = [C, A, D , dt]
    params = [eta, gamma, Lambda, omega, n]

    path = path + "{}/RK4/".format(itraj)
    os.makedirs(path, exist_ok=True)
    np.save(path+"states".format(itraj),np.array(states ))
    np.save(path+"covs".format(itraj),np.array(covs ))
    np.save(path+"signals".format(itraj),np.array(signals ))
    np.save(path+"params".format(itraj),params)
    return
