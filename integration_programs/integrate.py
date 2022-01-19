import os
import numpy as np
from misc import ct
from tqdm import tqdm
import sdeint
from scipy.integrate import solve_ivp


def vector_to_matrix(v):
    return np.array([[v[0], v[1]],[v[2], v[3]]])
def matrix_to_vector(v):
    return np.array([v[0,0], v[0,1], v[1,0], v[1,1]])


def generate_traj_RK(ppp=500, periods = 40, itraj=0, path = ".", seed=0):

    times = np.linspace(0,periods,ppp*periods)
    gamma = 1 #damping from outside
    Gamma = 1 #measurement rate
    eta = 1 # measurement efficiency
    n = 2 # number of photons

    w = 2*np.pi
    T = (2*np.pi)/w

    dt = 1/ppp

    C = np.array([[np.sqrt(4*eta*Gamma), 0] ,[0, np.sqrt(4*eta*Gamma)]])

    A = np.array([
        [0., w],
        [-w, 0.]])

    D = np.array([[gamma*(n + 0.5) + Gamma, 0], [0,gamma*(n + 0.5) + Gamma]])

    su = n + 0.5 + Gamma/gamma
    cov_in = np.array([[np.sqrt(1+ (16*eta*Gamma*su/gamma) -1)*gamma/(8*eta*Gamma), 0],
                       [0,np.sqrt(1+ (16*eta*Gamma*su/gamma) -1)*gamma/(8*eta*Gamma)]])

    xi = lambda cov: np.dot(cov, ct(C)) + ct(D)


    def dcovdt(t,cov):
        cov= vector_to_matrix(cov)
        XiCov = xi(cov)
        ev_cov = np.dot(A,cov) + np.dot(cov, ct(A)) + D - np.dot(XiCov, ct(XiCov))
        return matrix_to_vector(ev_cov)


    integrate_cov = solve_ivp(dcovdt, y0=matrix_to_vector(cov_in), t_span=(0,times[-1]), t_eval=times, max_step = dt, atol=1, rtol=1)
    covs = np.reshape(integrate_cov.y.T, (len(times),2,2))

    np.random.seed(seed)
    global i
    global it
    it=0
    i=0

    def f(x, t):
        return np.dot(A,x)

    def g(x, t):
        global it
        global i
        if it!=t:
            i+=1
            it=t
        #return 0*np.eye(2)
        return xi(covs[i])

    states = sdeint.itoSRI2(f, g, y0=np.array([1.,0.]), tspan=times)
    diffs = states[1:]-states[:-1]
    invsXiCov = np.array([np.linalg.inv(xi(cov)) for cov in covs[:-1]])
    CAx = np.einsum('ij,bj->bi',C-A,states[:-1])*dt
    signals = np.einsum('bij,bj->bi',invsXiCov,CAx + diffs)
    coeffs = [C, A, D , dt]

    path = path + "{}/RK/".format(itraj)
    os.makedirs(path, exist_ok=True)
    np.save(path+"states".format(itraj),states )
    np.save(path+"covs".format(itraj),covs )
    np.save(path+"signals".format(itraj),signals )
    np.save(path+"D".format(itraj),D)
    np.save(path+"C".format(itraj),C)
    np.save(path+"dt".format(itraj),np.array([dt]))
    np.save(path+"A".format(itraj),A)

    return states, covs, signals, coeffs



def generate_traj_Euler(ppp=4000, periods = 5, itraj=0, path = ".", seed=0):
    #define parameters
    print("generating trajectory")
    np.random.seed(seed)

    gamma = 1 #damping from outside
    Gamma = 1 #measurement rate
    eta = 1 # measurement efficiency
    n = 2 # number of photons?

    w = 2*np.pi
    T = (2*np.pi)/w

    C = np.array([[np.sqrt(4*eta*Gamma), 0] ,[0, np.sqrt(4*eta*Gamma)]])

    A = np.array([
        [0., w],
        [-w, 0.]])

    D = np.array([[gamma*(n + 0.5) + Gamma, 0], [0,gamma*(n + 0.5) + Gamma]])

    su = n + 0.5 + Gamma/gamma
    cov_in = np.array([[np.sqrt(1+ (16*eta*Gamma*su/gamma) -1)*gamma/(8*eta*Gamma), 0],
                       [0,np.sqrt(1+ (16*eta*Gamma*su/gamma) -1)*gamma/(8*eta*Gamma)]])

    dt = 1/ppp
    total_points = int(periods*ppp)

    xi = lambda cov: np.dot(cov, ct(C)) + ct(D)

    signals = []
    covs = [cov_in]
    means = [np.array([1.,0.])] ## initial condition
    xicovs = [xi(covs[-1])]

    for k in tqdm(range(total_points)):
        x = means[-1]
        cov = covs[-1]
        XiCov = xicovs[-1]

        ##deterministic (just to check)
        #dy = np.dot(C, x )*dt
        dy = np.dot(C, x )*dt#+ np.dot(np.linalg.pinv(C), np.random.randn(2)/np.sqrt(dt)))*dt # signal
        signals.append(dy)

        dx = np.dot(A - np.dot(XiCov,C), x)*dt + np.dot(XiCov, dy)  #evolution update (according to what you measure)
        dcov = (np.dot(A,cov) + np.dot(cov, ct(A)) + D - np.dot(XiCov, ct(XiCov)))*dt  #covariance update

        covs.append(covs[-1] + dcov)
        means.append(means[-1] + dx)
        xicovs.append(xi(covs[-1]))

    means = np.array(means)
    covs = np.array(covs)
    xicovs = np.array(xicovs)
    signals = np.array(signals)
    coeffs = [C, A, D , dt]

    os.makedirs(path+"{}/euler/".format(itraj), exist_ok=True)
    np.save(path+"{}/euler/states".format(itraj),np.array(means) )
    np.save(path+"{}/euler/covs".format(itraj),np.array(covs) )
    np.save(path+"{}/euler/signals".format(itraj),np.array(signals) )
    np.save(path+"{}/euler/xicovs".format(itraj),np.array(xicovs) )
    np.save(path+"{}/euler/D".format(itraj),D)
    np.save(path+"{}/euler/C".format(itraj),C)
    np.save(path+"{}/euler/dt".format(itraj),np.array([dt]))
    np.save(path+"{}/euler/A".format(itraj),A)

    return means, covs, xicovs, signals, coeffs
