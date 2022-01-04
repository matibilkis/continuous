import os
import random
random.seed(1)
import numpy as np
np.random.seed(1)
from misc import ct
from tqdm import tqdm


def generate_traj(ppp=4000, periods = 5, itraj=0, path = "."):
    #define parameters
    print("gneerating trajectory")
    gamma = 1 #damping from outside
    Gamma = 1 #measurement rate
    eta = 1 # measurement efficiency
    n = 2 # number of photons?

    w = 2*np.pi
    T = (2*np.pi)/w

    np.random.seed(1)

    C = np.array([[np.sqrt(4*eta*Gamma), 0] ,[0, np.sqrt(4*eta*Gamma)]])

    A = np.array([
        [0., w],
        [-w, 0.]])

    D = np.array([[gamma*(n + 0.5) + Gamma, 0], [0,gamma*(n + 0.5) + Gamma]])

    su = n + 0.5 + Gamma/gamma
    cov_in = np.array([[np.sqrt(1+ (16*eta*Gamma*su/gamma) -1)*gamma/(8*eta*Gamma), 0],
                       [0,np.sqrt(1+ (16*eta*Gamma*su/gamma) -1)*gamma/(8*eta*Gamma)]])

    dt = T/ppp
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
        dy = np.dot(C, x + np.dot(np.linalg.pinv(C), np.random.randn(2)/np.sqrt(dt)))*dt # signal
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

    os.makedirs(path+"{}/".format(itraj), exist_ok=True)
    np.save(path+"{}/means".format(itraj),np.array(means) )
    np.save(path+"{}/covs".format(itraj),np.array(covs) )
    np.save(path+"{}/signals".format(itraj),np.array(signals) )
    np.save(path+"{}/xicovs".format(itraj),np.array(xicovs) )
    np.save(path+"{}/D".format(itraj),D)
    np.save(path+"{}/C".format(itraj),C)
    np.save(path+"{}/dt".format(itraj),np.array([dt]))
    np.save(path+"{}/A".format(itraj),A)

    return means, covs, xicovs, signals, coeffs
