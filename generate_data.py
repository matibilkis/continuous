import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from misc import *
import tensorflow as tf



covs = []
means = []
signals = []
xicovs = []

#define parameters
gamma = 1.0 #damping from outside
Gamma = 0.8 #measurement rate
eta = 1 # measurement efficiency
n = 2 # number of photons?
w = 0.4 # hamornic oscillator freq mecanical
m = 1 # mass harmonic oscillator mechanical
T = 2*np.pi/w

C= np.array([[np.sqrt(4*eta*Gamma), 0] ,[0, np.sqrt(4*eta*Gamma)]])

A = np.array([
    [0., 1/m],
    [-m*w**2, 0.]])

D = np.array([[gamma*(n + 0.5) + Gamma, 0], [0,gamma*(n + 0.5) + Gamma]])

## initial condition
su = n + 0.5 + Gamma/gamma
cov_in = np.array([[np.sqrt(1+ (16*eta*Gamma*su/gamma) -1)*gamma/(8*eta*Gamma), 0],
                   [0,np.sqrt(1+ (16*eta*Gamma*su/gamma) -1)*gamma/(8*eta*Gamma)]])
x_in = np.array([1,0])



dt = 5e-4
tfinal = 20
tot_steps = int(tfinal/dt)

xi = lambda cov: np.dot(cov, ct(C)) #+ ct(Gamma_matrix)

covs = [cov_in]
means = [x_in]
xicovs = [xi(covs[-1])]
for k in tqdm(range(tot_steps)):

    x = means[-1]
    cov = covs[-1]
    XiCov = xicovs[-1]

    dy = np.dot(C, x + np.dot(np.linalg.inv(C), np.random.randn(2)/np.sqrt(dt)))*dt # signal
    signals.append(dy)

    dx = np.dot(A - np.dot(XiCov,C), x)*dt + np.dot(XiCov, dy)  #evolution update (according to what you measure)
    dcov = (np.dot(A,cov) + np.dot(cov, ct(A)) + D - np.dot(XiCov, ct(XiCov)))*dt  #covariance update

    covs.append(covs[-1] + dcov)
    means.append(means[-1] + dx)
    xicovs.append(xi(covs[-1]))
