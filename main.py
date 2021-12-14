import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from misc import *
import tensorflow as tf
import argparse
import os

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--ppp", type=int, default=100) ###points per period
parser.add_argument("--periods", type=int, default=20)
parser.add_argument("--path", type=str, default="/data/uab-giq/scratch/matias/quantera/trajectories/")
parser.add_argument("--itraj", type=int, default=0)

args = parser.parse_args()
periods, ppp, path, itraj = args.periods, args.ppp, args.path, int(args.itraj)

#define parameters
gamma = 1 #damping from outside
Gamma = 1 #measurement rate
eta = 1 # measurement efficiency
n = 2 # number of photons?
w = 1
T = 2*np.pi/w

C = np.array([[np.sqrt(4*eta*Gamma), 0] ,[0, np.sqrt(4*eta*Gamma)]])

A = np.array([
    [0., w],
    [-w, 0.]])

D = np.array([[gamma*(n + 0.5) + Gamma, 0], [0,gamma*(n + 0.5) + Gamma]])

su = n + 0.5 + Gamma/gamma
cov_in = np.array([[np.sqrt(1+ (16*eta*Gamma*su/gamma) -1)*gamma/(8*eta*Gamma), 0],
                   [0,np.sqrt(1+ (16*eta*Gamma*su/gamma) -1)*gamma/(8*eta*Gamma)]])

dt = T/ppp
total_points = int(T*periods/dt)

xi = lambda cov: np.dot(cov, ct(C)) #+ ct(Gamma_matrix)

signals = []
covs = [cov_in]
means = [np.array([1,0])] ## initial condition
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

os.makedirs("/data/uab-giq/scratch/matias/quantera/trajectories/{}/".format(itraj), exist_ok=True)
np.save("/data/uab-giq/scratch/matias/quantera/trajectories/{}/means".format(itraj),np.array(means) )
np.save("/data/uab-giq/scratch/matias/quantera/trajectories/{}/covs".format(itraj),np.array(covs) )
np.save("/data/uab-giq/scratch/matias/quantera/trajectories/{}/signals".format(itraj),np.array(signals) )
np.save("/data/uab-giq/scratch/matias/quantera/trajectories/{}/xicovs".format(itraj),np.array(xicovs) )
np.save("/data/uab-giq/scratch/matias/quantera/trajectories/{}/D".format(itraj),D)
np.save("/data/uab-giq/scratch/matias/quantera/trajectories/{}/C".format(itraj),C)
np.save("/data/uab-giq/scratch/matias/quantera/trajectories/{}/dt".format(itraj),np.array([dt]))
np.save("/data/uab-giq/scratch/matias/quantera/trajectories/{}/A".format(itraj),A)
