import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from misc import *
import argparse
import os
from integrate import generate_traj

defpath = get_def_path()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--ppp", type=int, default=1000) ###points per period
parser.add_argument("--periods", type=int, default=5)
parser.add_argument("--path", type=str, default=defpath) #
parser.add_argument("--itraj", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()

periods, ppp, path, itraj, seed = int(float(args.periods)), args.ppp, args.path, int(float(args.itraj)), args.seed
path = path+"{}periods/{}ppp/".format(periods,ppp)

means, covs, xicovs, signals, coeffs = generate_traj(ppp=ppp, periods = periods, itraj=itraj, path = path, seed=seed) #
C, A, D , dt = coeffs

give_pred = lambda state: np.dot(C,state)*dt
def evolve_state(states, AA, dy):
    x, cov = states
    XiCov = xi(cov)
    dx = np.dot(AA - np.dot(XiCov,C), x)*dt + np.dot(XiCov, dy)  #evolution update (according to what you measure)
    dcov = (np.dot(AA,cov) + np.dot(cov, ct(AA)) + D - np.dot(XiCov, ct(XiCov)))*dt  #covariance update
    return [x + dx, cov + dcov]