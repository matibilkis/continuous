import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from misc import *
import argparse
import os
from integrate import *

defpath = get_def_path()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--ppp", type=int, default=500) ###points per period
parser.add_argument("--periods", type=int, default=5)
parser.add_argument("--path", type=str, default=defpath) #
parser.add_argument("--itraj", type=int, default=1)
parser.add_argument("--method", type=str, default="RK")

args = parser.parse_args()

periods = args.periods
ppp = args.ppp
path = args.path
itraj = args.itraj
method = args.method

path = path+"{}periods/{}ppp/".format(periods,ppp)
### INTEGRATE TRAJ
if method == "RK":
    generate_traj_RK(ppp=ppp, periods = periods, itraj=itraj, path = path, seed=itraj)
else:
    generate_traj_Euler(ppp=ppp, periods = periods, itraj=itraj, path = path, seed=itraj)
