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

periods = args.periods
ppp = args.ppp
path = args.path
itraj = args.itraj
seed = args.seed

path = path+"{}periods/{}ppp/".format(periods,ppp)

generate_traj(ppp=ppp, periods = periods, itraj=itraj, path = path, seed=seed) #
