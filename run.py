import os
import numpy as np
import sys
from numerics.utilities.misc import *
import argparse
#####

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--mode", type=str, default="damping")
parser.add_argument("--dt", type=float, default=1e-4)
parser.add_argument("--total_time", type=float, default=10)

args = parser.parse_args()

seed = args.seed
mode = args.mode
dt = args.dt
total_time = args.total_time

for k in range(100):
    os.system("python3 numerics/integration/integrate.py --itraj {} --mode {} --dt {} --total_time {}".format(seed+k, mode, dt, total_time))
    os.system("python3 numerics/integration/integrate.py --itraj {} --flip_params 1 --mode {} --dt {} --total_time {}".format(seed+k, mode, dt, total_time))
