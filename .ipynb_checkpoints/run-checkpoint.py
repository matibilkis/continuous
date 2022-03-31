import os
import numpy as np
import sys
from numerics.utilities.misc import *
import argparse
#####

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--mode", type=str, default="damping")

args = parser.parse_args()

seed = args.seed
mode = args.mode
#mode = "frequencies"
#mode = "damping"
### damping discrimination
###total_time = 4, dt = 1e-6

dt = 1e-6
for k in range(10):
    os.system("python3 numerics/integration/integrate.py --itraj {} --mode {} --dt {}".format(seed+k, mode, dt))
    os.system("python3 numerics/integration/integrate.py --itraj {} --flip_params 1 --mode {} --dt {}".format(seed+k, mode, dt))
