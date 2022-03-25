import os
import numpy as np
import sys
from numerics.utilities.misc import *
import argparse
#####

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--seed", type=int, default=10)
args = parser.parse_args()

seed = args.seed

os.system("python3 numerics/integration/integrate.py --itraj {}".format(seed))### default rppp = 1
os.system("python3 numerics/integration/integrate.py --itraj {} --h1true 1".format(seed))### default rppp = 1

# os.system("python3 numerics/integration/integrate_with_fisher.py --ppp {} --periods {} --method {} --params {} --itraj {}".format(ppp, periods, method,  params_to_string(params), seed))### default rppp = 1
# for itraj in range(10):
    # print("train_id {}".format(itraj))
    # os.system("python3 numerics/integration/integrate.py --dt {} --total_time {} --itraj {}".format(dt,periods, itraj))### default rppp = 1
