import os
import numpy as np
import sys
from numerics.utilities.misc import *
import argparse
#####

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

params = give_def_params() #params = [eta, gamma, kappa, omega, n]

method = "rossler"
ppp = 1000
periods = 10
seed = args.seed

# os.system("python3 numerics/integration/integrate_with_fisher.py --ppp {} --periods {} --method {} --params {} --itraj {}".format(ppp, periods, method,  params_to_string(params), seed))### default rppp = 1
for train_id in range(3,10):
    print("train_id {}".format(train_id))
    os.system("python3 numerics/machine_learning/train.py --ppp {} --periods {} --params {} --itraj {} --trainid {}".format(ppp,periods, params_to_string(params), seed, train_id))### default rppp = 1
