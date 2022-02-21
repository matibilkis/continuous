import os
import numpy as np
from misc import *
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

#os.system("python3 integrate_fisher.py --ppp {} --periods {} --method {} --params {} --itraj {}".format(ppp, periods, method,  params_to_string(params), seed))### default rppp = 1
os.system("python3 train.py --ppp {} --periods {} --params {} --itraj {}".format(ppp,periods, params_to_string(params), seed))### default rppp = 1
