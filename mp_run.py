import os
import multiprocessing as mp
from numerics.utilities.misc import *
params = give_def_params() #params = [eta, gamma, kappa, omega, n]

method = "rossler"
ppp = 1000
periods = 10

cores = mp.cpu_count()

def int_seed(seed, ppp=1000, periods=10, method="rossler"):
    os.system("python3 numerics/integration/integrate_with_fisher.py --ppp {} --periods {} --method {} --params {} --itraj {}".format(ppp, periods, method,  params_to_string(params), seed))### default rppp = 1

jobs = 10000
with mp.Pool(cores-2) as p:
    p.map(int_seed, range(1001,jobs))
