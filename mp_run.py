import os
import multiprocessing as mp
from numerics.utilities.misc import *

method = "rossler"

cores = mp.cpu_count()

def int_seed(seed, ppp=1000, periods=50, method="rossler"):
    os.system("python3 numerics/integration/integrate.py --ppp {} --periods {} --itraj {}".format(ppp,periods, seed))### default rppp = 1

jobs = 1000
with mp.Pool(cores-2) as p:
    p.map(int_seed, range(1000, 1000+jobs))
