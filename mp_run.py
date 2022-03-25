import os
import multiprocessing as mp
from numerics.utilities.misc import *

method = "rossler"

cores = mp.cpu_count()

def int_seed(seed, periods=2, method="rossler"):
    os.system("python3 numerics/integration/integrate.py --itraj {}".format(seed))### default rppp = 1
    os.system("python3 numerics/integration/integrate.py --itraj {} --h1true 1".format(seed))### default rppp = 1

jobs = 1000
with mp.Pool(cores-2) as p:
    p.map(int_seed, range(0, jobs))
