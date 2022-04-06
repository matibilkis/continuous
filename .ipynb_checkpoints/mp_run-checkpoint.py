import os
import multiprocessing as mp
from numerics.utilities.misc import *


cores = mp.cpu_count()

mode="damping"
dt = 1e-4
total_time = 3.
def int_seed(seed):
    for k in range(10):
        os.system("python3 numerics/integration/integrate.py --itraj {} --mode {} --dt {} --total_time {}".format(seed+k, mode, dt, total_time))
        os.system("python3 numerics/integration/integrate.py --itraj {} --flip_params 1 --mode {} --dt {} --total_time {}".format(seed+k, mode, dt, total_time))
        print(f"{k}, {seed}, done")


jobs = 2
with mp.Pool(cores-2) as p:
    p.map(int_seed, range(100,1000, 10))
