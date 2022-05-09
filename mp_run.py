import os
import multiprocessing as mp
from numerics.utilities.misc import *


cores = mp.cpu_count()
total_time = 4.
dt = 1e-3

def int_seed(seed):
    os.system("python3 numerics/integration/integrate.py --itraj {} --total_time {} --dt {}".format(seed, total_time, dt))
    print(f"{k}, {seed}, done")

with mp.Pool(cores-2) as p:
    p.map(int_seed, range(1,10, 1))
