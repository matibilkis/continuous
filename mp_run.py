import os
import multiprocessing as mp
from numerics.utilities.misc import *


cores = mp.cpu_count()
total_time = 100.
dt = 1e-3

def int_seed(seed):
    for ext_signal in [1]:
        os.system("python3 numerics/integration/integrate.py --itraj {} --total_time {} --dt {} --ext_signal {}".format(seed, total_time, dt, ext_signal))
        print(f"{seed}, done")

with mp.Pool(cores-2) as p:
    p.map(int_seed, [1])
