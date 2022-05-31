
# import os
# import multiprocessing as mp
# from numerics.utilities.misc import *
#
#
# cores = mp.cpu_count()
# mode="damping"
# dt = 1e-6
# total_time = 4.
# ppp=5*1e3
# mode "frequencies"
# total_time = 50.
#
# def int_seed(seed):
#     for k in range(10):
#         os.system("python3 numerics/integration/integrate.py --itraj {} --mode {} --dt {} --total_time {} --ppp {}".format(seed+k, mode, dt, total_time, ppp))
#         os.system("python3 numerics/integration/integrate.py --itraj {} --flip_params 1 --mode {} --dt {} --total_time {} --ppp {}".format(seed+k, mode, dt, total_time, ppp))
#         print(f"{k}, {seed}, done")
#
#
# with mp.Pool(cores-2) as p:
#     p.map(int_seed, range(1,1000, 10))





import os
import multiprocessing as mp
from numerics.utilities.misc import *


cores = mp.cpu_count()

mode="frequencies"
dt = 1e-4
total_time = 200.
ppp=5*1e3
def int_seed(seed):
    for k in range(10):
        os.system("python3 numerics/integration/integrate.py --itraj {} --mode {} --dt {} --total_time {} --ppp {}".format(seed+k, mode, dt, total_time, ppp))
        os.system("python3 numerics/integration/integrate.py --itraj {} --flip_params 1 --mode {} --dt {} --total_time {} --ppp {}".format(seed+k, mode, dt, total_time, ppp))
        print(f"{k}, {seed}, done")


with mp.Pool(cores-4) as p:
    p.map(int_seed, range(1,1000, 10))
