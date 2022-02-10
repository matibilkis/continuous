import os
import numpy as np

windows = np.concatenate([(10**k)*np.arange(2,11) for k in range(6)])
method = "rossler"
for rppp in [1] + list(windows):
    os.system("python3 integrate_convergence.py --ppp 100000 --periods 100 --r_ppp {} --method {}".format(rppp, method))
