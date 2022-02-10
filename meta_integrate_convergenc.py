import os
import numpy as np

windows = np.concatenate([(10**k)*np.arange(2,11) for k in range(4)])
method = "rossler"
ppp = 10000
periods = 100
for rppp in [1] + list(windows):
    os.system("python3 integrate_convergence.py --ppp {} --periods {} --rppp {} --method {}".format(ppp, periods, rppp, method))
