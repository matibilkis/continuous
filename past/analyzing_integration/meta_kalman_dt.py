import os
import numpy as np

windows = np.concatenate([(10**k)*np.arange(2,11) for k in range(3)])
method = "rossler"
ppp = 10000
periods = 100
for eu_rppp in [1] + list(windows):
    os.system("python3 kalman_integration_step.py --periods {} --ppp {} --rppp 1 --euler_rppp {}".format(periods, ppp, eu_rppp))
