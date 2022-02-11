import os
import numpy as np

windows = np.concatenate([(10**k)*np.arange(2,11) for k in range(1)])
method = "rossler"
ppp = 1000
periods = 1

##integrate time trace with different steps
for rppp in [1] + list(windows):
    os.system("python3 integrate.py --ppp {} --periods {} --rppp {} --method {}".format(ppp, periods, rppp, method))

# how does the landscape looks for the big time trace?
os.system("python3 landscape_cost.py --ppp {} --periods {} --rppp {}".format(ppp,periods,1))

# ## kalman inte
for eu_rppp in [1] + list(windows):
    os.system("python3 kalman_integration_step.py --periods {} --ppp {} --rppp 1 --euler_rppp {}".format(periods, ppp, eu_rppp))
