import os
import numpy as np
from misc import params_to_string



#####


params = give_def_params() #params = [eta, gamma, kappa, omega, n]

windows = np.concatenate([(10**k)*np.arange(2,11) for k in range(1)])
method = "rossler"
ppp = 1000
periods = 10

##integrate time trace with different steps
# for rppp in [1]:# + list(windows):
#     os.system("python3 integrate.py --ppp {} --periods {} --rppp {} --method {} --params {}".format(ppp, periods, rppp, method,  params_to_string(params)))
#
# # how does the landscape looks for the big time trace?
# os.system("python3 landscape_cost.py --ppp {} --periods {} --rppp {} --params {}".format(ppp,periods,rppp, params_to_string(params)))
# #
# # # ## kalman inte
# for eu_rppp in [1]:# + list(windows):
#     os.system("python3 kalman_integration_step.py --periods {} --ppp {} --rppp 1 --euler_rppp {} --params {}".format(periods, ppp, eu_rppp, params_to_string(params)))

os.system("python3 plot.py --periods {} --ppp {} --params {}".format(periods, ppp, params_to_string(params)))
