import os
import numpy as np
from misc import *

#####

params = give_def_params() #params = [eta, gamma, kappa, omega, n]

windows = np.concatenate([(10**k)*np.arange(2,11) for k in range(2)])
windows= []

save_windows(windows)
method = "rossler"
ppp = 10000
periods = 10
rppp_reference = 1


only_plot = 1
only_traj=0
no_kalman = 1
seed = 0
#integrate time trace with different steps

#
# if only_plot != 1:
#     for rppp in [1] + list(windows):
#         print("INTEGRATING TIME-TRACE, stroboscopic factor  x{}".format(rppp))
#         os.system("python3 integrate.py --ppp {} --periods {} --rppp {} --method {} --params {} --itraj {}".format(ppp, periods, rppp, method,  params_to_string(params), seed))
#
#     if only_traj != 1:
#         # how does the landscape looks for the big time trace?
#         print("LOOKING AT COST LANDSCAPE!")
#         os.system("python3 landscape_cost.py --ppp {} --periods {} --rppp {} --params {} --itraj {}".format(ppp,periods,rppp_reference, params_to_string(params), seed))
#
#         if no_kalman != 1:
#             ## kalman inte
#             for eu_rppp in [1]+ list(windows):
#                 print("TRACKING WITH EULER stroboscopic factor x{}".format(eu_rppp))
#                 os.system("python3 kalman_integration_step.py --periods {} --ppp {} --rppp 1 --euler_rppp {} --params {}".format(periods, ppp, eu_rppp, params_to_string(params), seed))

os.system("python3 plot.py --periods {} --ppp {} --params {} --rppp {} --only_traj {} --no_kalman {} --itraj {}".format(periods, ppp, params_to_string(params), rppp_reference, only_traj, no_kalman ,seed))
