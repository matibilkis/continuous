import os
import numpy as np
from misc import *

#####

params = give_def_params() #params = [eta, gamma, kappa, omega, n]
method = "rossler"
ppp = 1000
periods = 100
rppp_reference = 1

os.system("python3 train.py --ppp {} --periods {} --rppp {} --params {}".format(ppp,periods,rppp_reference, params_to_string(params)))

#os.system("python3 train.py --periods {} --ppp {} --method {} --itraj {} --rppp {} --trainid {} --params {}")
