from misc import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--itraj", type=int, default=0)
parser.add_argument("--periods",type=int, default=40)
parser.add_argument("--ppp", type=int,default=500)

args = parser.parse_args()
itraj = args.itraj
periods = args.periods
ppp = args.ppp

means, covs, signals, coeffs = load_data(get_def_path()+"{}periods/{}ppp/".format(periods,ppp), itraj=itraj)
A, dt, C, D  = coeffs
xi = lambda cov: np.dot(cov, ct(C)) + ct(D)


symplectic = np.array([[0,1],[-1,0]])
parameters = np.array([2*np.pi]) + np.linspace(-2,2,41)
predictions, states = {t:[] for t in range(len(parameters))},{t:[means[0], covs[0]] for t in range(len(parameters))}

give_pred = lambda state: np.dot(C,state)*dt
def evolve_state(states, AA, dy):
    x, cov = states
    XiCov = xi(cov)
    dx = np.dot(AA - np.dot(XiCov,C), x)*dt + np.dot(XiCov, dy)  #evolution update (according to what you measure)
    dcov = (np.dot(AA,cov) + np.dot(cov, ct(AA)) + D - np.dot(XiCov, ct(XiCov)))*dt  #covariance update
    return [x + dx, cov + dcov]


for dy in tqdm(signals):
    for i in range(len(parameters)):
        predictions[i].append(give_pred(states[i][-1][0]))
        states[i].append(evolve_state(states[i][-1], parameters[i]*symplectic, dy))


optimal_param, loss_val = {}, {}
cut_series = np.logspace(2,np.log10(len(signals)),20)
for length_series in tqdm(cut_series):
    losses = []
    length_series = int(length_series)
    sigs = signals[:length_series]

    for i in range(len(parameters)):
        preds = np.array(predictions[i])[:length_series]
        losses.append(np.sum(np.square(preds - sigs))/(2*dt*length_series))
    optimal_param[length_series] = parameters[np.argmin(losses)]
    loss_val[length_series] = np.min(losses)


convergence_path = get_def_path()+"{}periods/{}ppp/convergence/{}/".format(periods,ppp, itraj)
os.makedirs(convergence_path, exist_ok=True)
np.save(convergence_path+"optimal_parameters",np.array(list(optimal_param.values())))
np.save(convergence_path+"loss_evolution",list(loss_val.values()))
