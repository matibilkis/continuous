import argparse
import numpy as np
from misc import *
from tqdm import tqdm
import os 


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
seed = args.seed

ppp = 500
periods = 40


gamma = 1 #damping from outside
Gamma = 1 #measurement rate
eta = 1 # measurement efficiency
n = 2 # number of photons?

w = 2*np.pi
T = (2*np.pi)/w


C = np.array([[np.sqrt(4*eta*Gamma), 0] ,[0, np.sqrt(4*eta*Gamma)]])

A = np.array([
    [0., w],
    [-w, 0.]])

D = np.array([[gamma*(n + 0.5) + Gamma, 0], [0,gamma*(n + 0.5) + Gamma]])

su = n + 0.5 + Gamma/gamma
cov_in = np.array([[np.sqrt(1+ (16*eta*Gamma*su/gamma) -1)*gamma/(8*eta*Gamma), 0],
                   [0,np.sqrt(1+ (16*eta*Gamma*su/gamma) -1)*gamma/(8*eta*Gamma)]])


xi = lambda cov: np.dot(cov, ct(C)) + ct(D)

parameters = np.linspace(-1,1,101) + 2*np.pi

give_pred = lambda state: np.dot(C,state)*dt
def evolve_state(states, AA, dy):
    x, cov = states
    XiCov = xi(cov)
    dx = np.dot(AA - np.dot(XiCov,C), x)*dt + np.dot(XiCov, dy)  #evolution update (according to what you measure)
    dcov = (np.dot(AA,cov) + np.dot(cov, ct(AA)) + D - np.dot(XiCov, ct(XiCov)))*dt  #covariance update
    return [x + dx, cov + dcov]

itraj = 0 
np.random.seed(seed)
landscape = {}
for periods in [1,5,10,30,40]:

    dt = T/ppp
    total_points = int(periods*ppp)

    signals = []

    covs = [cov_in]
    means = [np.array([1.,0.])] ## initial condition
    xicovs = [xi(covs[-1])]

    symplectic = np.array([[0,1],[-1,0]])
    predictions, states = {t:[] for t in range(len(parameters))},{t:[[means[0], covs[0]]] for t in range(len(parameters))}
    signals=[]

    for k in tqdm(range(total_points)):
        x = means[-1]
        cov = covs[-1]
        XiCov = xi(cov)

        dy = np.dot(C, x + np.dot(np.linalg.pinv(C), np.random.randn(2)/np.sqrt(dt)))*dt # signal
        signals.append(dy)

        dx = np.dot(A - np.dot(XiCov,C), x)*dt + np.dot(XiCov, dy)  #evolution update (according to what you measure)
        dcov = (np.dot(A,cov) + np.dot(cov, ct(A)) + D - np.dot(XiCov, ct(XiCov)))*dt  #covariance update

        covs.append(covs[-1] + dcov)
        means.append(means[-1] + dx)

        for i in range(len(parameters)):
            predictions[i].append(give_pred(states[i][-1][0]))
            states[i].append(evolve_state(states[i][-1], parameters[i]*symplectic, dy))

    means = np.array(means)
    covs = np.array(covs)
    xicovs = np.array(xicovs)
    signals = np.array(signals)
    coeffs = [C, A, D , dt]

    loss = {}

    for i in range(len(parameters)):
        loss[i] = np.sum(np.square(np.array(predictions[i]) - np.array(signals)))/(2*dt*len(predictions[i]))
    landscape[periods] = list(loss.values())

    path = get_def_path()
    path = path + "landscape/{}/{}periods/{}ppp/".format(seed,periods,ppp)
    os.makedirs(path+"{}/".format(itraj), exist_ok=True)
    np.save(path+"{}/means".format(itraj),np.array(means) )
    np.save(path+"{}/covs".format(itraj),np.array(covs) )
    np.save(path+"{}/signals".format(itraj),np.array(signals) )
    np.save(path+"{}/xicovs".format(itraj),np.array(xicovs) )
    np.save(path+"{}/D".format(itraj),D)
    np.save(path+"{}/C".format(itraj),C)
    np.save(path+"{}/dt".format(itraj),np.array([dt]))
    np.save(path+"{}/A".format(itraj),A)
    for k,p in loss.items():
        np.save(path+"loss_{}".format(k), p)
    for k,p in predictions.items():
        np.save(path+"preds_{}".format(k), p)
