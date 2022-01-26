import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from misc import *
import argparse
import os
from integrate import *
import pickle

defpath = get_def_path()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--ppp", type=int, default=500) ###points per period
parser.add_argument("--periods", type=int, default=5)
parser.add_argument("--path", type=str, default=defpath) #
parser.add_argument("--itraj", type=int, default=1)

args = parser.parse_args()

periods = args.periods
ppp = args.ppp
path = args.path
itraj = args.itraj
method ="rossler"

path = path+"{}periods/{}ppp/".format(periods,ppp)
### INTEGRATE TRAJ
# integrate(periods, ppp, method="rossler", itraj=itraj, path="")


print("traj integrated")

states, covs, signals, params, times = load_data(ppp=ppp, periods=periods, method="rossler")
#states, covs, signals, [A,dt,C,D], params = load_data(periods=periods, ppp=ppp, itraj=itraj,method="RK4")
eta, gamma, Lambda, omega, n = params
[A,C,D] = build_matrix_from_params(params)


xi = lambda cov,D: np.dot(cov, ct(C)) + ct(D)

def evolve_simu_state(x,xE ,sigma, dy, simu_A, simu_exp_A, dt=1):
    cov = sigma
    XiCov = xi(cov, D)
    dx = np.dot(simu_A-np.dot(XiCov,C),x)*dt   + np.dot(XiCov,dy)#
    dxE = np.dot(simu_exp_A-np.eye(2), xE) - np.dot(np.dot(XiCov,C), xE)*dt + np.dot(XiCov, dy)
    dcov = (np.dot(simu_A,cov) + np.dot(cov, ct(simu_A)) + D - np.dot(XiCov, ct(XiCov)))*dt  #covariance update
    return [x + dx, cov + dcov, xE + dxE]



sstates, sstatesE, scovs = {}, {} ,{}
epo = np.pi/2
omegas = np.arange(0,4*np.pi+epo, epo)
gammas = [gamma]
dt = 1/ppp
cuts = [int(k) for k in np.logspace(1,np.log10(len(times)-1), 10)]
loss = np.zeros((len(omegas), len(gammas), len(cuts)))
for indomega,omega in tqdm(enumerate(omegas)):
    for indgamma,gamma in enumerate(gammas):

        simu_exp_A = np.array([[np.cos(omega*dt), np.sin(omega*dt)], [-np.sin(omega*dt), np.cos(omega*dt)]])*np.exp(-gamma*dt/2)
        simu_A = np.array([[-.5*gamma, omega], [-omega, -0.5*gamma]])

        sstates[str([omega, gamma])] = [states[0]]
        sstatesE[str([omega, gamma])] = [states[0]]
        scovs[str([omega, gamma])] = [covs[0]]

        for ind,dy in enumerate(tqdm(signals)):
            simu = evolve_simu_state(sstates[str([omega, gamma])][-1], sstatesE[str([omega, gamma])][-1], scovs[str([omega, gamma])][-1], dy, simu_A, simu_exp_A,  dt=dt)
            sstates[str([omega, gamma])].append(simu[0])
            scovs[str([omega, gamma])].append(simu[1])
            sstatesE[str([omega, gamma])].append(simu[2])

        for indcut, cut in enumerate(cuts):
            loss[indomega,indgamma, indcut] = np.sum(np.square(signals[:cut] - np.einsum('ij,bj->bi',C,sstates[str([omega,gamma])][:-1][:cut])*dt))/(2*times[cut])

path_landscape=get_def_path()+"{}periods/{}ppp/{}/cost_landscape/".format(periods,ppp,itraj)
os.makedirs(path_landscape,exist_ok=True)
np.save(path_landscape+"losses",loss)
np.save(path_landscape+"omegas",omegas)
np.save(path_landscape+"gammas",gammas)
np.save(path_landscape+"cuts",cuts)

with open(path_landscape+"sstates.pickle","wb") as f:
    pickle.dump(sstates,f, protocol=pickle.HIGHEST_PROTOCOL)

#
# for i,k in sstates.items():
#     np.save(path_landscape+str(i),np.array(k), allow_pickle=True)

# loss = {}
#
#
# with open(path_landscape+"losses.pickle","wb") as ff:
#     pickle.dump(loss,ff, protocol=pickle.HIGHEST_PROTOCOL)
# for i,k in loss.items():
#     np.save(path_landscape+str(i)+"loss",np.array(k), allow_pickle=True)
#
#

# landscape = {}
# cut_series = [int(k) for k in np.logspace(2,np.log10(len(signals)),10)]
# for length_series in tqdm(cut_series):
#     losses = []
#     for i in range(len(parameters)):
#         losses.append(np.sum(np.square(np.array(preds[i])[:length_series] - signals[:length_series]))/(2*dt*length_series))
#     landscape[length_series] = losses
#
# os.makedirs(path_landscape,exist_ok=True)
# np.save(path_landscape+method,list(landscape.values()), allow_pickle=True)
#
# print("landscapes done")
# plt.figure(figsize=(20,7))
# colors = plt.get_cmap("rainbow")
# ax = plt.subplot2grid((1,1),(0,0))
# for ind,p in enumerate(landscape.values()):
#     plt.plot(parameters,p, color=colors(np.linspace(0,1,len(landscape)))[ind], marker='.', label="{}".format(int(np.round(cut_series[ind]+1,0))),linewidth=3)
# plt.legend()
# plt.xlabel(r'$\tilde{\omega}$',size=30)
# plt.ylabel(r'$C(\omega, \tilde{\omega})$',size=30)
# plt.savefig(path_landscape+"{}.pdf".format(method))
# plt.savefig(path_landscape+"{}.png".format(method))
# #
