import matplotlib.pyplot as plt
from misc import *
import argparse
import os

defpath = get_def_path()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--ppp", type=int) ###points per period
parser.add_argument("--periods", type=int)
parser.add_argument("--path", type=str, default=defpath) #
parser.add_argument("--itraj", type=int, default=1)
parser.add_argument("--rppp", type=int, default=1)
parser.add_argument("--method", type=str, default="rossler")
parser.add_argument("--params", type=str, default="") #[eta, gamma, kappa, omega, n]

args = parser.parse_args()

periods = args.periods
ppp = args.ppp
path = args.path
itraj = args.itraj
rppp = args.rppp
method = args.method
params = args.params

params, exp_path = check_params(params)
# omega = 2*np.pi
# n = 14
# gamma = (4*np.pi*265/29)*1e-6
# kappa = 4*np.pi*0.36*1e-3
# eta = 0.74



export_dir = get_path_config(periods = periods, ppp= ppp, rppp=rppp, method=method, itraj=itraj)+"figures/"
os.makedirs(export_dir,exist_ok=True)


############# load cost landscape #####

for rppp in [1]:
    states, covs, signals, params, times = load_data(ppp=ppp, periods=periods, method=method, itraj=itraj, exp_path=exp_path , rppp = 1)
    [eta, gamma, kappa, omega, n] = params

path_landscape= get_path_config(periods = periods, ppp= ppp, rppp=rppp, method=method, itraj=itraj, exp_path = exp_path)+"landscape/"
loss = np.load(path_landscape+"losses.npy")
omegas = np.load(path_landscape+"omegas.npy")
cuts_final_time = np.load(path_landscape+"cuts.npy")
omega_looking = omega


########## load strobosocpic euler
windows = np.concatenate([(10**k)*np.arange(2,11) for k in range(1)])
states_eu_opt = {}
for indi, eu_rppp in enumerate(([1] + list(windows))[:10]):
    path_kalman_dt = get_path_config(periods = periods, ppp= ppp, rppp=rppp, method=method, itraj=itraj, exp_path=exp_path)+"stroboscopic_euler_rppp{}/".format(eu_rppp)
    if indi==0:
        omegas = np.load(path_kalman_dt+"omegas.npy")
    try:
        states_eu_opt[eu_rppp] = np.load(path_kalman_dt+"states/states{}.npy".format(np.argmin(np.abs(np.array(omegas)-omega_looking))))
    except Exception:
        pass
errs = {}
for rdt in states_eu_opt.keys():
    errs[rdt] = np.sqrt(np.mean(np.square(states[::rdt] - states_eu_opt[rdt])))




#### PLOTING

fig = plt.figure(figsize=(20,20))
gs = plt.GridSpec(nrows=6, ncols=4)# First axes
plt.suptitle("[eta, gamma, kappa, omega, n]= "+str( params)+"\n longest time trace: \n periods {} ppp {}".format(periods,ppp), size=20)

for i in range(2):
    for k in range(2):
        #### TRAJECTORIES
        ax = fig.add_subplot(gs[i, k])
        ax.plot(covs[:,i,k][:10000])
        ax.set_title("cov[{},{}]".format(i,k))

        ax = fig.add_subplot(gs[i, k+2])
        if i%2==0:
            ax.plot(states[:,0])
            ax.set_title("states")
        else:
            ax.plot(signals[:,0])
            ax.set_title("signals")
        if k%2==0:
            ax.set_xscale("log")

##### COST LANDSCAPE ###
ax = fig.add_subplot(gs[2:4, 0:2])
for k, cut in enumerate(cuts_final_time[10:]):
    ax.plot(omegas, loss[:,k], label=times[cut])
ax.set_xlabel(r'$\tilde{\omega}$')
ax.set_ylabel("cost landscape")
ax.set_yscale("log")

ax = fig.add_subplot(gs[4:6, 0:2])
ax.set_title("cost landscape \nlongest time trace")
ax.plot(loss[:,-1])
ax.set_xlabel(r'$\tilde{\omega}$')



#### KALMAN UPDATE ###
ax = fig.add_subplot(gs[2:4, 2:4])
ax.set_title("KALMAN UPDATE \noriginal\nperiods {}\n ppp {}".format(periods, ppp))
ax.scatter(errs.keys(), errs.values(),s=100)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel(r'$\sqrt{MSE}$'+"wrt rossler",size=20)
ax.set_xlabel("multiplying factor in the integration step", size=20)

path_fig = export_dir +"big_fig.png"
short_path_fig = get_def_path()+"figures/parms{}_{}itraj{}_real_traj_method{}periods{}ppp{}rppp/".format(params,itraj, method, periods, ppp, rppp)

print("saving plot in {}".format(path_fig))
plt.savefig(path_fig)
os.makedirs(short_path_fig,exist_ok=True)
os.system("cp {} {}".format(path_fig, short_path_fig))
