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
parser.add_argument("--only_traj", type=int, default=0) #[eta, gamma, kappa, omega, n]


args = parser.parse_args()

periods = args.periods
ppp = args.ppp
itraj = args.itraj
rppp = args.rppp
method = args.method
params = args.params
only_traj = args.only_traj

rppp_reference = 1

params, exp_path = check_params(params)


export_dir = get_path_config(periods = periods, ppp= ppp, rppp=rppp, method=method, itraj=itraj, exp_path=exp_path  )+"figures/"
os.makedirs(export_dir,exist_ok=True)


 #### LOAD ""REAL"" trajectory

states, covs, signals, params, times = load_data(ppp=ppp, periods=periods, method=method, itraj=itraj, exp_path=exp_path , rppp = rppp_reference)
[eta, gamma, kappa, omega, n] = params
times_reference = times

### COMPUTE EXPECTRAL POWER
negs = 0
Period = 2*np.pi/params[-2]

fourier_signal = np.fft.fft(signals[:,0])
freqs_signal = np.fft.fftfreq(n = len(fourier_signal), d= Period/ppp)

filter_cond = freqs_signal>negs
freqs_signal = freqs_signal[filter_cond]
spectra_signal = np.abs(fourier_signal[filter_cond])**2

fourier_state = np.fft.fft(states[:,0])
freqs_state = np.fft.fftfreq(n = len(fourier_state), d= Period/ppp)

filter_cond = freqs_state>negs
freqs_state = freqs_state[filter_cond]
spectra_state = np.abs(fourier_state[filter_cond])**2

#### NOW PLOT TIME-WINDOWS IF avaialabel...
if only_traj != 1:

    ### LOAD LANDSCAPE COST RESULTS
    path_landscape= get_path_config(periods = periods, ppp= ppp, rppp=rppp_reference, method=method, itraj=itraj, exp_path = exp_path)+"landscape/"
    loss = np.load(path_landscape+"losses.npy")
    omegas_landscape = np.load(path_landscape+"omegas.npy")
    cuts_final_time = np.load(path_landscape+"cuts.npy")
    omega_looking = omega

    ### LOAD "STROBOSCOPIC REAL" trajectories (integrated with longer dt's, same noise)
    rossler_dt = {}
    windows = get_windows()
    rppps = [1] + list(windows)
    for rppp in rppps:
        rossler_dt[rppp], covs, signals, params, times = load_data(periods=periods, ppp=ppp, method="rossler", rppp = rppp, exp_path=exp_path)

    errs_rossler_strobo = {}
    for rppp in rossler_dt.keys():
        errs_rossler_strobo[rppp] = np.sqrt(np.mean(np.square(states[::rppp] - rossler_dt[rppp])))

    ########## load strobosocpic euler
    states_eu_opt = {}
    for indi, eu_rppp in enumerate(([1] + list(windows))):
        path_kalman_dt = get_path_config(periods = periods, ppp= ppp, rppp=rppp_reference, method=method, itraj=itraj, exp_path=exp_path)+"stroboscopic_euler_rppp{}/".format(eu_rppp)
        if indi==0:
            omegas = np.load(path_kalman_dt+"omegas.npy")
        try:
            states_eu_opt[eu_rppp] = np.load(path_kalman_dt+"states/states{}.npy".format(np.argmin(np.abs(np.array(omegas)-omega_looking))))
        except Exception:
            print("loading error in kalman dt..{}".format([indi, eu_rppp, "line 93", path_kalman_dt]))
            pass
    errs = {}
    for rdt in states_eu_opt.keys():
        errs[rdt] = np.sqrt(np.mean(np.square(states[::rdt] - states_eu_opt[rdt])))

#### PLOTING

fig = plt.figure(figsize=(40,40))

ls = 20
ts = 20
plt.rcParams.update({'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': ls,
         'axes.titlesize':ls,
         'xtick.labelsize':ts,
         'ytick.labelsize':ts}
)

gs = plt.GridSpec(nrows=6, ncols=6)# First axes
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

### plot spectral power

ax = fig.add_subplot(gs[0,4:6])
ax.plot(freqs_signal,spectra_signal)
ax.set_ylabel("|dy(f)|^2")
ax.set_xlabel("f")
#ax.axvline(params[-2]/(2*np.pi),color="black")
ax.set_xscale("log")
ax.set_yscale("log")

ax = fig.add_subplot(gs[1,4:6])
ax.plot(freqs_state,spectra_state)
#ax.axvline(params[-2]/(2*np.pi),color="black")
ax.set_ylabel("|x(f)|^2",size=20)
#ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.set_xlabel("f",size=20)
ax.set_xscale("log")
ax.set_yscale("log")

if only_traj != 1:
    print("plotting cost landscape!")
    ##### COST LANDSCAPE ###
    ax = fig.add_subplot(gs[2:4, 0:2])
    ax.set_title("cost landscape", size=20)
    for k, cut in enumerate(cuts_final_time):
        if (k%10 == 1) or (k == len(cuts_final_time)-1):
            ax.plot(omegas_landscape, loss[:,k],  '.-', label=times_reference[cut],linewidth=5)
        else:
            ax.plot(omegas_landscape, loss[:,k], '.-', linewidth=5)
    ax.set_xlabel(r'$\tilde{\omega}$')
    ax.legend(prop={"size":25})
    # ax.set_yscale("log")

    ax = fig.add_subplot(gs[4:6, 0:2])
    ax.set_title("cost landscape")
    ax.plot(omegas_landscape, loss[:,len(cuts_final_time)-1],'.-', color="red",linewidth=5, label=" T_long = {}".format(times_reference[cuts_final_time[-1]]))
    # ax.axvline(omega_looking,linewidth=3, color="black")
    ax.set_xlabel(r'$\tilde{\omega}$')
    ax.legend(prop={"size":25})

    ### ROSSLER STROBOSCOP
    ax = fig.add_subplot(gs[2:4, 2:6])
    ax.set_title("Errors Rossler integration - stroboscopic")
    ax.scatter(errs_rossler_strobo.keys(), np.abs(list(errs_rossler_strobo.values())), s=600)
    ax.plot(errs_rossler_strobo.keys(), np.abs(list(errs_rossler_strobo.values())), linewidth=5)

    ax.set_ylabel(r'$\sqrt{MSE}$')
    ax.set_xlabel(r'$window \; size$')
    # ax.set_xscale("log")
    # ax.set_yscale("log")


    #### KALMAN UPDATE ###
    ax = fig.add_subplot(gs[4:6, 2:6])
    ax.scatter(errs.keys(), np.abs(list(errs.values())),s=500)
    ax.plot(errs.keys(), np.abs(list(errs.values())), linewidth=5)
    # ax.set_yscale("log")
    # ax.set_xscale("log")
    ax.set_ylabel(r'$\sqrt{MSE}$'+"KALMAN UPDATE vs. rossler",size=20)
    ax.set_xlabel("multiplying factor in the integration step", size=20)




# plt.xticks(fontsize= 20)
# plt.yticks(fontsize= 20)
# plt.setp(ax.get_xticklabels(), fontsize=16)

short_path_fig = get_def_path()+"figures/{}periods_{}ppp_{}params/".format(periods, ppp, params)
path_fig = export_dir +"big_fig.png"

os.makedirs(short_path_fig,exist_ok=True)
print("saving plot in {}".format(path_fig))
plt.savefig(path_fig)
plt.savefig(short_path_fig+"img.png")
