import numpy as np
import ast
import os

def ct(A):
    return np.transpose(np.conjugate(A))

def get_time(total_time, dt):
    return np.arange(0, total_time+dt, dt)

def get_def_path(mode="ML_genoni/"):
    import getpass
    try:
        user = getpass.getuser()
    except Exception:
        user = "giq"
    if user == "cooper-cooper":
        defpath = '../quantera/trajectories/'
    elif (user =="matias") or (user == "mati"):# or (user=="giq"):
        defpath = '../quantera/trajectories/'
    elif (user=="giq"):
        defpath = "/media/giq/Nuevo vol/quantera/trajectories/"
    else:
        defpath = "/data/uab-giq/scratch/matias/quantera/trajectories/"
    if mode[-1] != "/":
        mode+="/"
    defpath+=mode
    return defpath

def def_params():
    kappa = 1
    xi =0. #.1*kappa
    omega = 0.*kappa
    eta = 1
    params = [xi, kappa, omega, eta]
    exp_path = '{}/'.format(params)
    return params, exp_path

def get_path_config(total_time=10,dt=1e-3,itraj=1,exp_path="",ext_signal=1):
    pp = get_def_path()+ exp_path +"{}itraj/T_{}_dt_{}_ext_signal_{}/".format(itraj, total_time, dt, ext_signal)
    return pp


#### load_data

def load(itraj = 1,total_time = 50., dt = 1e-3, exp_path="", ext_signal=1):
    pp = get_path_config(total_time=total_time, dt=dt, itraj=itraj, exp_path = exp_path, ext_signal=ext_signal)
    states = np.load(pp+"states.npy")
    dys = np.load(pp+"dys.npy")
    return states, dys

























#
# def load_data_discrimination_liks(exp_path="", itraj=1, dt=1e-3,total_time=10):
#     """
#     hyp 1 is the true, that generated the data!
#     """
#     path = get_path_config(total_time = total_time, dt= dt, itraj=itraj, exp_path=exp_path)
#
#     logliks = np.load(path+"logliks.npy",allow_pickle=True,fix_imports=True,encoding='latin1') ### this is \textbf{q}(t)
#     #tims = np.load(path+"times.npy",allow_pickle=True,fix_imports=True,encoding='latin1')
#     return logliks#, tims
#
#
#
#
# def load_data_discrimination_liks_v2(exp_path="", itraj=1, dt=1e-3,total_time=10):
#     """
#     hyp 1 is the true, that generated the data!
#     """
#     path = get_path_config(total_time = total_time, dt= dt, itraj=itraj, exp_path=exp_path)
#
#     logliks = np.load(path+"logliks.npy",allow_pickle=True,fix_imports=True,encoding='latin1') ### this is \textbf{q}(t)
#     return logliks
#
#
#
#
#
#
#
#
# #### analysis stopping time ###
#
# def load_liks(itraj, mode="frequencies", dtt=1e-6, total_time_in=50.):
#     pars = give_def_params_discrimination(flip=0, mode = mode)
#     params, exp_path = check_params_discrimination(pars)
#     [gamma1, omega1, n1, eta1, kappa1], [gamma0, omega0, n0, eta0, kappa0] = params
#     logliks =load_data_discrimination_liks(itraj=itraj, total_time = total_time_in, dt=dtt, exp_path = exp_path)
#     l1  = logliks[:,0] - logliks[:,1]
#
#     pars = give_def_params_discrimination(flip=1, mode = mode)
#     params, exp_path = check_params_discrimination(pars)
#     [gamma1, omega1, n1, eta1, kappa1], [gamma0, omega0, n0, eta0, kappa0] = params
#     logliks =load_data_discrimination_liks(itraj=itraj, total_time = total_time_in, dt=dtt, exp_path = exp_path)
#     l0  = logliks[:,1] - logliks[:,0]
#
#     return l0, l1#, tims
#
#
# def get_stop_time(ell,b, times):
#     logicals = np.logical_and(ell < b, ell > -b)
#     ind_times = np.argmin(logicals)
#
#     if (np.sum(logicals) == 0) or (ind_times==0):
#         return np.nan
#     else:
#         return times[ind_times]
#
#
# def prob(t, b, kappa0, kappa1, eta0 , eta1, n0, n1, gamma0, gamma1):
#     Su1 = n1 + 0.5 + (kappa1 / gamma1)
#     Su0 = n0 + 0.5 + (kappa0 / gamma0)
#
#     S1 = (np.sqrt(1 + (16.0*eta1*kappa1*Su1/gamma1)) - 1)*(gamma1/(8.0*eta1*kappa1))
#     S0 = (np.sqrt(1 + (16.0*eta0*kappa0*Su0/gamma0)) - 1)*( gamma0/(8.0*eta0*kappa0))
#
#     lam = gamma0 + (8*eta0*kappa0*S0)
#
#     aa = (4*eta1*kappa1*(S1**2))/gamma1
#     bb =(4*eta0*kappa0*S0**2)*(1+((16.0*eta1*kappa1*S1)/ (gamma1 + lam)) + (64.0*(eta1 * kappa1 * S1)**(2)/(gamma1 * (gamma1 + lam))))/ lam
#     c =8 *(S0*S1*(eta0*kappa0 *eta1*kappa1)**(0.5)) * (gamma1+ (4.0*eta1*kappa1*S1) ) / ((gamma1 + lam)*gamma1)
#
#     mu = 4*(eta1*kappa1*aa + (eta0*kappa0*bb) - 2*np.sqrt(eta1*kappa1*eta0*kappa0)*c)
#     S= np.sqrt(2*mu)
#
#     div = (np.sqrt(2*np.pi)*S*(t**(3/2)))
#     return  abs(b)*np.exp(-((abs(b)-mu*t)**2)/(2*t*(S**2)))/div, mu
