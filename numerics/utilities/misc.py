import numpy as np
import ast
import os

def get_def_path(mode="discrimination/"):
    import getpass
    user = getpass.getuser()
    if user == "cooper-cooper":
        defpath = '../quantera/trajectories/'
    elif (user =="matias") or (user == "mati"):
        defpath = '../quantera/trajectories/'
    else:
        defpath = "/data/uab-giq/scratch/matias/quantera/trajectories/"
    if mode[-1] != "/":
        mode+="/"
    defpath+=mode
    return defpath


def give_def_params_discrimination(flip =0, mode="damping"):
    if mode == "frequencies":
        #print("FREQUENCY DISCRIMINATION!\n")
        gamma0 = gamma1 = 100
        eta0 = eta1 = 1
        kappa0 = kappa1 = 1e6
        n0 = n1 = 1
        omega0, omega1 = 1e4, 1e4 + 1e3
    elif mode=="damping":
        #print("DAMPING DISCRIMINATION!")
        gamma1 = 14*2*np.pi
        gamma0 = 19*2*np.pi #(Hz)
        eta1 = 0.9
        eta0 = 0.9
        n1 = 14.0
        n0 = 14.0
        kappa1 = 2*np.pi*360
        kappa0 = 2*np.pi*360 #(Hz)
        omega0 = omega1 = 0.

    h0 = [gamma0, omega0, n0, eta0, kappa0]
    h1 = [gamma1, omega1, n1, eta1, kappa1]
    if flip == 0:
        return [h1, h0]
    else:
        return [h0, h1]

def check_params_discrimination(params):
    if isinstance(params, str):
        params = ast.literal_eval(params)
    exp_path = '{}/'.format(params)
    return params, exp_path


def ct(A):
    return np.transpose(np.conjugate(A))



def get_total_time_dt(params, ppp=1000, dt=1e-5, total_time=4):
    [gamma1, omega1, n1, eta1, kappa1], [gamma0, omega0, n0, eta0, kappa0] = params
    if omega1 != 0:
        Period = (2*np.pi/omega1)
        dt = Period/(ppp)
        total_time = total_time*Period
    return total_time, dt



def get_path_config(total_time=10,dt=1e-3,itraj=1,method="hybrid",exp_path=""):
    if exp_path!="":
        pp = get_def_path()+ exp_path +"{}itraj/T_{}_dt_{}/".format(itraj, total_time, dt)
    else:
        pp = get_def_path()+"{}itraj/T_{}_dt_{}/".format(itraj,total_time, dt)
    return pp



def load_data_discrimination_liks(exp_path="", itraj=1, dt=1e-3,total_time=10):
    """
    hyp 1 is the true, that generated the data!
    """
    path = get_path_config(total_time = total_time, dt= dt, itraj=itraj, exp_path=exp_path)

    logliks = np.load(path+"logliks.npy",allow_pickle=True,fix_imports=True,encoding='latin1') ### this is \textbf{q}(t)
    return logliks








#### analysis stopping time ###


def load_liks(itrajj, mode="damping", dtt=1e-6, total_time_in=4.):
    pars = give_def_params_discrimination(flip=0, mode = mode, )
    params, exp_path = check_params_discrimination(pars)

    total_time, dt = get_total_time_dt(params, ppp=1000, dt=dtt, total_time = total_time_in)
    
    logliks =load_data_discrimination_liks(itraj=itrajj, total_time = total_time, dt=dt, exp_path = exp_path)
    l0,l1 = logliks[:,0], logliks[:,1]
    log_lik_ratio = l1 - l0  ###

    pars = give_def_params_discrimination(flip=1, mode=mode)
    params, exp_path = check_params_discrimination(pars)
    total_time, dt = get_total_time_dt(params, ppp=1000,dt=dtt ,total_time = total_time_in)
    logliks_swap =load_data_discrimination_liks(itraj=itrajj, total_time = total_time, dt=dt, exp_path = exp_path)
    log_lik_ratio_swap = logliks_swap[:,0] - logliks_swap[:,1]  
    
    return log_lik_ratio, log_lik_ratio_swap


def get_stop_time(ell,b, times):
    logicals = np.logical_and(ell < b, ell > -b)
    ind_times = np.argmin(logicals)

    if (np.sum(logicals) == 0) or (ind_times==0):
        return np.nan
    else:
        return times[ind_times]
    
    
def prob(t, b, kappa0, kappa1, eta0 , eta1, n0, n1, gamma0, gamma1):
    Su1 = n1 + 0.5 + (kappa1 / gamma1)
    Su0 = n0 + 0.5 + (kappa0 / gamma0)

    S1 = (np.sqrt(1 + (16.0*eta1*kappa1*Su1/gamma1)) - 1)*(gamma1/(8.0*eta1*kappa1))
    S0 = (np.sqrt(1 + (16.0*eta0*kappa0*Su0/gamma0)) - 1)*( gamma0/(8.0*eta0*kappa0))

    lam = gamma0 + (8*eta0*kappa0*S0)

    aa = (4*eta1*kappa1*(S1**2))/gamma1
    bb =(4*eta0*kappa0*S0**2)*(1+((16.0*eta1*kappa1*S1)/ (gamma1 + lam)) + (64.0*(eta1 * kappa1 * S1)**(2)/(gamma1 * (gamma1 + lam))))/ lam
    c =8 *(S0*S1*(eta0*kappa0 *eta1*kappa1)**(0.5)) * (gamma1+ (4.0*eta1*kappa1*S1) ) / ((gamma1 + lam)*gamma1)

    mu = 4*(eta1*kappa1*aa + (eta0*kappa0*bb) - 2*np.sqrt(eta1*kappa1*eta0*kappa0)*c)
    S= np.sqrt(2*mu)

    div = (np.sqrt(2*np.pi)*S*(t**(3/2)))
    return  abs(b)*np.exp(-((abs(b)-mu*t)**2)/(2*t*(S**2)))/div, mu