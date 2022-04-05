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
        total_time = 10*Period
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

    logliks = np.load(path+"logliks.npy") ### this is \textbf{q}(t)
    return logliks
