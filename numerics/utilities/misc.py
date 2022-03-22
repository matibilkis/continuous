import numpy as np
import ast
import os

def get_def_path(mode="discrimination/"):
    import getpass
    user = getpass.getuser()
    if user == "cooper-cooper":
        defpath = '../quantera/trajectories/'
    elif user =="matias":
        defpath = '../quantera/trajectories/'
    else:
        defpath = "/data/uab-giq/scratch/matias/quantera/trajectories/"
    defpath+=mode
    return defpath

def give_def_params(mode="test"):
    [eta, gamma, kappa, omega, n] = [1,  .3 , 0., 10]
    return [eta, gamma, gamma1, kappa, omega, n]

def give_def_params_discrimination(mode="test"):
    gamma = 0.3
    gamma1 = 1.
    omega = 0#2*np.pi
    omega1 = 0#2*np.pi
    eta = 1.
    kappa = 20
    n = 20
    return [gamma, gamma1, omega, omega1, eta, kappa, n]


def check_params_discrimination(params):
    if params == "":
        params = give_def_params_discrimination()
        exp_path = '{}/'.format(params)
        #exp_path = ""
    else:
        if isinstance(params, str):
            params = ast.literal_eval(params)
        exp_path = '{}/'.format(params)
    return params, exp_path

def check_params(params):
    if params == "":
        params = give_def_params()
        exp_path = '{}/'.format(params)
        #exp_path = ""
    else:
        if isinstance(params, str):
            params = ast.literal_eval(params)
        exp_path = '{}/'.format(params)
    return params, exp_path


def get_windows():
    try:
        windows = np.load(get_def_path()+"tmp/windows.npy")
    except Exception:
        windows = np.concatenate([(10**k)*np.arange(2,11) for k in range(2)])
    return windows

def save_windows(win):
    wdir = get_def_path()+"tmp/"
    os.makedirs(wdir,exist_ok=True)
    np.save(wdir+"windows.npy", win)
    return

def params_to_string(params):
    return "'{}'".format(params)


def ct(A):
    return np.transpose(np.conjugate(A))

def s_to_cov(s,begin_cov=4):
    varx, varp,covxy = s[begin_cov:]
    cov = np.array([[varx, covxy], [covxy, varp]])
    return cov


def convert_solution(ss):
    states = ss[:,0:2]

    ### we want the measurement results at each time-step
    signals = ss[:,2:4]
    signals = signals[1:] - signals[:-1]

    covss = ss[:,4:7]
    covs = [s_to_cov(s,begin_cov=0) for s in covss]

    u_th = ss[:,7:9]

    covss_th = ss[:,9:12]
    covs_th = [s_to_cov(s, begin_cov=0) for s in covss_th]
    return states, signals, covs, u_th, covs_th

def convert_solution_discrimination(ss):
    states = ss[:,0:2]

    ### we want the measurement results at each time-step
    signals = ss[:,2:4]
    signals = signals[1:] - signals[:-1]

    covss = ss[:,4:7]
    covs = [s_to_cov(s,begin_cov=0) for s in covss]

    states1 = ss[:,7:9]
    covss1 = ss[:,9:12]
    covs1 = [s_to_cov(s, begin_cov=0) for s in covss1]

    l0 = ss[:,12]
    l1 = ss[:,13]

    return states, signals, covs, states1, covs1, l0, l1


def get_path_config(periods=100,ppp=1000,itraj=1,method="rossler", rppp=1, exp_path=""):
    if exp_path!="":
        pp = get_def_path()+ exp_path + "{}itraj/{}_real_traj_method/{}periods/{}ppp/{}rppp/".format(itraj, method, periods, ppp, rppp)
    else:
        pp = get_def_path()+"{}itraj/{}_real_traj_method/{}periods/{}ppp/{}rppp/".format(itraj, method, periods, ppp, rppp)
    return pp

def load_data(exp_path="", itraj=1, ppp=1000,periods=100, rppp=1, method="rossler", display=False):

    path = get_path_config(periods = periods, ppp= ppp, rppp=rppp, method=method, itraj=itraj, exp_path=exp_path)

    times = np.load(path+"times.npy", allow_pickle=True).astype(np.float32) ### this is \textbf{q}(t)
    states = np.load(path+"states.npy", allow_pickle=True).astype(np.float32) ### this is \textbf{q}(t)
    covs = np.load(path+"covs.npy", allow_pickle=True).astype(np.float32) ## this is the \Sigma(t)
    states1 = np.load(path+"states1.npy", allow_pickle=True).astype(np.float32) ### this is \textbf{q}(t)
    covs1 = np.load(path+"covs1.npy", allow_pickle=True).astype(np.float32) ## this is the \Sigma(t)
    l0 = np.load(path+"loglik0.npy", allow_pickle=True).astype(np.float32) ### this is \textbf{q}(t)
    l1 = np.load(path+"loglik1.npy", allow_pickle=True).astype(np.float32) ### this is \textbf{q}(t)
    signals = np.load(path+"signals.npy", allow_pickle=True).astype(np.float32) ##this is the dy's
    params = np.load(path+"params.npy", allow_pickle=True).astype(np.float32) ##this is the dy's
    #coeffs = np.load(path+"coeffs.npy".format(itraj), allow_pickle=True).astype(np.float32) ##this is the dy's
    if display is True:
        print("Traj loaded \nppp: {}\nperiods: {}\nmethod: {}\nitraj: {}".format(ppp,periods,method,itraj))
    return times, l0, l1, states, states1, signals, covs, covs1

def build_matrix_from_params(params):
    [eta, gamma, kappa, omega, n] = params
    C = np.array([[np.sqrt(2*eta*kappa),0],[0,0]]) #homodyne
    A = np.array([[-gamma/2, omega],[-omega, -gamma/2]])
    D = np.diag([(gamma*(n+0.5))]*2)
    Lambda = np.zeros((2,2))
    return [C, A, D , Lambda]


def sliced_dataset(signals, xicovs, t):
    import tensorflow as tf
    tfsignals = tf.convert_to_tensor(signals)[:t] ## this are dy's
    tfxicovs = tf.convert_to_tensor(xicovs)[:-1][:t] ### since we get (\mu_t, \Sigma_t) ---> Measurement ( = signal) ---> (\mu_{t+1}, \Sigma_{t+1})
    return (tfxicovs[tf.newaxis,:,:,:], tfsignals[tf.newaxis,:,:])
