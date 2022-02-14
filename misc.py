import numpy as np
import ast
import os
def get_def_path():
    import getpass
    user = getpass.getuser()
    if user == "cooper-cooper":
        defpath = '../quantera/trajectories/'
    else:
        defpath = "/data/uab-giq/scratch/matias/quantera/trajectories/"
    return defpath

def give_def_params():
    ######  https://arxiv.org/pdf/2005.03429.pdf

    omega = (2*np.pi)*(1.14)*1e6
    n = 14
    gamma = 19*2*np.pi#(4*np.pi*265/29)*1e-6
    kappa = np.pi*0.36*1e3#4*np.pi*0.36*1e-3
    eta = 0.74

    ### ASPELMAYER  p.16 (correctons w/giulio)
    n = 20
    g = 6*(10**5)
    k_aspel = 2*np.pi*(6.6)*10**5
    [eta, gamma, kappa, omega, n] = [0.74,  2*np.pi*1e3,  2*(g**2)*n/k_aspel, 4.2*1e4, n]

    return [eta, gamma, kappa, omega, n]


def check_params(params):
    if params == "":
        params = give_def_params()
        exp_path = ""
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

    signals = ss[:,2:4]
    signals = signals[1:] - signals[:-1]

    covss = ss[:,-3:]
    covs = [s_to_cov(s,begin_cov=0) for s in covss]
    return states, signals, covs


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
    signals = np.load(path+"signals.npy", allow_pickle=True).astype(np.float32) ##this is the dy's
    params = np.load(path+"params.npy", allow_pickle=True).astype(np.float32) ##this is the dy's
    #coeffs = np.load(path+"coeffs.npy".format(itraj), allow_pickle=True).astype(np.float32) ##this is the dy's
    if display is True:
        print("Traj loaded \nppp: {}\nperiods: {}\nmethod: {}\nitraj: {}".format(ppp,periods,method,itraj))
    return states, covs, signals, params, times

def build_matrix_from_params(params):
    [eta, gamma, kappa, omega, n] = params
    C = np.array([[np.sqrt(2*eta*kappa),0],[0,0]]) #homodyne
    A = np.array([[-gamma/2, omega],[-omega, -gamma/2]])
    D = np.diag([(gamma*(n+0.5))]*2)
    Lambda = np.zeros((2,2))
    return [C, A, D , Lambda]


# def load_train_results(path="",train_path="",periods=20, ppp=1000, train_id=1):
#     if path == "":
#         path = get_def_path()
#     if train_path == "":
#         train_path = path+"{}periods/{}ppp/training/train_id_{}/".format(periods, ppp, train_id)
#     else:
#         train_path = path+"{}periods/{}ppp/".format(periods,ppp) + train_path + "training/train_id_{}/".format(train_id)
#
#     hist_A = np.load(train_path+"Coeffs_A.npy")
#     hist_loss = np.load(train_path+"total_loss.npy")
#     hist_grads = np.load(train_path+"grads.npy")
#     return hist_A, hist_loss, hist_grads


def sliced_dataset(signals, xicovs, t):
    import tensorflow as tf
    tfsignals = tf.convert_to_tensor(signals)[:t] ## this are dy's
    tfxicovs = tf.convert_to_tensor(xicovs)[:-1][:t] ### since we get (\mu_t, \Sigma_t) ---> Measurement ( = signal) ---> (\mu_{t+1}, \Sigma_{t+1})
    return (tfxicovs[tf.newaxis,:,:,:], tfsignals[tf.newaxis,:,:])
