import numpy as np

def get_def_path():
    import getpass
    user = getpass.getuser()
    if user == "cooper-cooper":
        defpath = '../quantera/trajectories/'
    else:
        defpath = "/data/uab-giq/scratch/matias/quantera/trajectories/"
    return defpath

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



def load_data(path="", itraj=1, ppp=500,periods=40, method="rossler"):
    if path == "":
        path = get_def_path()
    path +="{}periods/{}ppp/{}/{}/".format(periods,ppp, method, itraj)

    times = np.load(path+"times.npy", allow_pickle=True).astype(np.float32) ### this is \textbf{q}(t)
    states = np.load(path+"states.npy", allow_pickle=True).astype(np.float32) ### this is \textbf{q}(t)
    covs = np.load(path+"covs.npy", allow_pickle=True).astype(np.float32) ## this is the \Sigma(t)
    signals = np.load(path+"signals.npy", allow_pickle=True).astype(np.float32) ##this is the dy's
    params = np.load(path+"params.npy", allow_pickle=True).astype(np.float32) ##this is the dy's
    #coeffs = np.load(path+"coeffs.npy".format(itraj), allow_pickle=True).astype(np.float32) ##this is the dy's
    print("Traj loaded \nppp: {}\nperiods: {}\nmethod: {}\nitraj: {}".format(ppp,periods,method,itraj))
    return states, covs, signals, params, times

def build_matrix_from_params(params):
    [eta, gamma, kappa, omega, n] = params
    C = np.array([[np.sqrt(2*eta*kappa),0],[0,0]]) #homodyne
    A = np.array([[-gamma/2, omega],[-omega, -gamma/2]])
    D = np.diag([(gamma*(n+0.5))]*2)
    Lambda = np.zeros((2,2))
    return [C, A, D , Lambda] 


def load_train_results(path="",train_path="",periods=20, ppp=1000, train_id=1):
    if path == "":
        path = get_def_path()
    if train_path == "":
        train_path = path+"{}periods/{}ppp/training/train_id_{}/".format(periods, ppp, train_id)
    else:
        train_path = path+"{}periods/{}ppp/".format(periods,ppp) + train_path + "training/train_id_{}/".format(train_id)

    hist_A = np.load(train_path+"Coeffs_A.npy")
    hist_loss = np.load(train_path+"total_loss.npy")
    hist_grads = np.load(train_path+"grads.npy")
    return hist_A, hist_loss, hist_grads


def sliced_dataset(signals, xicovs, t):
    import tensorflow as tf
    tfsignals = tf.convert_to_tensor(signals)[:t] ## this are dy's
    tfxicovs = tf.convert_to_tensor(xicovs)[:-1][:t] ### since we get (\mu_t, \Sigma_t) ---> Measurement ( = signal) ---> (\mu_{t+1}, \Sigma_{t+1})
    return (tfxicovs[tf.newaxis,:,:,:], tfsignals[tf.newaxis,:,:])
