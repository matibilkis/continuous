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


def load_data(path="", itraj=1, ppp=500,periods=40, method="rossler", unphysical=False):
    if path == "":
        path = get_def_path()
    path +="{}periods/{}ppp/{}/{}/".format(periods,ppp, method, itraj)
    if unphysical is True:
        path+="unphysical_"
    times = np.load(path+"times.npy", allow_pickle=True).astype(np.float32) ### this is \textbf{q}(t)
    states = np.load(path+"states.npy", allow_pickle=True).astype(np.float32) ### this is \textbf{q}(t)
    covs = np.load(path+"covs.npy", allow_pickle=True).astype(np.float32) ## this is the \Sigma(t)
    signals = np.load(path+"signals.npy", allow_pickle=True).astype(np.float32) ##this is the dy's
    params = np.load(path+"params.npy", allow_pickle=True).astype(np.float32) ##this is the dy's
    #coeffs = np.load(path+"coeffs.npy".format(itraj), allow_pickle=True).astype(np.float32) ##this is the dy's
    print("Traj loaded \nppp: {}\nperiods: {}\nmethod: {}\nitraj: {}\nUnphyisical (testing): {}".format(ppp,periods,method,itraj, unphysical))
    return states, covs, signals, params, times

def build_matrix_from_params(params):
    [eta, gamma, Lambda, omega, n] = params
    A = np.array([[-.5*gamma, omega], [-omega, -0.5*gamma]])
    D = np.diag([(gamma*(n+0.5)) + Lambda]*2)
    C = np.diag([np.sqrt(4*eta*Lambda)]*2)
    return [A,C,D]


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
