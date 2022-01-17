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


def load_data(path="", itraj=0, ppp=500,periods=40, method="RK"):
    if path == "":
        path = get_def_path()
    path +="{}periods/{}ppp/{}/{}/".format(periods,ppp,itraj, method)
    states = np.load(path+"states.npy".format(itraj), allow_pickle=True).astype(np.float32) ### this is \textbf{q}(t)
    covs = np.load(path+"covs.npy".format(itraj), allow_pickle=True).astype(np.float32) ## this is the \Sigma(t)
    signals = np.load(path+"signals.npy".format(itraj), allow_pickle=True).astype(np.float32) ##this is the dy's
    A = np.load(path+"A.npy".format(itraj), allow_pickle=True).astype(np.float32)
    dt = np.load(path+"dt.npy".format(itraj), allow_pickle=True)[0]
    C = np.load(path+"C.npy".format(itraj), allow_pickle=True).astype(np.float32)
    D = np.load(path+"D.npy".format(itraj), allow_pickle=True).astype(np.float32)
    return states, covs, signals, [A, dt, C, D]


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
