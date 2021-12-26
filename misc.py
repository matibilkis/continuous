import numpy as np

def get_def_path():
    import getpass
    user = getpass.getuser()
    if user == "cooper-cooper":
        defpath = '../quantera/sanity/trajectories/'
    else:
        defpath = "/data/uab-giq/scratch/matias/quantera/sanity/trajectories/"
    return defpath


def ct(A):
    return np.transpose(np.conjugate(A))

def sliced_dataset(signals, xicovs, t):
    import tensorflow as tf
    tfsignals = tf.convert_to_tensor(signals)[:t] ## this are dy's
    tfxicovs = tf.convert_to_tensor(xicovs)[:-1][:t] ### since we get (\mu_t, \Sigma_t) ---> Measurement ( = signal) ---> (\mu_{t+1}, \Sigma_{t+1})
    return (tfxicovs[tf.newaxis,:,:,:], tfsignals[tf.newaxis,:,:])


def load_data(path, itraj=0):
    means = np.load(path+"{}/means.npy".format(itraj), allow_pickle=True).astype(np.float32) ### this is \textbf{q}(t)
    covs = np.load(path+"{}/covs.npy".format(itraj), allow_pickle=True).astype(np.float32) ## this is the \Sigma(t)
    xicovs = np.load(path+"{}/xicovs.npy".format(itraj), allow_pickle=True).astype(np.float32) ## this is the \Chi(\Sigma) (evolution)
    signals = np.load(path+"{}/signals.npy".format(itraj), allow_pickle=True).astype(np.float32) ##this is the dy's
    A = np.load(path+"{}/A.npy".format(itraj), allow_pickle=True).astype(np.float32)
    dt = np.load(path+"{}/dt.npy".format(itraj), allow_pickle=True)[0]
    C = np.load(path+"{}/C.npy".format(itraj), allow_pickle=True).astype(np.float32)
    D = np.load(path+"{}/D.npy".format(itraj), allow_pickle=True).astype(np.float32)
    return means, covs, signals, [A, dt, C, D]
