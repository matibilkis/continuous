import numpy as np

def genoni_matrices(xi, kappa, omega, eta, **kwargs):
    type = kwargs.get("type","64")
    A = np.array([[-(xi + .5*kappa), omega], [-omega, xi - 0.5*kappa]])
    D = kappa*np.eye(2)
    E = B = -np.sqrt(eta*kappa)*np.array([[1.,0.],[0.,0.]])
    if str(type)=="32":
        A, D, E, B = A.astype(np.float32), D.astype(np.float32), E.astype(np.float32), B.astype(np.float32)
    return A, D, E, B


def genoni_xi_cov(A,D,E,B,params, stat=False):
    xi, kappa, omega, eta = params
    vx = (kappa*(2*eta -1) - 2*xi + np.sqrt(kappa**2 - 4*xi*kappa*(2*eta -1) + 4*xi**2))/(2*eta*kappa)
    vp = kappa/(kappa - 2*xi)
    CovSS = np.diag([vx, vp])
    XiCov = (E - np.dot(CovSS, B))/np.sqrt(2)
    if stat == True:
        return XiCov, CovSS
    else:
        return XiCov
