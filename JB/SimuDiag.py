
import numpy as np
import time
def SimuDiag(S_mu,S_ep):
    a,b = np.linalg.eig(S_mu)
    C = np.linalg.cholesky(S_ep)
    C_inv = np.linalg.inv(C)
    S_mu_proj = np.dot(np.dot(C_inv,S_mu),C_inv.T)
    # U.T*C_inv*S_mu*C_inv.T*U = k
    # U.T*C_inv*S_ep*C_inv.T*U = I
    # fai = C_inv.T*U
    # fai.T*S_ep*fai = I
    # fai.T*S_mu*fai = k
    k,U=np.linalg.eig(S_mu_proj)
    # k = k[k>1e-6]
    k = np.diag(k)
    U = U[:,:len(k)]
    U_inv = np.linalg.pinv(U)
    fai = np.dot(C_inv.T,U)

    ###########
    # the next lines are debugging...
    # G=np.dot(np.dot(fai.T,S_mu),fai)-k
    # H=np.dot(np.dot(fai.T,S_ep),fai)-np.identity(k.shape[0])
    # print(np.max(H))
    # print(np.min(H))
    # print(np.max(G))
    # print(np.min(G))
    return fai,k
if __name__ == "__main__":
    path = "../exp/jb/xvectors_callhome1/JB"
    S_ep=np.load(path+'/S_ep.npy')
    S_mu=np.load(path+'/S_mu.npy')
    SimuDiag(S_mu,S_ep)
