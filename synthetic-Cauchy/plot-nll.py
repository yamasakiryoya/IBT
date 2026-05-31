import os
import sys
import numpy as np
import numpy.random as rd
import scipy as sp
from scipy.special import logit, expit, log_expit, erf
from scipy.stats import skewnorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.isotonic import IsotonicRegression
from scipy.stats import kendalltau,spearmanr
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy import stats
plt.rcParams["font.size"] = 20
plt.rcParams['text.usetex'] = True
import warnings
warnings.simplefilter('ignore')


args = sys.argv
seed, n, r, T = int(args[1]), int(args[2]), float(args[3]), int(args[4])
if os.path.isdir("Results-nll3/")==False: os.makedirs("Results-nll3/", exist_ok=True)
if os.path.exists("Results-nll3/plot-%d-%f-%d-%d.png"%(n,r,T,seed))==False:
    rd.seed(seed)
    # data
    R = rd.normal(0,1,n)
    W = np.zeros((n,n)); Y = np.zeros((n,n))
    for i in range(n-1):
        for j in range(i+1,n):
            W[i,j] = rd.binomial(T, np.arctan(R[i]-R[j])/np.pi+0.5)/T
            W[j,i] = 1.-W[i,j]
            if W[i,j]>=0.5: Y[i,j] = 1.
            if W[j,i]>=0.5: Y[j,i] = 1.
    # data split
    all_ij = []
    for i in range(n-1):
        for j in range(i+1,n):
            if W[i,j]!=-1: all_ij.append((i,j))
    ind = np.arange(len(all_ij))
    np.random.shuffle(ind)
    split_idx = int(r*len(all_ij))
    train_idx_list = ind[:split_idx]; test_idx_list = ind[split_idx:]
    train_P = np.zeros((n, n)); test_P = np.zeros((n, n))
    for k in train_idx_list:
        train_P[all_ij[k][0],all_ij[k][1]] = 1; train_P[all_ij[k][1],all_ij[k][0]] = 1
    for k in test_idx_list:
        test_P[all_ij[k][0],all_ij[k][1]] = 1; test_P[all_ij[k][1],all_ij[k][0]] = 1

    # loss & gradient function
    def obj(R, W, P):
        Q = R.reshape(-1,1)-R.reshape(1,-1)
        tmp1 = W[P==1]; tmp2 = expit(Q[P==1])
        tmp = np.sum(-tmp1*np.log(tmp2))
        return tmp/np.sum(np.ones((n,n))[P==1])
    def objk(R, W, P, k):
        Q = R.reshape(-1,1)-R.reshape(1,-1)
        tmp1 = W[k,:][P[k,:]==1]; tmp2 = expit(Q[k,:][P[k,:]==1])
        tmp3 = W[:,k][P[:,k]==1]; tmp4 = expit(Q[:,k][P[:,k]==1])
        tmp = np.sum(-tmp1*np.log(tmp2))+np.sum(-tmp3*np.log(tmp4))
        return np.nan_to_num(tmp, nan=10**8)
    def gradk(R, W, P, k):
        Q = R.reshape(-1,1)-R.reshape(1,-1)
        tmp1 = W[k,:][P[k,:]==1]; tmp2 = expit(Q[k,:][P[k,:]==1])
        tmp3 = W[:,k][P[:,k]==1]; tmp4 = expit(Q[:,k][P[:,k]==1])
        tmp = -np.sum(tmp1*(1-tmp2))+np.sum(tmp3*(1-tmp4))
        return tmp
    def rnk1(R, W, P):
        Q = R.reshape(-1,1)-R.reshape(1,-1)
        tmp1 = W[P==1]; tmp2 = expit(Q[P==1])
        tau, _ = kendalltau(tmp1, tmp2)
        return tau
    def rnk2(R, W, P):
        Q = R.reshape(-1,1)-R.reshape(1,-1)
        tmp1 = W[P==1]; tmp2 = expit(Q[P==1])
        rho, _ = spearmanr(tmp1, tmp2)
        return rho
    def rnk3(R, W, P):
        Q = R.reshape(-1,1)-R.reshape(1,-1)
        tmp1 = W[P==1]; tmp2 = expit(Q[P==1])
        return np.sum(tmp2==0.5)/tmp2.size
    def rnk4(R, W, P):
        Q = R.reshape(-1,1)-R.reshape(1,-1)
        tmp1 = W[P==1]; tmp2 = Q[P==1]
        tau, _ = kendalltau(tmp1, tmp2)
        return tau
    def rnk5(R, W, P):
        Q = R.reshape(-1,1)-R.reshape(1,-1)
        tmp1 = W[P==1]; tmp2 = Q[P==1]
        rho, _ = spearmanr(tmp1, tmp2)
        return rho
    def rnk6(R, W, P):
        Q = R.reshape(-1,1)-R.reshape(1,-1)
        tmp1 = W[P==1]; tmp2 = Q[P==1]
        return np.sum(tmp2==0.5)/tmp2.size

    ITE = 1
    est = np.zeros((ITE,n))
    for ite in range(ITE):
        # preparation
        EP1 = 100; EP2 = 100; res = np.zeros((EP1,n+2))
        # epoch 0
        if ite==0:
            res[0,:n] = np.zeros(n)
            res[0,-2] = obj(res[0,:n], W, train_P)
        if ite!=0:
            old_PX = PX.copy(); old_PY = PY.copy()
            res[0,:n] = est[ite-1,:].copy()
            res[0,-2] = iso_obj(res[0,:n], W, train_P, old_PX, old_PY)
        #
        res_old = res[0,:].copy(); res_new = res[0,:].copy()
        # epoch t
        for t in range(1,EP1):
            for k in range(n):
                # update res[t,k]
                for s in range(EP2):
                    # initialize res[t,k]
                    if ite==0: res_old[-2] = objk(res_old[:n], W, train_P, k)
                    if ite!=0: res_old[-2] = iso_objk(res_old[:n], W, train_P, old_PX, old_PY, k)
                    if ite==0: tmp = gradk(res_old[:n], W, train_P, k)
                    if ite!=0: tmp = iso_gradk(res_old[:n], W, train_P, old_PX, old_PY, k)
                    #
                    LR = 1./np.max([.1**30,np.abs(tmp)])
                    #
                    flag1 = 0
                    while flag1 == 0:
                        res_new[k] = res_old[k]-LR*tmp
                        if ite==0: res_new[-2] = objk(res_new[:n], W, train_P, k)
                        if ite!=0: res_new[-2] = iso_objk(res_new[:n], W, train_P, old_PX, old_PY, k)
                        #
                        if res_new[-2]<res_old[-2]:
                            flag1 = 1; res_old = res_new.copy()
                        elif res_new[-2]==res_old[-2]:
                            flag1 = 2; res_new = res_old.copy()
                        elif LR<.1**30:
                            flag1 = 2; res_new = res_old.copy()
                        else:
                            LR *= 0.5
                    if flag1 == 2:
                        break
            # update res[t]
            res[t,:n] = res_new[:n].copy()
            if ite==0:
                res[t,-2] = obj(res[t,:n], W, train_P)
            if ite!=0:
                res[t,-2] = iso_obj(res[t,:n], W, train_P, old_PX, old_PY)
            # compare res[t,-2] & res[t-1,-2]
            if res[t,-2]==res[t-1,-2]: break
        final_t = t+1
        est[ite] = res[t,:n].copy()


        # data preparation
        Rij = (est[ite].reshape(-1,1)-est[ite].reshape(1,-1))[train_P==1]
        Wij = W[train_P==1]
        L = len(Wij)

        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(Rij.astype(np.float64), Wij.astype(np.float64))
        PX = ir.X_thresholds_.astype(np.float64)
        PY = ir.y_thresholds_.astype(np.float64)


        def model(u, PX, PY):
            return np.interp(u, PX, PY, left=PY[0], right=PY[-1])
        def iso_obj(R, W, P, PX, PY):
            Q = R.reshape(-1,1)-R.reshape(1,-1)
            M = model(Q, PX, PY)
            tmp1 = W[P==1]; tmp2 = M[P==1]
            tmp = np.sum(-tmp1[tmp1!=0]*np.log(tmp2[tmp1!=0]))
            return tmp/np.sum(np.ones((n,n))[P==1])
        def iso_objk(R, W, P, PX, PY, k):
            Q = R.reshape(-1,1)-R.reshape(1,-1)
            M = model(Q, PX, PY)
            tmp1 = W[k,:][P[k,:]==1]; tmp2 = M[k,:][P[k,:]==1]
            tmp3 = W[:,k][P[:,k]==1]; tmp4 = M[:,k][P[:,k]==1]
            tmp = np.sum(-tmp1[tmp1!=0]*np.log(tmp2[tmp1!=0]))+np.sum(-tmp3[tmp3!=0]*np.log(tmp4[tmp3!=0]))
            return np.nan_to_num(tmp, nan=10**8)

        def iso_gradk(R, W, P, PX, PY, k):
            Q_k_all = R[k] - R
            Q_all_k = R - R[k]
            slopes = (PY[1:] - PY[:-1]) / (PX[1:] - PX[:-1])
            L = len(PX)
            def get_slope(Q_vec):
                indices = np.searchsorted(PX, Q_vec, side='left')
                current_slopes = np.zeros_like(Q_vec, dtype=float)
                valid = (indices > 0) & (indices < L)
                idx = indices[valid]
                current_slopes[valid] = slopes[idx - 1]
                on_node = valid & (np.isin(Q_vec, PX))
                if np.any(on_node):
                    idx_node = indices[on_node]
                    left = slopes[idx_node - 1]
                    right = slopes[np.minimum(idx_node, L-2)]
                    current_slopes[on_node] = np.maximum(left, right)
                return current_slopes
            M_k = model(Q_k_all, PX, PY)
            slope_k = get_slope(Q_k_all)
            mask_k = (P[k, :] == 1) & (M_k!=0)
            term1 = np.sum(W[k, mask_k] / M_k[mask_k] * slope_k[mask_k])
            M_i = model(Q_all_k, PX, PY)
            slope_i = get_slope(Q_all_k)
            mask_i = (P[:, k] == 1) & (M_i!=0)
            term2 = np.sum(W[mask_i, k] / M_i[mask_i] * slope_i[mask_i])
            return -term1 + term2
        def iso_rnk1(R, W, P, PX, PY):
            Q = R.reshape(-1,1)-R.reshape(1,-1)
            M = model(Q, PX, PY)
            tmp1 = W[P==1]; tmp2 = M[P==1]
            tau, _ = kendalltau(tmp1, tmp2)
            return tau
        def iso_rnk2(R, W, P, PX, PY):
            Q = R.reshape(-1,1)-R.reshape(1,-1)
            M = model(Q, PX, PY)
            tmp1 = W[P==1]; tmp2 = M[P==1]
            rho, _ = spearmanr(tmp1, tmp2)
            return rho
        def iso_rnk3(R, W, P, PX, PY):
            Q = R.reshape(-1,1)-R.reshape(1,-1)
            M = model(Q, PX, PY)
            tmp1 = W[P==1]; tmp2 = M[P==1]
            return np.sum(tmp2==0.5)/tmp2.size
        def iso_rnk4(R, W, P, PX, PY):
            Q = R.reshape(-1,1)-R.reshape(1,-1)
            tmp1 = W[P==1]; tmp2 = Q[P==1]
            tau, _ = kendalltau(tmp1, tmp2)
            return tau
        def iso_rnk5(R, W, P, PX, PY):
            Q = R.reshape(-1,1)-R.reshape(1,-1)
            tmp1 = W[P==1]; tmp2 = Q[P==1]
            rho, _ = spearmanr(tmp1, tmp2)
            return rho
        def iso_rnk6(R, W, P, PX, PY):
            Q = R.reshape(-1,1)-R.reshape(1,-1)
            tmp1 = W[P==1]; tmp2 = Q[P==1]
            return np.sum(tmp2==0.5)/tmp2.size


        fig, ax = plt.subplots(figsize=(10, 5)); cmap = plt.get_cmap('jet')
        Xtra = (est[ite].reshape(-1,1)-est[ite].reshape(1,-1))[train_P==1].flatten()
        ax.scatter(Xtra, W[train_P==1].flatten(), s=10, marker="D", edgecolor='k', facecolor='none', zorder=2)
        Xtes = (est[ite].reshape(-1,1)-est[ite].reshape(1,-1))[test_P==1].flatten()
        ax.scatter(Xtes, W[test_P==1].flatten(),  s=10, marker="o", edgecolor='r', facecolor='none', zorder=1)
        LIM = np.max([np.max(np.fabs(Xtra)),np.max(np.fabs(Xtes))])
        fx = np.linspace(-LIM*1.1,LIM*1.1,10000)#fx = np.linspace(-7,7,10000)#
        ax.plot(fx, expit(fx), lw=3, color="g", zorder=3)
        ax.plot(fx, model(fx, PX, PY), lw=3, color="b", linestyle="--", zorder=4)
        ax.set_xlabel(r'$u=\hat{r}_i^{[1]}-\hat{r}_j^{[1]}$')
        ax.set_ylabel(r'$y_{i,j}$, $\hat{\sigma}^{[0]}(u)$, $\hat{\sigma}^{[1]}(u)$')
        ax.set_xlim(-LIM*1.1,LIM*1.1); ax.grid(True)#ax.set_xlim(-7,7); ax.grid(True)#
        plt.savefig("Results-nll3/plot-%d-%f-%d-%d.png"%(n,r,T,seed), bbox_inches="tight", pad_inches=.02, facecolor=fig.get_facecolor(), dpi=100, edgecolor='none', metadata={'Software': None}, pil_kwargs={'optimize': True}); plt.close()#,format='webp', 