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

args = sys.argv
seed, r = int(args[1]), float(args[2])
if os.path.isdir("Results-nll/%f"%r)==False: os.makedirs("Results-nll/%f"%r, exist_ok=True)
if os.path.exists("Results-nll/%f/error-%f-%d.csv"%(r,r,seed))==False:
    rd.seed(seed)
    # data
    W = np.loadtxt("win_rate_matrix.csv", delimiter=",")
    n = len(W[0])
    Y = np.zeros((n,n))
    for i in range(n-1):
        for j in range(i+1,n):
            if W[i,j]>=0.5: Y[i,j] = 1.
            if W[j,i]>=0.5: Y[j,i] = 1.
    # data split
    train_P = np.zeros((n,n)); test_P = np.zeros((n,n))
    for i in range(n-1):
        for j in range(i+1,n):
            if rd.rand()<=r: train_P[i,j] = 1; train_P[j,i] = 1
            else: test_P[i,j] = 1; test_P[j,i] = 1

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
        return 1-len(np.unique(tmp2))/tmp2.size
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
        return 1-len(np.unique(tmp2))/tmp2.size

    ITE = 10
    est = np.zeros((ITE,n))
    err = np.zeros((ITE,32))
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
        if np.array_equal(est[ite],est[ite-1]): break


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
            return 1-len(np.unique(tmp2))/tmp2.size
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
            return 1-len(np.unique(tmp2))/tmp2.size

        # evaluation
        if ite==0:
            err[ite,0] = obj(est[ite], W, train_P)
            err[ite,1] = obj(est[ite], W, test_P)
            err[ite,2] = obj(est[ite], Y, train_P)
            err[ite,3] = obj(est[ite], Y, test_P)
            err[ite,4] = rnk1(est[ite], W, train_P)
            err[ite,5] = rnk1(est[ite], W, test_P)
            err[ite,6] = rnk2(est[ite], W, train_P)
            err[ite,7] = rnk2(est[ite], W, test_P)
            err[ite,8] = rnk3(est[ite], W, train_P)
            err[ite,9] = rnk3(est[ite], W, test_P)
            err[ite,10]= rnk4(est[ite], W, train_P)
            err[ite,11]= rnk4(est[ite], W, test_P)
            err[ite,12]= rnk5(est[ite], W, train_P)
            err[ite,13]= rnk5(est[ite], W, test_P)
            err[ite,14]= rnk6(est[ite], W, train_P)
            err[ite,15]= rnk6(est[ite], W, test_P)
        else:
            err[ite,0] = iso_obj(est[ite], W, train_P, old_PX, old_PY)
            err[ite,1] = iso_obj(est[ite], W, test_P,  old_PX, old_PY)
            err[ite,2] = iso_obj(est[ite], Y, train_P, old_PX, old_PY)
            err[ite,3] = iso_obj(est[ite], Y, test_P,  old_PX, old_PY)
            err[ite,4] = iso_rnk1(est[ite], W, train_P, old_PX, old_PY)
            err[ite,5] = iso_rnk1(est[ite], W, test_P,  old_PX, old_PY)
            err[ite,6] = iso_rnk2(est[ite], W, train_P, old_PX, old_PY)
            err[ite,7] = iso_rnk2(est[ite], W, test_P,  old_PX, old_PY)
            err[ite,8] = iso_rnk3(est[ite], W, train_P, old_PX, old_PY)
            err[ite,9] = iso_rnk3(est[ite], W, test_P,  old_PX, old_PY)
            err[ite,10]= iso_rnk4(est[ite], W, train_P, old_PX, old_PY)
            err[ite,11]= iso_rnk4(est[ite], W, test_P,  old_PX, old_PY)
            err[ite,12]= iso_rnk5(est[ite], W, train_P, old_PX, old_PY)
            err[ite,13]= iso_rnk5(est[ite], W, test_P,  old_PX, old_PY)
            err[ite,14]= iso_rnk6(est[ite], W, train_P, old_PX, old_PY)
            err[ite,15]= iso_rnk6(est[ite], W, test_P,  old_PX, old_PY)
        err[ite,16] = iso_obj(est[ite], W, train_P, PX, PY)
        err[ite,17] = iso_obj(est[ite], W, test_P,  PX, PY)
        err[ite,18] = iso_obj(est[ite], Y, train_P, PX, PY)
        err[ite,19] = iso_obj(est[ite], Y, test_P,  PX, PY)
        err[ite,20] = iso_rnk1(est[ite], W, train_P, PX, PY)
        err[ite,21] = iso_rnk1(est[ite], W, test_P,  PX, PY)
        err[ite,22] = iso_rnk2(est[ite], W, train_P, PX, PY)
        err[ite,23] = iso_rnk2(est[ite], W, test_P,  PX, PY)
        err[ite,24] = iso_rnk3(est[ite], W, train_P, PX, PY)
        err[ite,25] = iso_rnk3(est[ite], W, test_P,  PX, PY)
        err[ite,26] = iso_rnk4(est[ite], W, train_P, PX, PY)
        err[ite,27] = iso_rnk4(est[ite], W, test_P,  PX, PY)
        err[ite,28] = iso_rnk5(est[ite], W, train_P, PX, PY)
        err[ite,29] = iso_rnk5(est[ite], W, test_P,  PX, PY)
        err[ite,30] = iso_rnk6(est[ite], W, train_P, PX, PY)
        err[ite,31] = iso_rnk6(est[ite], W, test_P,  PX, PY)

    np.savetxt("Results-nll/%f/error-%f-%d.csv"%(r,r,seed), err, delimiter=",")

