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
# if os.path.isdir("Results-PR/%f"%r)==False: os.makedirs("Results-PR/%f"%r, exist_ok=True)
if os.path.isdir("Results-sq/%f"%r)==False: os.makedirs("Results-sq/%f"%r, exist_ok=True)
if os.path.exists("Results-sq/%f/error-%f-%d.csv"%(r,r,seed))==False:
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
        tmp = np.sum(np.square(tmp1-tmp2))
        return tmp/np.sum(np.ones((n,n))[P==1])
    def objk(R, W, P, k):
        Q = R.reshape(-1,1)-R.reshape(1,-1)
        tmp1 = W[k,:][P[k,:]==1]; tmp2 = expit(Q[k,:][P[k,:]==1])
        tmp3 = W[:,k][P[:,k]==1]; tmp4 = expit(Q[:,k][P[:,k]==1])
        tmp = np.sum(np.square(tmp1-tmp2))+np.sum(np.square(tmp3-tmp4))
        return tmp
    def gradk(R, W, P, k):
        Q = R.reshape(-1,1) - R.reshape(1,-1)
        mask_k = (P[k,:] == 1)
        mask_i = (P[:,k] == 1)
        tmp2 = expit(Q[k,:][mask_k])
        tmp1 = W[k,:][mask_k]
        tmp4 = expit(Q[:,k][mask_i])
        tmp3 = W[:,k][mask_i]
        term1 = np.sum(2 * (tmp2 - tmp1) * tmp2 * (1 - tmp2))
        term2 = np.sum(2 * (tmp4 - tmp3) * tmp4 * (1 - tmp4))
        return term1 - term2
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
            # res[0,-1] = obj(res[0,:n], W, test_P)
        if ite!=0:
            old_PX = PX.copy(); old_PY = PY.copy()
            res[0,:n] = est[ite-1,:].copy()
            res[0,-2] = iso_obj(res[0,:n], W, train_P, old_PX, old_PY)
            # res[0,-1] = iso_obj(res[0,:n], W, test_P,  old_PX, old_PY)
        # print("iteration:",ite+1,"\tepoch:",1,"\ttrain:",res[0,-2],"\ttest:",res[0,-1])
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
                    # print("iteration:",ite+1,"\tepoch:",t+1,"\tinner epoch:",k,0,"\ttrain:",res_old[-2])
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
                            # print("iteration:",ite+1,"\tepoch:",t+1,"\tinner epoch:",k,s+1,"\ttrain:",res_new[-2])
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
                # res[t,-1] = obj(res[t,:n], W, test_P)
            if ite!=0:
                res[t,-2] = iso_obj(res[t,:n], W, train_P, old_PX, old_PY)
                # res[t,-1] = iso_obj(res[t,:n], W, test_P,  old_PX, old_PY)
            # print("iteration:",ite+1,"\tepoch:",t+1,"\ttrain:",res[t,-2],"\ttest:",res[t,-1])
            # compare res[t,-2] & res[t-1,-2]
            if res[t,-2]==res[t-1,-2]: break
        final_t = t+1
        est[ite] = res[t,:n].copy()
        if np.array_equal(est[ite],est[ite-1]): break
        # if ite!=0 and res[t,-2]>err[ite,10]: break


        # data preparation
        Rij = (est[ite].reshape(-1,1)-est[ite].reshape(1,-1))[train_P==1]
        Wij = W[train_P==1]
        L = len(Wij)

        # # isotonic learning
        # S = np.argsort(Rij); Rij = Rij[S]; Wij = Wij[S]
        # unique_Rij = np.unique(Rij); N = len(unique_Rij); B = []
        # for t in range(N): B.append(np.arange(L)[Rij==unique_Rij[t]].tolist())
        # N = len(B); Z = np.zeros(N)
        # for i in range(N): Z[i] = np.mean(Wij[B[i]])#np.median(Wij[B[i]])
        # #
        # for t in range(10000):
        #     Flag = 0
        #     for i in range(N-1,0,-1):
        #         if Z[i-1]>=Z[i]: B[i-1] += B[i]; _ = B.pop(i); Flag = 1
        #     N = len(B); Z = np.zeros(N)
        #     for i in range(N): Z[i] = np.mean(Wij[B[i]])#np.median(Wij[B[i]])
        #     if Flag == 0: break


        # PX = [Rij[B[0][0]]]
        # PY = [Z[0]]
        # for i in range(N):
        #     if Rij[B[i][0]]!=PX[-1]:
        #         PX.append(Rij[B[i][0]])
        #         PY.append(Z[i])
        #     if Rij[B[i][-1]]!=PX[-1]:
        #         PX.append(Rij[B[i][-1]])
        #         PY.append(Z[i])
        # PX = np.array(PX)
        # PY = np.array(PY)

        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(Rij.astype(np.float64), Wij.astype(np.float64))
        PX = ir.X_thresholds_.astype(np.float64)
        PY = ir.y_thresholds_.astype(np.float64)

        # model
        # def model(u, PX, PY):
        #     L = len(PX)
        #     if isinstance(u, float)==True:
        #         k = np.count_nonzero([u>=PX])
        #         if k==0: res = PY[0]
        #         elif k==L: res = PY[-1]
        #         else: res = PY[k-1]+(u-PX[k-1])*(PY[k]-PY[k-1])/(PX[k]-PX[k-1])
        #     elif u.ndim==1:
        #         k = np.zeros(u.shape, dtype=np.int32)
        #         res = np.zeros(u.shape)
        #         for i in range(u.shape[0]):
        #             k[i] = np.count_nonzero([u[i]>=PX])
        #         k = np.clip(k, a_min=1, a_max=L-1)
        #         res = PY.take(k-1)+(u-PX.take(k-1))*(PY.take(k)-PY.take(k-1))/(PX.take(k)-PX.take(k-1))
        #         res = np.clip(res, a_min=PY[0], a_max=PY[-1])
        #     elif u.ndim==2:
        #         k = np.zeros(u.shape, dtype=np.int32)
        #         res = np.zeros(u.shape)
        #         for i in range(u.shape[0]):
        #             for j in range(u.shape[1]):
        #                 k[i,j] = np.count_nonzero([u[i,j]>=PX])
        #         k = np.clip(k, a_min=1, a_max=L-1)
        #         res = PY.take(k-1)+(u-PX.take(k-1))*(PY.take(k)-PY.take(k-1))/(PX.take(k)-PX.take(k-1))
        #         res = np.clip(res, a_min=PY[0], a_max=PY[-1])
        #     return res
        def model(u, PX, PY):
            return np.interp(u, PX, PY, left=PY[0], right=PY[-1])
        def iso_obj(R, W, P, PX, PY):
            Q = R.reshape(-1,1)-R.reshape(1,-1)
            M = model(Q, PX, PY)
            tmp1 = W[P==1]; tmp2 = M[P==1]
            tmp = np.sum(np.square(tmp1-tmp2))
            return tmp/np.sum(np.ones((n,n))[P==1])
        def iso_objk(R, W, P, PX, PY, k):
            Q = R.reshape(-1,1)-R.reshape(1,-1)
            M = model(Q, PX, PY)
            tmp1 = W[k,:][P[k,:]==1]; tmp2 = M[k,:][P[k,:]==1]
            tmp3 = W[:,k][P[:,k]==1]; tmp4 = M[:,k][P[:,k]==1]
            tmp = np.sum(np.square(tmp1-tmp2))+np.sum(np.square(tmp3-tmp4))
            return tmp
        # def iso_gradk(R, W, P, PX, PY, k):
        #     Q = R.reshape(-1,1)-R.reshape(1,-1)
        #     L = len(PX)
        #     tmp = 0.
        #     for j in np.arange(n)[P[k,:]==1]:
        #         l = np.count_nonzero([Q[k,j]>=PX])
        #         if l==0 or l==L:
        #             pass
        #         elif Q[k,j] in PX and l!=L-1:
        #             tmp -= (W[k,j]-model(Q[k,j], PX, PY)) * np.max([(PY[l]-PY[l-1])/(PX[l]-PX[l-1]), (PY[l+1]-PY[l])/(PX[l+1]-PX[l])])
        #         else:
        #             tmp -= (W[k,j]-model(Q[k,j], PX, PY)) * (PY[l]-PY[l-1])/(PX[l]-PX[l-1])
        #     for j in np.arange(n)[P[:,k]==1]:
        #         l = np.count_nonzero([Q[j,k]>=PX])
        #         if l==0 or l==L:
        #             pass
        #         elif Q[j,k] in PX and l!=L-1:
        #             tmp += (W[j,k]-model(Q[j,k], PX, PY)) * np.max([(PY[l]-PY[l-1])/(PX[l]-PX[l-1]), (PY[l+1]-PY[l])/(PX[l+1]-PX[l])])
        #         else:
        #             tmp += (W[j,k]-model(Q[j,k], PX, PY)) * (PY[l]-PY[l-1])/(PX[l]-PX[l-1])
        #     return tmp
        # def iso_gradk(R, W, P, PX, PY, k):
        #     Q_k_all = R[k] - R
        #     Q_all_k = R - R[k]
        #     slopes = (PY[1:] - PY[:-1]) / (PX[1:] - PX[:-1])
        #     L = len(PX)
        #     def get_grad_contribution(Q_vec, W_vec, P_mask, is_row=True):
        #         M_vec = model(Q_vec, PX, PY)
        #         indices = np.searchsorted(PX, Q_vec)
        #         valid_mask = (indices > 0) & (indices < L) & (P_mask == 1)
        #         current_slopes = np.zeros_like(Q_vec)
        #         idx_to_use = indices[valid_mask] - 1
        #         current_slopes[valid_mask] = slopes[idx_to_use]
        #         diff = (W_vec - M_vec) * current_slopes * P_mask
        #         return np.sum(diff)
        #     term1 = get_grad_contribution(Q_k_all, W[k, :], P[k, :])
        #     term2 = get_grad_contribution(Q_all_k, W[:, k], P[:, k])
        #     return -term1 + term2
        # def iso_gradk(R, W, P, PX, PY, k):
        #     Q_k_all = R[k] - R
        #     Q_all_k = R - R[k]
        #     slopes = (PY[1:] - PY[:-1]) / (PX[1:] - PX[:-1])
        #     L = len(PX)
        #     def get_grad_contribution(Q_vec, W_vec, P_mask):
        #         M_vec = model(Q_vec, PX, PY)
        #         indices = np.searchsorted(PX, Q_vec, side='right')
        #         valid_mask = (indices > 0) & (indices < L) & (P_mask == 1)
        #         current_slopes = np.zeros_like(Q_vec, dtype=float)
        #         active_indices = indices[valid_mask]
        #         Q_valid = Q_vec[valid_mask]
        #         is_on_node_vals = np.isin(Q_valid, PX) & (active_indices != L - 1)
        #         is_on_node = np.zeros_like(valid_mask, dtype=bool)
        #         is_on_node[valid_mask] = is_on_node_vals
        #         current_slopes[valid_mask] = slopes[active_indices - 1]
        #         node_mask = valid_mask & is_on_node
        #         if np.any(node_mask):
        #             idx_node = indices[node_mask]
        #             left_slopes = slopes[idx_node - 1]
        #             right_slopes = slopes[idx_node]
        #             current_slopes[node_mask] = np.maximum(left_slopes, right_slopes)
        #         diff = (W_vec - M_vec) * current_slopes * P_mask
        #         return np.sum(diff)
        #     term1 = get_grad_contribution(Q_k_all, W[k, :], P[k, :])
        #     term2 = get_grad_contribution(Q_all_k, W[:, k], P[:, k])
        #     return -term1 + term2
        def iso_gradk(R, W, P, PX, PY, k):
            Q_k_all = R[k] - R
            Q_all_k = R - R[k]
            slopes = (PY[1:] - PY[:-1]) / (PX[1:] - PX[:-1])
            L = len(PX)
            def get_grad_contribution(Q_vec, W_vec, P_mask):
                M_vec = model(Q_vec, PX, PY)
                indices = np.searchsorted(PX, Q_vec, side='left')
                interior = (indices > 0) & (indices < L) & (P_mask == 1)
                current_slopes = np.zeros_like(Q_vec, dtype=float)
                on_node = interior & np.array([np.isclose(q, PX).any() for q in Q_vec])
                normal = interior & ~on_node
                current_slopes[normal] = slopes[indices[normal] - 1]
                if np.any(on_node):
                    idx = indices[on_node]
                    left_s = slopes[idx - 1]
                    right_s = np.where(idx < L-1, slopes[np.minimum(idx, L-2)], left_s)
                    current_slopes[on_node] = np.maximum(left_s, right_s)
                diff = 2 * (M_vec - W_vec) * current_slopes * P_mask
                return np.sum(diff)
            term1 = get_grad_contribution(Q_k_all, W[k, :], P[k, :])
            term2 = get_grad_contribution(Q_all_k, W[:, k], P[:, k])
            return term1 - term2
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
        # print("comparison @ iteration %d"%(ite+1))
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
        # print("rate(t),sigma(t-1):",err[ite,:10])
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
        # print("rate(t),sigma( t ):",err[ite,10:])

    # if T==1: np.savetxt("Results-PR/%f/error-%f-%d.csv"%(r,r,seed), err, delimiter=",")
    np.savetxt("Results-sq/%f/error-%f-%d.csv"%(r,r,seed), err, delimiter=",")

