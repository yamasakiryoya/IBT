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
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
plt.rcParams["font.size"] = 20
plt.rcParams['text.usetex'] = True
cmap = plt.get_cmap('jet')

rd.seed(0)

for n in [1000]:# number of subjects
    for m in [100]:# number of problems
        # data
        A = rd.normal(0.,1.,n)
        B = rd.normal(0.,1.,m)
        Y = np.zeros((n,m))# correct or incorrect
        for i in range(n):
            for j in range(m):
                if rd.rand()<(0.2+0.8*(np.arctan(A[i]-B[j])/np.pi+0.5)): Y[i,j] = 1.#(np.arctan(A[i]-B[j])/np.pi+0.5)#expit(A[i]-B[j])
        # data split
        train_pair = []
        for i in range(n):
            for j in range(m):
                train_pair.append([i,j])

        # true objective
        def true_obj(A, B, Y, pairs):
            C = A.reshape(-1,1)-B.reshape(1,-1)
            tmp  = np.sum(-np.log(1.-(0.2+0.8*(np.arctan(C[Y==0.])/np.pi+0.5))))#(np.arctan(C[Y==0.])/np.pi+0.5)#expit(C[Y==0.])
            tmp += np.sum(-np.log(0.2+0.8*(np.arctan(C[Y==1.])/np.pi+0.5)))#(np.arctan(C[Y==1.])/np.pi+0.5)#expit(C[Y==1.])
            return tmp/len(pairs)
        print("true\ttrain:",true_obj(A, B, Y, train_pair))

        # loss & gradient function
        def obj(A, B, Y, pairs):
            C = A.reshape(-1,1)-B.reshape(1,-1)
            tmp = np.sum(-log_expit(-C[Y==0.]))+np.sum(-log_expit(C[Y==1.]))
            """
            tmp = 0.
            for i,j in pairs:
                if Y[i,j]==0.: tmp += -log_expit(-(A[i]-B[j]))
                if Y[i,j]==1.: tmp += -log_expit(A[i]-B[j])
                #tmp += (Y[i,j]-expit(A[i]-B[j]))**2
            """
            return tmp/len(pairs)
        def grad(A, B, Y, pairs):
            C = A.reshape(-1,1)-B.reshape(1,-1)
            D = expit(C);  D[Y==1.] = 0.
            E = expit(-C); E[Y==0.] = 0.
            tmpA = np.sum(D-E, axis=1)
            tmpB = np.sum(E-D, axis=0)
            """
            #tmpA = np.zeros(n)
            #tmpB = np.zeros(m)
            for i in range(n): tmpA[i] = np.sum(expit(C[i,:][Y[i,:]==0.]))+np.sum(-expit(-C[i,:][Y[i,:]==1.]))
            for j in range(m): tmpB[j] = np.sum(-expit(C[:,j][Y[:,j]==0.]))+np.sum(expit(-C[:,j][Y[:,j]==1.]))
            """
            """
            for i,j in pairs:
                if Y[i,j]==0.:
                    tmpA[i] += expit(A[i]-B[j])
                    tmpB[j] += -expit(A[i]-B[j])
                if Y[i,j]==1.:
                    tmpA[i] += -expit(-(A[i]-B[j]))
                    tmpB[j] += expit(-(A[i]-B[j]))
                #tmpA[i] -= (Y[i,j]-expit(A[i]-B[j]))*expit(A[i]-B[j])*expit(-(A[i]-B[j]))
                #tmpB[j] += (Y[i,j]-expit(A[i]-B[j]))*expit(A[i]-B[j])*expit(-(A[i]-B[j]))
            """
            return tmpA/len(pairs), tmpB/len(pairs)


        ITE = 10
        estA = np.zeros((ITE,n))
        estB = np.zeros((ITE,m))
        estO = np.zeros(ITE)
        for ite in range(ITE):
            if ite==0:
                EP = 10000
                resA = np.zeros((EP,n)); resB = np.zeros((EP,m)); resO = np.zeros(EP)
                resO[0] = obj(resA[0], resB[0], Y, train_pair)
            else:
                EP = 100
                resA = np.zeros((EP,n)); resB = np.zeros((EP,m)); resO = np.zeros(EP)
                old_PX = PX.copy(); old_PY = PY.copy()
                resA[0] = estA[ite-1].copy(); resB[0] = estB[ite-1].copy()
                resO[0] = iso_obj(resA[0], resB[0], Y, train_pair, old_PX, old_PY)
            print("iteration:",ite+1,"\tepoch:",1,"\ttrain:",resO[0])
            # epoch t
            for t in range(1,EP):
                #
                if ite==0:
                    tmpA, tmpB = grad(resA[t-1], resB[t-1], Y, train_pair)
                    if t%10==1: LR = 0.1/np.max([.1**15,np.max(np.abs(tmpA)),np.max(np.abs(tmpB))])
                else:
                    tmpA, tmpB = iso_grad(resA[t-1], resB[t-1], Y, train_pair, old_PX, old_PY)
                    if t%10==1: LR = 1000./np.max([.1**15,np.max(np.abs(tmpA)),np.max(np.abs(tmpB))])
                #
                flag = 0
                while flag == 0:
                    resA[t] = resA[t-1]-LR*tmpA
                    resB[t] = resB[t-1]-LR*tmpB
                    if ite==0: resO[t] = obj(resA[t], resB[t], Y, train_pair)
                    else:  resO[t] = iso_obj(resA[t], resB[t], Y, train_pair, old_PX, old_PY)
                    #
                    if resO[t]<resO[t-1]:
                        flag = 1
                        print("iteration:",ite+1,"\tepoch:",t+1,"\ttrain:",resO[t])
                    elif resO[t]==resO[t-1]:
                        flag = 2
                        print("iteration:",ite+1,"\tepoch:",t+1,"\ttrain:",resO[t])
                    else:
                        LR *= 0.5
                if flag==2:
                    break
            final_t = t+1
            estA[ite] = resA[t].copy()
            estB[ite] = resB[t].copy()
            estO[ite] = resO[t].copy()
            if np.array_equal(estA[ite],estA[ite-1]) and np.array_equal(estB[ite],estB[ite-1]) and estO[ite]==estO[ite-1]: break


            # data preparation
            L = len(train_pair); Xij = np.zeros(L); Yij = np.zeros(L)
            for t, (i,j) in enumerate(train_pair): Xij[t] = estA[ite,i]-estB[ite,j]; Yij[t] = Y[i,j]


            # isotonic learning
            S = np.argsort(Xij); Xij = Xij[S]; Yij = Yij[S]
            unique_Xij = np.unique(Xij); N = len(unique_Xij); G = []
            for t in range(N): G.append(np.arange(L)[Xij==unique_Xij[t]].tolist())
            N = len(G); Z = np.zeros(N)
            for i in range(N): Z[i] = np.mean(Yij[G[i]])
            #
            for t in range(10000):
                Flag = 0
                for i in range(N-1,0,-1):
                    if Z[i-1]>=Z[i]: G[i-1] += G[i]; _ = G.pop(i); Flag = 1
                N = len(G); Z = np.zeros(N)
                for i in range(N): Z[i] = np.mean(Yij[G[i]])
                if Flag == 0: break


            PX = [Xij[G[0][0]]]
            PY = [Z[0]]
            for i in range(N):
                if Xij[G[i][0]]!=PX[-1]:
                    PX.append(Xij[G[i][0]])
                    PY.append(Z[i])
                if Xij[G[i][-1]]!=PX[-1]:
                    PX.append(Xij[G[i][-1]])
                    PY.append(Z[i])
            PX = np.array(PX)
            PY = np.array(PY)


            # model
            def model(u, PX, PY):
                L = len(PX)
                if isinstance(u, float)==True:
                    k = np.count_nonzero([u>=PX])
                    if k==0: res = PY[0]
                    elif k==L: res = PY[-1]
                    else: res = PY[k-1]+(u-PX[k-1])*(PY[k]-PY[k-1])/(PX[k]-PX[k-1])
                elif u.ndim==1:
                    k = np.zeros(u.shape, dtype=np.int32)
                    res = np.zeros(u.shape)
                    for i in range(u.shape[0]):
                        k[i] = np.count_nonzero([u[i]>=PX])
                    k = np.clip(k, a_min=1, a_max=L-1)
                    res = PY.take(k-1)+(u-PX.take(k-1))*(PY.take(k)-PY.take(k-1))/(PX.take(k)-PX.take(k-1))
                    res = np.clip(res, a_min=PY[0], a_max=PY[-1])
                elif u.ndim==2:
                    k = np.zeros(u.shape, dtype=np.int32)
                    res = np.zeros(u.shape)
                    for i in range(u.shape[0]):
                        for j in range(u.shape[1]):
                            k[i,j] = np.count_nonzero([u[i,j]>=PX])
                    k = np.clip(k, a_min=1, a_max=L-1)
                    res = PY.take(k-1)+(u-PX.take(k-1))*(PY.take(k)-PY.take(k-1))/(PX.take(k)-PX.take(k-1))
                    res = np.clip(res, a_min=PY[0], a_max=PY[-1])
                return res
            def iso_obj(A, B, Y, pairs, PX, PY):
                C = A.reshape(-1,1)-B.reshape(1,-1)
                M = model(C, PX, PY)
                #M = np.clip(M, a_min=.1**10, a_max=1.-.1**10)
                tmp = np.sum(-np.log(1.-M[Y==0.]))+np.sum(-np.log(M[Y==1.]))
                """
                tmp = 0.
                for i,j in pairs:
                    if Y[i,j]==0.: tmp += -np.log(1.-model(A[i]-B[j], PX, PY))
                    if Y[i,j]==1.: tmp += -np.log(model(A[i]-B[j], PX, PY))
                    #tmp += (Y[i,j]-model(A[i]-B[j], PX, PY))**2
                """
                return tmp/len(pairs)
            def iso_grad(A, B, Y, pairs, PX, PY):
                L = len(PX)
                C = A.reshape(-1,1)-B.reshape(1,-1)
                tmpA = np.zeros(n)
                tmpB = np.zeros(m)
                for i in range(n):
                    for j in range(m):
                        l = np.count_nonzero([C[i,j]>=PX])
                        if l==0 or l==L:
                            pass
                        elif C[i,j] in PX and l!=L-1:
                            if Y[i,j]==0. and model(C[i,j], PX, PY)!=1.:
                                tmpA[i] += 1./(1.-model(C[i,j], PX, PY)) * np.max([(PY[l]-PY[l-1])/(PX[l]-PX[l-1]), (PY[l+1]-PY[l])/(PX[l+1]-PX[l])])
                                tmpB[j] -= 1./(1.-model(C[i,j], PX, PY)) * np.max([(PY[l]-PY[l-1])/(PX[l]-PX[l-1]), (PY[l+1]-PY[l])/(PX[l+1]-PX[l])])
                            if Y[i,j]==1. and model(C[i,j], PX, PY)!=0.:
                                tmpA[i] -= 1./model(C[i,j], PX, PY) * np.max([(PY[l]-PY[l-1])/(PX[l]-PX[l-1]), (PY[l+1]-PY[l])/(PX[l+1]-PX[l])])
                                tmpB[j] += 1./model(C[i,j], PX, PY) * np.max([(PY[l]-PY[l-1])/(PX[l]-PX[l-1]), (PY[l+1]-PY[l])/(PX[l+1]-PX[l])])
                        else:
                            if Y[i,j]==0. and model(C[i,j], PX, PY)!=1.:
                                tmpA[i] += 1./(1.-model(C[i,j], PX, PY)) * (PY[l]-PY[l-1])/(PX[l]-PX[l-1])
                                tmpB[j] -= 1./(1.-model(C[i,j], PX, PY)) * (PY[l]-PY[l-1])/(PX[l]-PX[l-1])
                            if Y[i,j]==1. and model(C[i,j], PX, PY)!=0.:
                                tmpA[i] -= 1./model(C[i,j], PX, PY) * (PY[l]-PY[l-1])/(PX[l]-PX[l-1])
                                tmpB[j] += 1./model(C[i,j], PX, PY) * (PY[l]-PY[l-1])/(PX[l]-PX[l-1])
                return tmpA, tmpB

            # evaluation
            print("comparison @ iteration %d"%(ite+1))
            if ite==0: a = obj(estA[ite], estB[ite], Y, train_pair)
            else:  a = iso_obj(estA[ite], estB[ite], Y, train_pair, old_PX, old_PY)
            c = iso_obj(estA[ite], estB[ite], Y, train_pair, PX, PY)
            print("train:",a,c)


            # plot 1
            fig = plt.figure(figsize=(10,6)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); A_argsort = np.argsort(A)
            for i in range(n): ax.plot(range(final_t), resA[:final_t,A_argsort[i]], lw=1, color=cmap(i/(n-1)))
            ax.set_xlabel(r'epoch'); ax.grid(True); plt.tight_layout()
            plt.savefig("./%d-%d-%d-1-A.png"%(n,m,ite), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
            #
            fig = plt.figure(figsize=(10,6)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); B_argsort = np.argsort(B)
            for i in range(m): ax.plot(range(final_t), resB[:final_t,B_argsort[i]], lw=1, color=cmap(i/(m-1)))
            ax.set_xlabel(r'epoch'); ax.grid(True); plt.tight_layout()
            plt.savefig("./%d-%d-%d-1-B.png"%(n,m,ite), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()


            # plot 2
            fig = plt.figure(figsize=(10,6)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111)
            # data points
            train_E = (estA[ite].reshape(-1,1)-estB[ite].reshape(1,-1)).flatten()
            train_F = (Y).flatten()
            train_G = (A.reshape(-1,1)-B.reshape(1,-1)).flatten()
            maxijE = np.max(np.abs(train_E))
            maxijG = np.max(np.abs(train_G))
            ax.scatter(train_E, train_F, s=3, marker=".", color="r", zorder=4)
            VIO = ax.violinplot([train_E[train_F==0], train_E[train_F==1]], positions=[0,1], widths=0.1, vert=False, showmeans=False, showextrema=False, showmedians=False)
            for pc in VIO['bodies']: pc.set_facecolor("r"); pc.set_alpha(0.3); #pc.set_edgecolor("r")
            ax.scatter(train_E[0]-10.**8, train_F[0], s=200, marker=".", color="r", label="data")
            # prediction curves
            FX = np.linspace(-maxijE,maxijE,1000)
            if ite==0:
                ax.plot(FX, expit(FX), lw=2, color="k", zorder=2, label=r"prediction $\sigma_{\rm Log.}(\hat{a}_i^{[1]}-\hat{b}_j^{[1]})$")#"logistic rating")
                ax.plot(FX, model(FX, PX, PY), lw=2, linestyle="--", color="k", zorder=3, label=r"prediction $\hat{\sigma}^{[1]}(\hat{a}_i^{[1]}-\hat{b}_j^{[1]})$")#"isotonic rating")
            else:
                ax.plot(FX, model(FX, old_PX, old_PY), lw=2, color="k", zorder=2, label=r"prediction $\hat{\sigma}^{[%d]}(\hat{a}_i^{[%d]}-\hat{b}_j^{[%d]})$"%(ite,ite+1,ite+1))#"isotonic rating @ iter.=%d"%ite)
                ax.plot(FX, model(FX, PX, PY), lw=2, linestyle="--", color="k", zorder=3, label=r"prediction $\hat{\sigma}^{[%d]}(\hat{a}_i^{[%d]}-\hat{b}_j^{[%d]})$"%(ite+1,ite+1,ite+1))#"isotonic rating @ iter.=%d"%(ite+1))
            #
            ax.set_xlabel(r"$\hat{a}_i^{[%d]}-\hat{b}_j^{[%d]}$"%(ite+1,ite+1)); ax.set_ylabel(r"{\rm data} $y_{i,j}$ {\rm \&~prediction}"); ax.set_xlim(-maxijE*1.1,maxijE*1.1); plt.legend(loc="lower right")
            ax.set_title(r"${\rm loss}_{\rm train}=%.8f,%.8f$"%(a,c)); ax.grid(True); plt.tight_layout()
            plt.savefig("./%d-%d-%d-2.png"%(n,m,ite), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()


            # plot 3
            fig = plt.figure(figsize=(6,6)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111)
            ax.scatter(train_G, train_E, s=10, marker=".", color="k", zorder=3)
            ax.set_xlabel(r"$\bar{a}_i-\bar{b}_j$"); ax.set_ylabel(r"$\hat{a}_i^{[%d]}-\hat{b}_j^{[%d]}$"%(ite+1,ite+1))
            ax.set_xlim(-maxijG*1.1,maxijG*1.1); ax.set_ylim(-maxijE*1.1,maxijE*1.1); ax.grid(True); plt.tight_layout()
            plt.savefig("./%d-%d-%d-3.png"%(n,m,ite), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()

            MORA = 0
            for i in range(n):
                for j in range(n):
                    if (A[i]<A[j] and estA[ite,i]>estA[ite,j]) or (A[i]>A[j] and estA[ite,i]<estA[ite,j]): MORA += 1
            MORB = 0
            for i in range(m):
                for j in range(m):
                    if (B[i]<B[j] and estB[ite,i]>estB[ite,j]) or (B[i]>B[j] and estB[ite,i]<estB[ite,j]): MORB += 1

            # plot 4
            fig = plt.figure(figsize=(6,6)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111)
            ax.scatter(A, estA[ite], s=10, marker=".", color="k", zorder=3)
            ax.set_title(r"${\rm misordering}_{\rm train}=\frac{%d}{%d}$"%(MORA,n*(n-1)))
            ax.set_xlabel(r"$\bar{a}_i$"); ax.set_ylabel(r"$\hat{a}_i^{[%d]}$"%(ite+1)); ax.grid(True); plt.tight_layout()
            plt.savefig("./%d-%d-%d-4-A.png"%(n,m,ite), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
            #
            fig = plt.figure(figsize=(6,6)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111)
            ax.scatter(B, estB[ite], s=10, marker=".", color="k", zorder=3)
            ax.set_title(r"${\rm misordering}_{\rm train}=\frac{%d}{%d}$"%(MORB,m*(m-1)))
            ax.set_xlabel(r"$\bar{b}_i$"); ax.set_ylabel(r"$\hat{b}_i^{[%d]}$"%(ite+1)); ax.grid(True); plt.tight_layout()
            plt.savefig("./%d-%d-%d-4-B.png"%(n,m,ite), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
