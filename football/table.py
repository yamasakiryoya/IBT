import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy import stats
plt.rcParams["font.size"] = 20
plt.rcParams['text.usetex'] = True
import warnings
warnings.simplefilter('ignore')



table = pd.DataFrame(np.zeros((6, 15)))

for ind2 in range(5):
    r = [.1,.3,.5,.7,.9][ind2]
    res = np.zeros((1000,10,32))
    result = np.zeros((1000,20,16))
    for seed in range(1000):
        res[seed] = np.loadtxt("Results-nll/%f/error-%f-%d.csv"%(r,r,seed), delimiter=",")
        for k in range(10):
            if np.all(res[seed,k,:] == 0):
                res[seed,k,:16] = res[seed,k-1,16:]
                res[seed,k,16:] = res[seed,k-1,16:]
    for k in range(10):
        result[:,2*k,:] = res[:,k,:16]
        result[:,2*k+1,:] = res[:,k,16:]
    #
    res2 = np.zeros((1000,10,32))
    result2 = np.zeros((1000,20,16))
    for seed in range(1000):
        res2[seed] = np.loadtxt("Results-nll2/%f/error-%f-%d.csv"%(r,r,seed), delimiter=",")
        for k in range(10):
            if np.all(res2[seed,k,:] == 0):
                res2[seed,k,:16] = res2[seed,k-1,16:]
                res2[seed,k,16:] = res2[seed,k-1,16:]
    for k in range(10):
        result2[:,2*k,:] = res2[:,k,:16]
        result2[:,2*k+1,:] = res2[:,k,16:]
    #
    BT  = result[:,0,:]
    IBT = np.zeros((1000,16))
    for i in range(1000):
        IBT[i,1] = result[i,np.argmin(result2[i,:,1]),1]
        IBT[i,5] = result[i,19-np.argmax(result2[i,:,5][::-1]),5]

    test1 = stats.mannwhitneyu(IBT[:,1], BT[:,1], alternative='less')
    test5 = stats.mannwhitneyu(IBT[:,5], BT[:,5], alternative='greater')

    if test1.pvalue<0.05:
        table.iloc[0,ind2] = "\\tcr{$%.4f_{%.4f}$}"%(np.mean(BT[:,1]),np.std(BT[:,1]))
        table.iloc[1,ind2] = "\\tcr{$%.4f_{%.4f}$}"%(np.mean(IBT[:,1]),np.std(IBT[:,1]))
        table.iloc[2,ind2] = "(\\tcr{$%.4f$})"%test1.pvalue
    else:
        table.iloc[1,ind2] = "$%.4f_{%.4f}$"%(np.mean(IBT[:,1]),np.std(IBT[:,1]))
        table.iloc[0,ind2] = "$%.4f_{%.4f}$"%(np.mean(BT[:,1]),np.std(BT[:,1]))
        table.iloc[2,ind2] = "($%.4f$)"%test1.pvalue
    if test5.pvalue<0.05:
        table.iloc[3,ind2] = "\\tcr{$%.4f_{%.4f}$}"%(np.mean(BT[:,5]),np.std(BT[:,5]))
        table.iloc[4,ind2] = "\\tcr{$%.4f_{%.4f}$}"%(np.mean(IBT[:,5]),np.std(IBT[:,5]))
        table.iloc[5,ind2] = "(\\tcr{$%.4f$})"%test5.pvalue
    else:
        table.iloc[4,ind2] = "$%.4f_{%.4f}$"%(np.mean(IBT[:,5]),np.std(IBT[:,5]))
        table.iloc[3,ind2] = "$%.4f_{%.4f}$"%(np.mean(BT[:,5]),np.std(BT[:,5]))
        table.iloc[5,ind2] = "($%.4f$)"%test5.pvalue

for i in range(6):
    for j in range(5):
        print(end="&")
        print(table.iloc[i,j],end="")
    print("\\\\")



table = pd.DataFrame(np.zeros((6, 15)))

for ind2 in range(5):
    r = [.1,.3,.5,.7,.9][ind2]
    res = np.zeros((1000,10,32))
    result = np.zeros((1000,20,16))
    for seed in range(1000):
        res[seed] = np.loadtxt("Results-sq/%f/error-%f-%d.csv"%(r,r,seed), delimiter=",")
        for k in range(10):
            if np.all(res[seed,k,:] == 0):
                res[seed,k,:16] = res[seed,k-1,16:]
                res[seed,k,16:] = res[seed,k-1,16:]
    for k in range(10):
        result[:,2*k,:] = res[:,k,:16]
        result[:,2*k+1,:] = res[:,k,16:]
    #
    res2 = np.zeros((1000,10,32))
    result2 = np.zeros((1000,20,16))
    for seed in range(1000):
        res2[seed] = np.loadtxt("Results-sq2/%f/error-%f-%d.csv"%(r,r,seed), delimiter=",")
        for k in range(10):
            if np.all(res2[seed,k,:] == 0):
                res2[seed,k,:16] = res2[seed,k-1,16:]
                res2[seed,k,16:] = res2[seed,k-1,16:]
    for k in range(10):
        result2[:,2*k,:] = res2[:,k,:16]
        result2[:,2*k+1,:] = res2[:,k,16:]
    #
    BT  = result[:,0,:]
    IBT = np.zeros((1000,16))
    for i in range(1000):
        IBT[i,1] = result[i,np.argmin(result2[i,:,1]),1]
        IBT[i,5] = result[i,19-np.argmax(result2[i,:,5][::-1]),5]

    test1 = stats.mannwhitneyu(IBT[:,1], BT[:,1], alternative='less')
    test5 = stats.mannwhitneyu(IBT[:,5], BT[:,5], alternative='greater')

    if test1.pvalue<0.05:
        table.iloc[0,ind2] = "\\tcr{$%.4f_{%.4f}$}"%(np.mean(BT[:,1]),np.std(BT[:,1]))
        table.iloc[1,ind2] = "\\tcr{$%.4f_{%.4f}$}"%(np.mean(IBT[:,1]),np.std(IBT[:,1]))
        table.iloc[2,ind2] = "(\\tcr{$%.4f$})"%test1.pvalue
    else:
        table.iloc[1,ind2] = "$%.4f_{%.4f}$"%(np.mean(IBT[:,1]),np.std(IBT[:,1]))
        table.iloc[0,ind2] = "$%.4f_{%.4f}$"%(np.mean(BT[:,1]),np.std(BT[:,1]))
        table.iloc[2,ind2] = "($%.4f$)"%test1.pvalue
    if test5.pvalue<0.05:
        table.iloc[3,ind2] = "\\tcr{$%.4f_{%.4f}$}"%(np.mean(BT[:,5]),np.std(BT[:,5]))
        table.iloc[4,ind2] = "\\tcr{$%.4f_{%.4f}$}"%(np.mean(IBT[:,5]),np.std(IBT[:,5]))
        table.iloc[5,ind2] = "(\\tcr{$%.4f$})"%test5.pvalue
    else:
        table.iloc[4,ind2] = "$%.4f_{%.4f}$"%(np.mean(IBT[:,5]),np.std(IBT[:,5]))
        table.iloc[3,ind2] = "$%.4f_{%.4f}$"%(np.mean(BT[:,5]),np.std(BT[:,5]))
        table.iloc[5,ind2] = "($%.4f$)"%test5.pvalue

for i in range(6):
    for j in range(5):
        print(end="&")
        print(table.iloc[i,j],end="")
    print("\\\\")
