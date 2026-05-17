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

table = pd.DataFrame(np.zeros((9, 15)))

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
        # IBT[i,5] = result[i,np.argmax(result2[i,:,5]),5]
        # IBT[i,7] = result[i,np.argmax(result2[i,:,7]),7]
        # IBT[i,1] = result[i,19-np.argmin(result2[i,:,1][::-1]),1]
        IBT[i,5] = result[i,19-np.argmax(result2[i,:,5][::-1]),5]
        IBT[i,7] = result[i,19-np.argmax(result2[i,:,7][::-1]),7]

    test1 = stats.mannwhitneyu(IBT[:,1], BT[:,1], alternative='less')
    test5 = stats.mannwhitneyu(IBT[:,5], BT[:,5], alternative='greater')
    test7 = stats.mannwhitneyu(IBT[:,7], BT[:,7], alternative='greater')

    # print(n,r,T,",",
    #     "$%.4f-%.4f$"%(np.mean(BT[:,1]),np.std(BT[:,1])),
    #     "$%.4f-%.4f$"%(np.mean(IBT[:,1]),np.std(IBT[:,1])),"(%.4f),"%test1.pvalue,
    #     "$%.4f-%.4f$"%(np.mean(BT[:,5]),np.std(BT[:,5])),
    #     "$%.4f-%.4f$"%(np.mean(IBT[:,5]),np.std(IBT[:,5])),"(%.4f),"%test5.pvalue,
    #     "$%.4f-%.4f$"%(np.mean(BT[:,7]),np.std(BT[:,7])),
    #     "$%.4f-%.4f$"%(np.mean(IBT[:,7]),np.std(IBT[:,7])),"(%.4f),"%test7.pvalue)
    if test1.pvalue<.05:
        table.iloc[0,ind2] = "\\tcr{$%.4f_{%.4f}$}"%(np.mean(BT[:,1]),np.std(BT[:,1]))
        table.iloc[1,ind2] = "\\tcr{$%.4f_{%.4f}$}"%(np.mean(IBT[:,1]),np.std(IBT[:,1]))
        table.iloc[2,ind2] = "(\\tcr{$%.4f$})"%test1.pvalue
    else:
        table.iloc[1,ind2] = "$%.4f_{%.4f}$"%(np.mean(IBT[:,1]),np.std(IBT[:,1]))
        table.iloc[0,ind2] = "$%.4f_{%.4f}$"%(np.mean(BT[:,1]),np.std(BT[:,1]))
        table.iloc[2,ind2] = "($%.4f$)"%test1.pvalue
    if test5.pvalue<.05:
        table.iloc[3,ind2] = "\\tcr{$%.4f_{%.4f}$}"%(np.mean(BT[:,5]),np.std(BT[:,5]))
        table.iloc[4,ind2] = "\\tcr{$%.4f_{%.4f}$}"%(np.mean(IBT[:,5]),np.std(IBT[:,5]))
        table.iloc[5,ind2] = "(\\tcr{$%.4f$})"%test5.pvalue
    else:
        table.iloc[4,ind2] = "$%.4f_{%.4f}$"%(np.mean(IBT[:,5]),np.std(IBT[:,5]))
        table.iloc[3,ind2] = "$%.4f_{%.4f}$"%(np.mean(BT[:,5]),np.std(BT[:,5]))
        table.iloc[5,ind2] = "($%.4f$)"%test5.pvalue
    if test7.pvalue<.05:
        table.iloc[6,ind2] = "\\tcr{$%.4f_{%.4f}$}"%(np.mean(BT[:,7]),np.std(BT[:,7]))
        table.iloc[7,ind2] = "\\tcr{$%.4f_{%.4f}$}"%(np.mean(IBT[:,7]),np.std(IBT[:,7]))
        table.iloc[8,ind2] = "(\\tcr{$%.4f$})"%test7.pvalue
    else:
        table.iloc[7,ind2] = "$%.4f_{%.4f}$"%(np.mean(IBT[:,7]),np.std(IBT[:,7]))
        table.iloc[6,ind2] = "$%.4f_{%.4f}$"%(np.mean(BT[:,7]),np.std(BT[:,7]))
        table.iloc[8,ind2] = "($%.4f$)"%test7.pvalue

for i in range(9):
    for j in range(5):
        print(end="&")
        print(table.iloc[i,j],end="")
    print("\\\\")

# nll
# &$2.5804_{1.2334}$&$0.5358_{0.3756}$&$0.3328_{0.0542}$&$0.3189_{0.0318}$&$0.3112_{0.0334}$\\
# &$inf_{nan}$&$inf_{nan}$&$inf_{nan}$&$inf_{nan}$&$inf_{nan}$\\
# &($0.9870$)&($0.6934$)&($0.7872$)&($0.7159$)&($0.7064$)\\
# &$0.1736_{0.0932}$&\tcr{$0.3206_{0.0575}$}&\tcr{$0.3711_{0.0488}$}&\tcr{$0.3934_{0.0693}$}&\tcr{$0.4154_{0.1429}$}\\
# &$0.1779_{0.0969}$&\tcr{$0.3299_{0.0604}$}&\tcr{$0.3833_{0.0513}$}&\tcr{$0.4051_{0.0740}$}&\tcr{$0.4279_{0.1460}$}\\
# &($0.1219$)&(\tcr{$0.0001$})&(\tcr{$0.0000$})&(\tcr{$0.0001$})&(\tcr{$0.0232$})\\
# &$0.2335_{0.1240}$&$0.4257_{0.0735}$&$0.4880_{0.0607}$&$0.5133_{0.0838}$&$0.5309_{0.1719}$\\
# &$0.2262_{0.1223}$&$0.4213_{0.0739}$&$0.4828_{0.0603}$&$0.5071_{0.0854}$&$0.5258_{0.1709}$\\
# &($0.9162$)&($0.9210$)&($0.9771$)&($0.9348$)&($0.7577$)\\
# sq
# &\tcr{$0.2186_{0.0420}$}&$0.1336_{0.0274}$&$0.1012_{0.0136}$&$0.0932_{0.0153}$&$0.0881_{0.0274}$\\
# &\tcr{$0.2125_{0.0403}$}&$0.1310_{0.0224}$&$0.1027_{0.0138}$&$0.0942_{0.0156}$&$0.0889_{0.0276}$\\
# &(\tcr{$0.0004$})&($0.1613$)&($0.9892$)&($0.9223$)&($0.7480$)\\
# &$0.1759_{0.0914}$&\tcr{$0.3048_{0.0643}$}&\tcr{$0.3637_{0.0512}$}&\tcr{$0.3872_{0.0701}$}&\tcr{$0.4096_{0.1441}$}\\
# &$0.1800_{0.0939}$&\tcr{$0.3127_{0.0671}$}&\tcr{$0.3745_{0.0541}$}&\tcr{$0.3989_{0.0740}$}&\tcr{$0.4243_{0.1499}$}\\
# &($0.1398$)&(\tcr{$0.0016$})&(\tcr{$0.0000$})&(\tcr{$0.0002$})&(\tcr{$0.0149$})\\
# &$0.2361_{0.1212}$&$0.4053_{0.0829}$&$0.4787_{0.0640}$&$0.5056_{0.0853}$&$0.5244_{0.1736}$\\
# &$0.2312_{0.1197}$&$0.4005_{0.0831}$&$0.4732_{0.0651}$&$0.4997_{0.0865}$&$0.5215_{0.1739}$\\
# &($0.8294$)&($0.9070$)&($0.9640$)&($0.9334$)&($0.6646$)\\