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
    if test7.pvalue<0.05:
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
# &$0.8440_{0.4711}$&$0.3604_{0.0484}$&$0.3491_{0.0046}$&$0.3459_{0.0046}$&$0.3444_{0.0078}$\\
# &$inf_{nan}$&$inf_{nan}$&$inf_{nan}$&$inf_{nan}$&$inf_{nan}$\\
# &($0.8242$)&($0.9027$)&($0.9894$)&($0.9837$)&($0.7897$)\\
# &$0.0642_{0.0516}$&\tcr{$0.1266_{0.0375}$}&\tcr{$0.1550_{0.0366}$}&\tcr{$0.1747_{0.0498}$}&\tcr{$0.1884_{0.0967}$}\\
# &$0.0654_{0.0531}$&\tcr{$0.1307_{0.0421}$}&\tcr{$0.1620_{0.0411}$}&\tcr{$0.1844_{0.0558}$}&\tcr{$0.2035_{0.1055}$}\\
# &($0.3034$)&(\tcr{$0.0041$})&(\tcr{$0.0001$})&(\tcr{$0.0000$})&(\tcr{$0.0004$})\\
# &$0.0903_{0.0726}$&$0.1775_{0.0523}$&$0.2171_{0.0508}$&$0.2445_{0.0685}$&$0.2620_{0.1317}$\\
# &$0.0894_{0.0726}$&$0.1736_{0.0551}$&$0.2128_{0.0520}$&$0.2405_{0.0704}$&$0.2613_{0.1323}$\\
# &($0.6080$)&($0.9060$)&($0.9650$)&($0.8663$)&($0.5071$)\\
# sq
# &\tcr{$0.1279_{0.0263}$}&$0.0700_{0.0059}$&$0.0626_{0.0048}$&$0.0597_{0.0063}$&$0.0582_{0.0118}$\\
# &\tcr{$0.1245_{0.0224}$}&$0.0704_{0.0061}$&$0.0629_{0.0049}$&$0.0600_{0.0064}$&$0.0586_{0.0121}$\\
# &(\tcr{$0.0201$})&($0.9633$)&($0.9188$)&($0.8654$)&($0.7673$)\\
# &$0.0637_{0.0517}$&\tcr{$0.1272_{0.0376}$}&\tcr{$0.1558_{0.0365}$}&\tcr{$0.1757_{0.0495}$}&\tcr{$0.1890_{0.0963}$}\\
# &$0.0650_{0.0530}$&\tcr{$0.1316_{0.0411}$}&\tcr{$0.1625_{0.0410}$}&\tcr{$0.1854_{0.0547}$}&\tcr{$0.2027_{0.1046}$}\\
# &($0.2890$)&(\tcr{$0.0065$})&(\tcr{$0.0001$})&(\tcr{$0.0000$})&(\tcr{$0.0010$})\\
# &$0.0896_{0.0727}$&$0.1784_{0.0525}$&$0.2182_{0.0507}$&$0.2459_{0.0682}$&$0.2631_{0.1314}$\\
# &$0.0890_{0.0725}$&$0.1751_{0.0539}$&$0.2136_{0.0520}$&$0.2426_{0.0685}$&$0.2612_{0.1314}$\\
# &($0.5745$)&($0.9193$)&($0.9680$)&($0.8497$)&($0.6314$)\\