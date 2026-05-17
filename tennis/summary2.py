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
# &$9.0269_{2.2872}$&$2.2744_{0.3300}$&$1.6829_{0.2006}$&$1.4985_{0.1946}$&$1.4047_{0.3220}$\\
# &$9.0269_{2.2872}$&$2.2744_{0.3300}$&$1.6829_{0.2006}$&$1.4985_{0.1946}$&$1.4047_{0.3220}$\\
# &($0.5000$)&($0.5000$)&($0.5000$)&($0.5000$)&($0.5000$)\\
# &$0.0678_{0.0239}$&\tcr{$0.1364_{0.0173}$}&\tcr{$0.1651_{0.0175}$}&\tcr{$0.1818_{0.0234}$}&\tcr{$0.1966_{0.0463}$}\\
# &$0.0693_{0.0244}$&\tcr{$0.1414_{0.0185}$}&\tcr{$0.1704_{0.0186}$}&\tcr{$0.1870_{0.0249}$}&\tcr{$0.2020_{0.0485}$}\\
# &($0.0814$)&(\tcr{$0.0000$})&(\tcr{$0.0000$})&(\tcr{$0.0000$})&(\tcr{$0.0064$})\\
# &$0.0832_{0.0293}$&$0.1690_{0.0216}$&$0.2047_{0.0217}$&$0.2255_{0.0292}$&$0.2436_{0.0574}$\\
# &$0.0818_{0.0290}$&$0.1683_{0.0219}$&$0.2036_{0.0218}$&$0.2243_{0.0293}$&$0.2425_{0.0578}$\\
# &($0.8150$)&($0.7762$)&($0.8663$)&($0.8121$)&($0.6636$)\\
# sq
# &\tcr{$0.4267_{0.0139}$}&\tcr{$0.4015_{0.0139}$}&\tcr{$0.2986_{0.0201}$}&\tcr{$0.2673_{0.0115}$}&$0.2543_{0.0177}$\\
# &\tcr{$0.4141_{0.0143}$}&\tcr{$0.3396_{0.0115}$}&\tcr{$0.2813_{0.0107}$}&\tcr{$0.2628_{0.0099}$}&$0.2529_{0.0171}$\\
# &(\tcr{$0.0000$})&(\tcr{$0.0000$})&(\tcr{$0.0000$})&(\tcr{$0.0000$})&($0.0593$)\\
# &$0.0646_{0.0266}$&\tcr{$0.1095_{0.0209}$}&\tcr{$0.1479_{0.0204}$}&\tcr{$0.1720_{0.0240}$}&\tcr{$0.1889_{0.0470}$}\\
# &$0.0664_{0.0273}$&\tcr{$0.1152_{0.0226}$}&\tcr{$0.1521_{0.0209}$}&\tcr{$0.1760_{0.0250}$}&\tcr{$0.1932_{0.0489}$}\\
# &($0.0697$)&(\tcr{$0.0000$})&(\tcr{$0.0000$})&(\tcr{$0.0002$})&(\tcr{$0.0190$})\\
# &$0.0786_{0.0324}$&$0.1348_{0.0258}$&$0.1824_{0.0254}$&$0.2128_{0.0298}$&$0.2338_{0.0583}$\\
# &$0.0780_{0.0320}$&$0.1328_{0.0252}$&$0.1815_{0.0254}$&$0.2115_{0.0300}$&$0.2328_{0.0583}$\\
# &($0.6567$)&($0.9591$)&($0.7777$)&($0.8434$)&($0.6398$)\\







