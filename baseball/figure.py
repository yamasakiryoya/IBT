import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy import stats
plt.rcParams["font.size"] = 20
plt.rcParams['text.usetex'] = True
import warnings
warnings.simplefilter('ignore')

if os.path.isdir("Base/sq-objW")==False: os.makedirs("Base/sq-objW", exist_ok=True)
if os.path.isdir("Base/sq-objY")==False: os.makedirs("Base/sq-objY", exist_ok=True)
if os.path.isdir("Base/sq-rank1")==False: os.makedirs("Base/sq-rank1", exist_ok=True)
if os.path.isdir("Base/sq-rank2")==False: os.makedirs("Base/sq-rank2", exist_ok=True)
if os.path.isdir("Base/sq-rank3")==False: os.makedirs("Base/sq-rank3", exist_ok=True)
if os.path.isdir("Base/sq-rank4")==False: os.makedirs("Base/sq-rank4", exist_ok=True)
if os.path.isdir("Base/sq-rank5")==False: os.makedirs("Base/sq-rank5", exist_ok=True)
if os.path.isdir("Base/sq-rank6")==False: os.makedirs("Base/sq-rank6", exist_ok=True)


for r in [.1,.3,.5,.7,.9]:
    res = np.zeros((1000,10,32))
    for seed in range(1000):
        res[seed] = np.loadtxt("Results-sq/%f/error-%f-%d.csv"%(r,r,seed), delimiter=",")#np.nan_to_num(, posinf=10.**20)
        for k in range(10):
            if np.all(res[seed,k,:] == 0):
                res[seed,k,:16] = res[seed,k-1,16:]
                res[seed,k,16:] = res[seed,k-1,16:]

    mean = np.mean(res, axis=0)
    q1 = np.quantile(res, q=0.25, axis=0)
    q2 = np.quantile(res, q=0.50, axis=0)
    q3 = np.quantile(res, q=0.75, axis=0)
    tra1 = []; tes1 = []; tra2 = []; tes2 = []; tra3 = []; tes3 = []; tra4 = []; tes4 = []

    for ite in range(10):
        tra1.append(mean[ite,0]); tra2.append(q1[ite,0]); tra3.append(q2[ite,0]); tra4.append(q3[ite,0])
        tes1.append(mean[ite,1]); tes2.append(q1[ite,1]); tes3.append(q2[ite,1]); tes4.append(q3[ite,1])
        tra1.append(mean[ite,16]); tra2.append(q1[ite,16]); tra3.append(q2[ite,16]); tra4.append(q3[ite,16])
        tes1.append(mean[ite,17]); tes2.append(q1[ite,17]); tes3.append(q2[ite,17]); tes4.append(q3[ite,17])
    tra1 = np.array(tra1); tra2 = np.array(tra2)
    tes1 = np.array(tes1); tes2 = np.array(tes2)

    fig, ax = plt.subplots(figsize=(10, 6)); cmap = plt.get_cmap('jet')
    ax.plot(range(1,20+1), tra1, lw=1, color='r', label=r"${\rm error}_{\rm train}$")
    ax.scatter(range(1,20+1), tra1, s=50, marker="D", edgecolor='r', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tra3[k-1]*np.ones(10), color='r')
        if tra2[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra2[k-1]*np.ones(10), color='r')
        if tra4[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra4[k-1]*np.ones(10), color='r')
        ax.plot(k*np.ones(10), np.linspace(tra2[k-1],tra4[k-1],10), color='r')
        if tra1[k-1]==np.min(tra1): ax.scatter(k, np.min(tra1), marker="D", s=50, color='r', zorder=3)
    ax.plot(range(1,20+1), tes1, lw=1, color='b', label=r"${\rm error}_{\rm test}$")
    ax.scatter(range(1,20+1), tes1, s=50, edgecolor='b', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tes3[k-1]*np.ones(10), color='b', linestyle="--")
        if tes2[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes2[k-1]*np.ones(10), color='b', linestyle="--")
        if tes4[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes4[k-1]*np.ones(10), color='b', linestyle="--")
        ax.plot(k*np.ones(10), np.linspace(tes2[k-1],tes4[k-1],10), color='b', linestyle="--")
        if tes1[k-1]==np.min(tes1): ax.scatter(k, np.min(tes1), marker="o", s=50, color='b', zorder=3)
    ax.set_xticks([1,2,3,4,8,12,16,20])
    ax.set_xticklabels(["0,1","1,1","1,2","2,2","4,4","6,6","8,8","10,10"])
    ax.set_xlabel(r'\#update: $s,t$ of $(\hat{\sigma}^{[s]},(\hat{r}_i^{[t]})_{i\in[n]})$')
    ax.set_ylabel(r'WPP error')
    ax.set_xlim(0,21); ax.grid(True)
    plt.savefig("./Base/sq-objW/%f.png"%r, bbox_inches="tight", pad_inches=.02, facecolor=fig.get_facecolor(), dpi=100, edgecolor='none', metadata={'Software': None}, pil_kwargs={'optimize': True}); plt.close()#,format='webp', 

    test = stats.mannwhitneyu(res[:,np.argmin(tes1)//2,16*(np.argmin(tes1)%2)+1], res[:,0,1], alternative='less')
    if test.pvalue<=0.05: print("./Plots/sq-objW/%f.png"%r)

    mean = np.mean(res, axis=0)
    q1 = np.quantile(res, q=0.25, axis=0)
    q2 = np.quantile(res, q=0.50, axis=0)
    q3 = np.quantile(res, q=0.75, axis=0)
    tra1 = []; tes1 = []; tra2 = []; tes2 = []; tra3 = []; tes3 = []; tra4 = []; tes4 = []

    for ite in range(10):
        tra1.append(mean[ite,2]); tra2.append(q1[ite,2]); tra3.append(q2[ite,2]); tra4.append(q3[ite,2])
        tes1.append(mean[ite,3]); tes2.append(q1[ite,3]); tes3.append(q2[ite,3]); tes4.append(q3[ite,3])
        tra1.append(mean[ite,18]); tra2.append(q1[ite,18]); tra3.append(q2[ite,18]); tra4.append(q3[ite,18])
        tes1.append(mean[ite,19]); tes2.append(q1[ite,19]); tes3.append(q2[ite,19]); tes4.append(q3[ite,19])
    tra1 = np.array(tra1); tra2 = np.array(tra2)
    tes1 = np.array(tes1); tes2 = np.array(tes2)

    fig, ax = plt.subplots(figsize=(10, 6)); cmap = plt.get_cmap('jet')
    ax.plot(range(1,20+1), tra1, lw=1, color='r', label=r"${\rm error}_{\rm train}$")
    ax.scatter(range(1,20+1), tra1, s=50, marker="D", edgecolor='r', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tra3[k-1]*np.ones(10), color='r')
        if tra2[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra2[k-1]*np.ones(10), color='r')
        if tra4[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra4[k-1]*np.ones(10), color='r')
        ax.plot(k*np.ones(10), np.linspace(tra2[k-1],tra4[k-1],10), color='r')
        if tra1[k-1]==np.min(tra1): ax.scatter(k, np.min(tra1), marker="D", s=50, color='r', zorder=3)
    ax.plot(range(1,20+1), tes1, lw=1, color='b', label=r"${\rm error}_{\rm test}$")
    ax.scatter(range(1,20+1), tes1, s=50, edgecolor='b', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tes3[k-1]*np.ones(10), color='b', linestyle="--")
        if tes2[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes2[k-1]*np.ones(10), color='b', linestyle="--")
        if tes4[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes4[k-1]*np.ones(10), color='b', linestyle="--")
        ax.plot(k*np.ones(10), np.linspace(tes2[k-1],tes4[k-1],10), color='b', linestyle="--")
        if tes1[k-1]==np.min(tes1): ax.scatter(k, np.min(tes1), marker="o", s=50, color='b', zorder=3)
    ax.set_xticks([1,2,3,4,8,12,16,20])
    ax.set_xticklabels(["0,1","1,1","1,2","2,2","4,4","6,6","8,8","10,10"])
    ax.set_xlabel(r'\#update: $s,t$ of $(\hat{\sigma}^{[s]},(\hat{r}_i^{[t]})_{i\in[n]})$')
    ax.set_ylabel(r'WPP error')
    ax.set_xlim(0,21); ax.grid(True)
    plt.savefig("./Base/sq-objY/%f.png"%r, bbox_inches="tight", pad_inches=.02, facecolor=fig.get_facecolor(), dpi=100, edgecolor='none', metadata={'Software': None}, pil_kwargs={'optimize': True}); plt.close()#,format='webp', 

    mean = np.mean(res, axis=0)
    q1 = np.quantile(res, q=0.25, axis=0)
    q2 = np.quantile(res, q=0.50, axis=0)
    q3 = np.quantile(res, q=0.75, axis=0)
    tra1 = []; tes1 = []; tra2 = []; tes2 = []; tra3 = []; tes3 = []; tra4 = []; tes4 = []

    for ite in range(10):
        tra1.append(mean[ite,4]); tra2.append(q1[ite,4]); tra3.append(q2[ite,4]); tra4.append(q3[ite,4])
        tes1.append(mean[ite,5]); tes2.append(q1[ite,5]); tes3.append(q2[ite,5]); tes4.append(q3[ite,5])
        tra1.append(mean[ite,20]); tra2.append(q1[ite,20]); tra3.append(q2[ite,20]); tra4.append(q3[ite,20])
        tes1.append(mean[ite,21]); tes2.append(q1[ite,21]); tes3.append(q2[ite,21]); tes4.append(q3[ite,21])
    tra1 = np.array(tra1); tra2 = np.array(tra2)
    tes1 = np.array(tes1); tes2 = np.array(tes2)

    fig, ax = plt.subplots(figsize=(10, 6)); cmap = plt.get_cmap('jet')
    ax.plot(range(1,20+1), tra1, lw=1, color='r', label=r"${\rm error}_{\rm train}$")
    ax.scatter(range(1,20+1), tra1, s=50, marker="D", edgecolor='r', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tra3[k-1]*np.ones(10), color='r')
        if tra2[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra2[k-1]*np.ones(10), color='r')
        if tra4[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra4[k-1]*np.ones(10), color='r')
        ax.plot(k*np.ones(10), np.linspace(tra2[k-1],tra4[k-1],10), color='r')
        if tra1[k-1]==np.max(tra1): ax.scatter(k, np.max(tra1), marker="D", s=50, color='r', zorder=3)
    ax.plot(range(1,20+1), tes1, lw=1, color='b', label=r"${\rm error}_{\rm test}$")
    ax.scatter(range(1,20+1), tes1, s=50, edgecolor='b', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tes3[k-1]*np.ones(10), color='b', linestyle="--")
        if tes2[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes2[k-1]*np.ones(10), color='b', linestyle="--")
        if tes4[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes4[k-1]*np.ones(10), color='b', linestyle="--")
        ax.plot(k*np.ones(10), np.linspace(tes2[k-1],tes4[k-1],10), color='b', linestyle="--")
        if tes1[k-1]==np.max(tes1): ax.scatter(k, np.max(tes1), marker="o", s=50, color='b', zorder=3)
    ax.set_xticks([1,2,3,4,8,12,16,20])
    ax.set_xticklabels(["0,1","1,1","1,2","2,2","4,4","6,6","8,8","10,10"])
    ax.set_xlabel(r'\#update: $s,t$ of $(\hat{\sigma}^{[s]},(\hat{r}_i^{[t]})_{i\in[n]})$')
    ax.set_ylabel(r"Kendall's Tau")
    ax.set_xlim(0,21); ax.grid(True)
    plt.savefig("./Base/sq-rank1/%f.png"%r, bbox_inches="tight", pad_inches=.02, facecolor=fig.get_facecolor(), dpi=100, edgecolor='none', metadata={'Software': None}, pil_kwargs={'optimize': True}); plt.close()#,format='webp', 

    test = stats.mannwhitneyu(res[:,np.argmax(tes1)//2,16*(np.argmax(tes1)%2)+5], res[:,0,5], alternative='greater')
    if test.pvalue<=0.05: print("./Plots/sq-rank1/%f.png"%r)
    
    mean = np.mean(res, axis=0)
    q1 = np.quantile(res, q=0.25, axis=0)
    q2 = np.quantile(res, q=0.50, axis=0)
    q3 = np.quantile(res, q=0.75, axis=0)
    tra1 = []; tes1 = []; tra2 = []; tes2 = []; tra3 = []; tes3 = []; tra4 = []; tes4 = []

    for ite in range(10):
        tra1.append(mean[ite,6]); tra2.append(q1[ite,6]); tra3.append(q2[ite,6]); tra4.append(q3[ite,6])
        tes1.append(mean[ite,7]); tes2.append(q1[ite,7]); tes3.append(q2[ite,7]); tes4.append(q3[ite,7])
        tra1.append(mean[ite,22]); tra2.append(q1[ite,22]); tra3.append(q2[ite,22]); tra4.append(q3[ite,22])
        tes1.append(mean[ite,23]); tes2.append(q1[ite,23]); tes3.append(q2[ite,23]); tes4.append(q3[ite,23])
    tra1 = np.array(tra1); tra2 = np.array(tra2)
    tes1 = np.array(tes1); tes2 = np.array(tes2)

    fig, ax = plt.subplots(figsize=(10, 6)); cmap = plt.get_cmap('jet')
    ax.plot(range(1,20+1), tra1, lw=1, color='r', label=r"${\rm error}_{\rm train}$")
    ax.scatter(range(1,20+1), tra1, s=50, marker="D", edgecolor='r', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tra3[k-1]*np.ones(10), color='r')
        if tra2[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra2[k-1]*np.ones(10), color='r')
        if tra4[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra4[k-1]*np.ones(10), color='r')
        ax.plot(k*np.ones(10), np.linspace(tra2[k-1],tra4[k-1],10), color='r')
        if tra1[k-1]==np.max(tra1): ax.scatter(k, np.max(tra1), marker="D", s=50, color='r', zorder=3)
    ax.plot(range(1,20+1), tes1, lw=1, color='b', label=r"${\rm error}_{\rm test}$")
    ax.scatter(range(1,20+1), tes1, s=50, edgecolor='b', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tes3[k-1]*np.ones(10), color='b', linestyle="--")
        if tes2[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes2[k-1]*np.ones(10), color='b', linestyle="--")
        if tes4[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes4[k-1]*np.ones(10), color='b', linestyle="--")
        ax.plot(k*np.ones(10), np.linspace(tes2[k-1],tes4[k-1],10), color='b', linestyle="--")
        if tes1[k-1]==np.max(tes1): ax.scatter(k, np.max(tes1), marker="o", s=50, color='b', zorder=3)
    ax.set_xticks([1,2,3,4,8,12,16,20])
    ax.set_xticklabels(["0,1","1,1","1,2","2,2","4,4","6,6","8,8","10,10"])
    ax.set_xlabel(r'\#update: $s,t$ of $(\hat{\sigma}^{[s]},(\hat{r}_i^{[t]})_{i\in[n]})$')
    ax.set_ylabel(r"Spearman's Rho")
    ax.set_xlim(0,21); ax.grid(True)
    plt.savefig("./Base/sq-rank2/%f.png"%r, bbox_inches="tight", pad_inches=.02, facecolor=fig.get_facecolor(), dpi=100, edgecolor='none', metadata={'Software': None}, pil_kwargs={'optimize': True}); plt.close()#,format='webp', 

    test = stats.mannwhitneyu(res[:,np.argmax(tes1)//2,16*(np.argmax(tes1)%2)+7], res[:,0,7], alternative='greater')
    if test.pvalue<=0.05: print("./Plots/sq-rank2/%f.png"%r)

    mean = np.mean((1-res), axis=0)
    q1 = np.quantile((1-res), q=0.25, axis=0)
    q2 = np.quantile((1-res), q=0.50, axis=0)
    q3 = np.quantile((1-res), q=0.75, axis=0)
    tra1 = []; tes1 = []; tra2 = []; tes2 = []; tra3 = []; tes3 = []; tra4 = []; tes4 = []

    for ite in range(10):
        tra1.append(mean[ite,8]); tra2.append(q1[ite,8]); tra3.append(q2[ite,8]); tra4.append(q3[ite,8])
        tes1.append(mean[ite,9]); tes2.append(q1[ite,9]); tes3.append(q2[ite,9]); tes4.append(q3[ite,9])
        tra1.append(mean[ite,24]); tra2.append(q1[ite,24]); tra3.append(q2[ite,24]); tra4.append(q3[ite,24])
        tes1.append(mean[ite,25]); tes2.append(q1[ite,25]); tes3.append(q2[ite,25]); tes4.append(q3[ite,25])
    tra1 = np.array(tra1); tra2 = np.array(tra2)
    tes1 = np.array(tes1); tes2 = np.array(tes2)

    fig, ax = plt.subplots(figsize=(10, 6)); cmap = plt.get_cmap('jet');  plt.yscale('log')
    ax.plot(range(1,20+1), tra1, lw=1, color='r', label=r"${\rm error}_{\rm train}$")
    ax.scatter(range(1,20+1), tra1, s=50, marker="D", edgecolor='r', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tra3[k-1]*np.ones(10), color='r')
        if tra2[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra2[k-1]*np.ones(10), color='r')
        if tra4[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra4[k-1]*np.ones(10), color='r')
        ax.plot(k*np.ones(10), np.linspace(tra2[k-1],tra4[k-1],10), color='r')
        if tra1[k-1]==np.min(tra1): ax.scatter(k, np.min(tra1), marker="D", s=50, color='r', zorder=3)
    ax.plot(range(1,20+1), tes1, lw=1, color='b', label=r"${\rm error}_{\rm test}$")
    ax.scatter(range(1,20+1), tes1, s=50, edgecolor='b', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tes3[k-1]*np.ones(10), color='b', linestyle="--")
        if tes2[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes2[k-1]*np.ones(10), color='b', linestyle="--")
        if tes4[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes4[k-1]*np.ones(10), color='b', linestyle="--")
        ax.plot(k*np.ones(10), np.linspace(tes2[k-1],tes4[k-1],10), color='b', linestyle="--")
        if tes1[k-1]==np.min(tes1): ax.scatter(k, np.min(tes1), marker="o", s=50, color='b', zorder=3)
    ax.set_xticks([1,2,3,4,8,12,16,20])
    ax.set_xticklabels(["0,1","1,1","1,2","2,2","4,4","6,6","8,8","10,10"])
    ax.set_xlabel(r'\#update: $s,t$ of $(\hat{\sigma}^{[s]},(\hat{r}_i^{[t]})_{i\in[n]})$')
    ax.set_ylabel(r'Tie criterion')
    ax.set_xlim(0,21); ax.grid(True)
    plt.savefig("./Base/sq-rank3/%f.png"%r, bbox_inches="tight", pad_inches=.02, facecolor=fig.get_facecolor(), dpi=100, edgecolor='none', metadata={'Software': None}, pil_kwargs={'optimize': True}); plt.close()#,format='webp', 


    mean = np.mean(res, axis=0)
    q1 = np.quantile(res, q=0.25, axis=0)
    q2 = np.quantile(res, q=0.50, axis=0)
    q3 = np.quantile(res, q=0.75, axis=0)
    tra1 = []; tes1 = []; tra2 = []; tes2 = []; tra3 = []; tes3 = []; tra4 = []; tes4 = []

    for ite in range(10):
        tra1.append(mean[ite,10]); tra2.append(q1[ite,10]); tra3.append(q2[ite,10]); tra4.append(q3[ite,10])
        tes1.append(mean[ite,11]); tes2.append(q1[ite,11]); tes3.append(q2[ite,11]); tes4.append(q3[ite,11])
        tra1.append(mean[ite,26]); tra2.append(q1[ite,26]); tra3.append(q2[ite,26]); tra4.append(q3[ite,26])
        tes1.append(mean[ite,27]); tes2.append(q1[ite,27]); tes3.append(q2[ite,27]); tes4.append(q3[ite,27])
    tra1 = np.array(tra1); tra2 = np.array(tra2)
    tes1 = np.array(tes1); tes2 = np.array(tes2)

    fig, ax = plt.subplots(figsize=(10, 6)); cmap = plt.get_cmap('jet')
    ax.plot(range(1,20+1), tra1, lw=1, color='r', label=r"${\rm error}_{\rm train}$")
    ax.scatter(range(1,20+1), tra1, s=50, marker="D", edgecolor='r', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tra3[k-1]*np.ones(10), color='r')
        if tra2[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra2[k-1]*np.ones(10), color='r')
        if tra4[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra4[k-1]*np.ones(10), color='r')
        ax.plot(k*np.ones(10), np.linspace(tra2[k-1],tra4[k-1],10), color='r')
        if tra1[k-1]==np.max(tra1): ax.scatter(k, np.max(tra1), marker="D", s=50, color='r', zorder=3)
    ax.plot(range(1,20+1), tes1, lw=1, color='b', label=r"${\rm error}_{\rm test}$")
    ax.scatter(range(1,20+1), tes1, s=50, edgecolor='b', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tes3[k-1]*np.ones(10), color='b', linestyle="--")
        if tes2[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes2[k-1]*np.ones(10), color='b', linestyle="--")
        if tes4[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes4[k-1]*np.ones(10), color='b', linestyle="--")
        ax.plot(k*np.ones(10), np.linspace(tes2[k-1],tes4[k-1],10), color='b', linestyle="--")
        if tes1[k-1]==np.max(tes1): ax.scatter(k, np.max(tes1), marker="o", s=50, color='b', zorder=3)
    ax.set_xticks([1,2,3,4,8,12,16,20])
    ax.set_xticklabels(["0,1","1,1","1,2","2,2","4,4","6,6","8,8","10,10"])
    ax.set_xlabel(r'\#update: $s,t$ of $(\hat{\sigma}^{[s]},(\hat{r}_i^{[t]})_{i\in[n]})$')
    ax.set_ylabel(r"Kendall's Tau")
    ax.set_xlim(0,21); ax.grid(True)
    plt.savefig("./Base/sq-rank4/%f.png"%r, bbox_inches="tight", pad_inches=.02, facecolor=fig.get_facecolor(), dpi=100, edgecolor='none', metadata={'Software': None}, pil_kwargs={'optimize': True}); plt.close()#,format='webp', 


    mean = np.mean(res, axis=0)
    q1 = np.quantile(res, q=0.25, axis=0)
    q2 = np.quantile(res, q=0.50, axis=0)
    q3 = np.quantile(res, q=0.75, axis=0)
    tra1 = []; tes1 = []; tra2 = []; tes2 = []; tra3 = []; tes3 = []; tra4 = []; tes4 = []

    for ite in range(10):
        tra1.append(mean[ite,12]); tra2.append(q1[ite,12]); tra3.append(q2[ite,12]); tra4.append(q3[ite,12])
        tes1.append(mean[ite,13]); tes2.append(q1[ite,13]); tes3.append(q2[ite,13]); tes4.append(q3[ite,13])
        tra1.append(mean[ite,28]); tra2.append(q1[ite,28]); tra3.append(q2[ite,28]); tra4.append(q3[ite,28])
        tes1.append(mean[ite,29]); tes2.append(q1[ite,29]); tes3.append(q2[ite,29]); tes4.append(q3[ite,29])
    tra1 = np.array(tra1); tra2 = np.array(tra2)
    tes1 = np.array(tes1); tes2 = np.array(tes2)

    fig, ax = plt.subplots(figsize=(10, 6)); cmap = plt.get_cmap('jet')
    ax.plot(range(1,20+1), tra1, lw=1, color='r', label=r"${\rm error}_{\rm train}$")
    ax.scatter(range(1,20+1), tra1, s=50, marker="D", edgecolor='r', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tra3[k-1]*np.ones(10), color='r')
        if tra2[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra2[k-1]*np.ones(10), color='r')
        if tra4[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra4[k-1]*np.ones(10), color='r')
        ax.plot(k*np.ones(10), np.linspace(tra2[k-1],tra4[k-1],10), color='r')
        if tra1[k-1]==np.max(tra1): ax.scatter(k, np.max(tra1), marker="D", s=50, color='r', zorder=3)
    ax.plot(range(1,20+1), tes1, lw=1, color='b', label=r"${\rm error}_{\rm test}$")
    ax.scatter(range(1,20+1), tes1, s=50, edgecolor='b', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tes3[k-1]*np.ones(10), color='b', linestyle="--")
        if tes2[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes2[k-1]*np.ones(10), color='b', linestyle="--")
        if tes4[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes4[k-1]*np.ones(10), color='b', linestyle="--")
        ax.plot(k*np.ones(10), np.linspace(tes2[k-1],tes4[k-1],10), color='b', linestyle="--")
        if tes1[k-1]==np.max(tes1): ax.scatter(k, np.max(tes1), marker="o", s=50, color='b', zorder=3)
    ax.set_xticks([1,2,3,4,8,12,16,20])
    ax.set_xticklabels(["0,1","1,1","1,2","2,2","4,4","6,6","8,8","10,10"])
    ax.set_xlabel(r'\#update: $s,t$ of $(\hat{\sigma}^{[s]},(\hat{r}_i^{[t]})_{i\in[n]})$')
    ax.set_ylabel(r"Spearman's Rho")
    ax.set_xlim(0,21); ax.grid(True)
    plt.savefig("./Base/sq-rank5/%f.png"%r, bbox_inches="tight", pad_inches=.02, facecolor=fig.get_facecolor(), dpi=100, edgecolor='none', metadata={'Software': None}, pil_kwargs={'optimize': True}); plt.close()#,format='webp', 


    mean = np.mean((1-res), axis=0)
    q1 = np.quantile((1-res), q=0.25, axis=0)
    q2 = np.quantile((1-res), q=0.50, axis=0)
    q3 = np.quantile((1-res), q=0.75, axis=0)
    tra1 = []; tes1 = []; tra2 = []; tes2 = []; tra3 = []; tes3 = []; tra4 = []; tes4 = []

    for ite in range(10):
        tra1.append(mean[ite,14]); tra2.append(q1[ite,14]); tra3.append(q2[ite,14]); tra4.append(q3[ite,14])
        tes1.append(mean[ite,15]); tes2.append(q1[ite,15]); tes3.append(q2[ite,15]); tes4.append(q3[ite,15])
        tra1.append(mean[ite,30]); tra2.append(q1[ite,30]); tra3.append(q2[ite,30]); tra4.append(q3[ite,30])
        tes1.append(mean[ite,31]); tes2.append(q1[ite,31]); tes3.append(q2[ite,31]); tes4.append(q3[ite,31])
    tra1 = np.array(tra1); tra2 = np.array(tra2)
    tes1 = np.array(tes1); tes2 = np.array(tes2)

    fig, ax = plt.subplots(figsize=(10, 6)); cmap = plt.get_cmap('jet');  plt.yscale('log')
    ax.plot(range(1,20+1), tra1, lw=1, color='r', label=r"${\rm error}_{\rm train}$")
    ax.scatter(range(1,20+1), tra1, s=50, marker="D", edgecolor='r', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tra3[k-1]*np.ones(10), color='r')
        if tra2[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra2[k-1]*np.ones(10), color='r')
        if tra4[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra4[k-1]*np.ones(10), color='r')
        ax.plot(k*np.ones(10), np.linspace(tra2[k-1],tra4[k-1],10), color='r')
        if tra1[k-1]==np.min(tra1): ax.scatter(k, np.min(tra1), marker="D", s=50, color='r', zorder=3)
    ax.plot(range(1,20+1), tes1, lw=1, color='b', label=r"${\rm error}_{\rm test}$")
    ax.scatter(range(1,20+1), tes1, s=50, edgecolor='b', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tes3[k-1]*np.ones(10), color='b', linestyle="--")
        if tes2[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes2[k-1]*np.ones(10), color='b', linestyle="--")
        if tes4[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes4[k-1]*np.ones(10), color='b', linestyle="--")
        ax.plot(k*np.ones(10), np.linspace(tes2[k-1],tes4[k-1],10), color='b', linestyle="--")
        if tes1[k-1]==np.min(tes1): ax.scatter(k, np.min(tes1), marker="o", s=50, color='b', zorder=3)
    ax.set_xticks([1,2,3,4,8,12,16,20])
    ax.set_xticklabels(["0,1","1,1","1,2","2,2","4,4","6,6","8,8","10,10"])
    ax.set_xlabel(r'\#update: $s,t$ of $(\hat{\sigma}^{[s]},(\hat{r}_i^{[t]})_{i\in[n]})$')
    ax.set_ylabel(r'Tie criterion')
    ax.set_xlim(0,21); ax.grid(True)
    plt.savefig("./Base/sq-rank6/%f.png"%r, bbox_inches="tight", pad_inches=.02, facecolor=fig.get_facecolor(), dpi=100, edgecolor='none', metadata={'Software': None}, pil_kwargs={'optimize': True}); plt.close()#,format='webp', 


if os.path.isdir("Base/sq-objW")==False: os.makedirs("Base/sq-objW", exist_ok=True)
if os.path.isdir("Base/sq-objY")==False: os.makedirs("Base/sq-objY", exist_ok=True)
if os.path.isdir("Base/sq-rank1")==False: os.makedirs("Base/sq-rank1", exist_ok=True)
if os.path.isdir("Base/sq-rank2")==False: os.makedirs("Base/sq-rank2", exist_ok=True)
if os.path.isdir("Base/sq-rank3")==False: os.makedirs("Base/sq-rank3", exist_ok=True)
if os.path.isdir("Base/sq-rank4")==False: os.makedirs("Base/sq-rank4", exist_ok=True)
if os.path.isdir("Base/sq-rank5")==False: os.makedirs("Base/sq-rank5", exist_ok=True)
if os.path.isdir("Base/sq-rank6")==False: os.makedirs("Base/sq-rank6", exist_ok=True)


for r in [.1,.3,.5,.7,.9]:
    res = np.zeros((100,10,32))
    for seed in range(100):
        res[seed] = np.loadtxt("Results-sq/%f/error-%f-%d.csv"%(r,r,seed), delimiter=",")#np.nan_to_num(, posinf=10.**20)
        for k in range(10):
            if np.all(res[seed,k,:] == 0):
                res[seed,k,:16] = res[seed,k-1,16:]
                res[seed,k,16:] = res[seed,k-1,16:]

    mean = np.mean(res, axis=0)
    q1 = np.quantile(res, q=0.25, axis=0)
    q2 = np.quantile(res, q=0.50, axis=0)
    q3 = np.quantile(res, q=0.75, axis=0)
    tra1 = []; tes1 = []; tra2 = []; tes2 = []; tra3 = []; tes3 = []; tra4 = []; tes4 = []

    for ite in range(10):
        tra1.append(mean[ite,0]); tra2.append(q1[ite,0]); tra3.append(q2[ite,0]); tra4.append(q3[ite,0])
        tes1.append(mean[ite,1]); tes2.append(q1[ite,1]); tes3.append(q2[ite,1]); tes4.append(q3[ite,1])
        tra1.append(mean[ite,16]); tra2.append(q1[ite,16]); tra3.append(q2[ite,16]); tra4.append(q3[ite,16])
        tes1.append(mean[ite,17]); tes2.append(q1[ite,17]); tes3.append(q2[ite,17]); tes4.append(q3[ite,17])
    tra1 = np.array(tra1); tra2 = np.array(tra2)
    tes1 = np.array(tes1); tes2 = np.array(tes2)

    fig, ax = plt.subplots(figsize=(10, 6)); cmap = plt.get_cmap('jet')
    ax.plot(range(1,20+1), tra1, lw=1, color='r', label=r"${\rm error}_{\rm train}$")
    ax.scatter(range(1,20+1), tra1, s=50, marker="D", edgecolor='r', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tra3[k-1]*np.ones(10), color='r')
        if tra2[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra2[k-1]*np.ones(10), color='r')
        if tra4[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra4[k-1]*np.ones(10), color='r')
        ax.plot(k*np.ones(10), np.linspace(tra2[k-1],tra4[k-1],10), color='r')
        if tra1[k-1]==np.min(tra1): ax.scatter(k, np.min(tra1), marker="D", s=50, color='r', zorder=3)
    ax.plot(range(1,20+1), tes1, lw=1, color='b', label=r"${\rm error}_{\rm test}$")
    ax.scatter(range(1,20+1), tes1, s=50, edgecolor='b', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tes3[k-1]*np.ones(10), color='b', linestyle="--")
        if tes2[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes2[k-1]*np.ones(10), color='b', linestyle="--")
        if tes4[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes4[k-1]*np.ones(10), color='b', linestyle="--")
        ax.plot(k*np.ones(10), np.linspace(tes2[k-1],tes4[k-1],10), color='b', linestyle="--")
        if tes1[k-1]==np.min(tes1): ax.scatter(k, np.min(tes1), marker="o", s=50, color='b', zorder=3)
    ax.set_xticks([1,2,3,4,8,12,16,20])
    ax.set_xticklabels(["0,1","1,1","1,2","2,2","4,4","6,6","8,8","10,10"])
    ax.set_xlabel(r'\#update: $s,t$ of $(\hat{\sigma}^{[s]},(\hat{r}_i^{[t]})_{i\in[n]})$')
    ax.set_ylabel(r'WPP error')
    ax.set_xlim(0,21); ax.grid(True)
    plt.savefig("./Base/sq-objW/%f.png"%r, bbox_inches="tight", pad_inches=.02, facecolor=fig.get_facecolor(), dpi=100, edgecolor='none', metadata={'Software': None}, pil_kwargs={'optimize': True}); plt.close()#,format='webp', 

    mean = np.mean(res, axis=0)
    q1 = np.quantile(res, q=0.25, axis=0)
    q2 = np.quantile(res, q=0.50, axis=0)
    q3 = np.quantile(res, q=0.75, axis=0)
    tra1 = []; tes1 = []; tra2 = []; tes2 = []; tra3 = []; tes3 = []; tra4 = []; tes4 = []

    for ite in range(10):
        tra1.append(mean[ite,2]); tra2.append(q1[ite,2]); tra3.append(q2[ite,2]); tra4.append(q3[ite,2])
        tes1.append(mean[ite,3]); tes2.append(q1[ite,3]); tes3.append(q2[ite,3]); tes4.append(q3[ite,3])
        tra1.append(mean[ite,18]); tra2.append(q1[ite,18]); tra3.append(q2[ite,18]); tra4.append(q3[ite,18])
        tes1.append(mean[ite,19]); tes2.append(q1[ite,19]); tes3.append(q2[ite,19]); tes4.append(q3[ite,19])
    tra1 = np.array(tra1); tra2 = np.array(tra2)
    tes1 = np.array(tes1); tes2 = np.array(tes2)

    fig, ax = plt.subplots(figsize=(10, 6)); cmap = plt.get_cmap('jet')
    ax.plot(range(1,20+1), tra1, lw=1, color='r', label=r"${\rm error}_{\rm train}$")
    ax.scatter(range(1,20+1), tra1, s=50, marker="D", edgecolor='r', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tra3[k-1]*np.ones(10), color='r')
        if tra2[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra2[k-1]*np.ones(10), color='r')
        if tra4[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra4[k-1]*np.ones(10), color='r')
        ax.plot(k*np.ones(10), np.linspace(tra2[k-1],tra4[k-1],10), color='r')
        if tra1[k-1]==np.min(tra1): ax.scatter(k, np.min(tra1), marker="D", s=50, color='r', zorder=3)
    ax.plot(range(1,20+1), tes1, lw=1, color='b', label=r"${\rm error}_{\rm test}$")
    ax.scatter(range(1,20+1), tes1, s=50, edgecolor='b', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tes3[k-1]*np.ones(10), color='b', linestyle="--")
        if tes2[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes2[k-1]*np.ones(10), color='b', linestyle="--")
        if tes4[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes4[k-1]*np.ones(10), color='b', linestyle="--")
        ax.plot(k*np.ones(10), np.linspace(tes2[k-1],tes4[k-1],10), color='b', linestyle="--")
        if tes1[k-1]==np.min(tes1): ax.scatter(k, np.min(tes1), marker="o", s=50, color='b', zorder=3)
    ax.set_xticks([1,2,3,4,8,12,16,20])
    ax.set_xticklabels(["0,1","1,1","1,2","2,2","4,4","6,6","8,8","10,10"])
    ax.set_xlabel(r'\#update: $s,t$ of $(\hat{\sigma}^{[s]},(\hat{r}_i^{[t]})_{i\in[n]})$')
    ax.set_ylabel(r'WPP error')
    ax.set_xlim(0,21); ax.grid(True)
    plt.savefig("./Base/sq-objY/%f.png"%r, bbox_inches="tight", pad_inches=.02, facecolor=fig.get_facecolor(), dpi=100, edgecolor='none', metadata={'Software': None}, pil_kwargs={'optimize': True}); plt.close()#,format='webp', 

    mean = np.mean(res, axis=0)
    q1 = np.quantile(res, q=0.25, axis=0)
    q2 = np.quantile(res, q=0.50, axis=0)
    q3 = np.quantile(res, q=0.75, axis=0)
    tra1 = []; tes1 = []; tra2 = []; tes2 = []; tra3 = []; tes3 = []; tra4 = []; tes4 = []

    for ite in range(10):
        tra1.append(mean[ite,4]); tra2.append(q1[ite,4]); tra3.append(q2[ite,4]); tra4.append(q3[ite,4])
        tes1.append(mean[ite,5]); tes2.append(q1[ite,5]); tes3.append(q2[ite,5]); tes4.append(q3[ite,5])
        tra1.append(mean[ite,20]); tra2.append(q1[ite,20]); tra3.append(q2[ite,20]); tra4.append(q3[ite,20])
        tes1.append(mean[ite,21]); tes2.append(q1[ite,21]); tes3.append(q2[ite,21]); tes4.append(q3[ite,21])
    tra1 = np.array(tra1); tra2 = np.array(tra2)
    tes1 = np.array(tes1); tes2 = np.array(tes2)

    fig, ax = plt.subplots(figsize=(10, 6)); cmap = plt.get_cmap('jet')
    ax.plot(range(1,20+1), tra1, lw=1, color='r', label=r"${\rm error}_{\rm train}$")
    ax.scatter(range(1,20+1), tra1, s=50, marker="D", edgecolor='r', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tra3[k-1]*np.ones(10), color='r')
        if tra2[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra2[k-1]*np.ones(10), color='r')
        if tra4[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra4[k-1]*np.ones(10), color='r')
        ax.plot(k*np.ones(10), np.linspace(tra2[k-1],tra4[k-1],10), color='r')
        if tra1[k-1]==np.max(tra1): ax.scatter(k, np.max(tra1), marker="D", s=50, color='r', zorder=3)
    ax.plot(range(1,20+1), tes1, lw=1, color='b', label=r"${\rm error}_{\rm test}$")
    ax.scatter(range(1,20+1), tes1, s=50, edgecolor='b', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tes3[k-1]*np.ones(10), color='b', linestyle="--")
        if tes2[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes2[k-1]*np.ones(10), color='b', linestyle="--")
        if tes4[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes4[k-1]*np.ones(10), color='b', linestyle="--")
        ax.plot(k*np.ones(10), np.linspace(tes2[k-1],tes4[k-1],10), color='b', linestyle="--")
        if tes1[k-1]==np.max(tes1): ax.scatter(k, np.max(tes1), marker="o", s=50, color='b', zorder=3)
    ax.set_xticks([1,2,3,4,8,12,16,20])
    ax.set_xticklabels(["0,1","1,1","1,2","2,2","4,4","6,6","8,8","10,10"])
    ax.set_xlabel(r'\#update: $s,t$ of $(\hat{\sigma}^{[s]},(\hat{r}_i^{[t]})_{i\in[n]})$')
    ax.set_ylabel(r"Kendall's Tau")
    ax.set_xlim(0,21); ax.grid(True)
    plt.savefig("./Base/sq-rank1/%f.png"%r, bbox_inches="tight", pad_inches=.02, facecolor=fig.get_facecolor(), dpi=100, edgecolor='none', metadata={'Software': None}, pil_kwargs={'optimize': True}); plt.close()#,format='webp', 

    mean = np.mean(res, axis=0)
    q1 = np.quantile(res, q=0.25, axis=0)
    q2 = np.quantile(res, q=0.50, axis=0)
    q3 = np.quantile(res, q=0.75, axis=0)
    tra1 = []; tes1 = []; tra2 = []; tes2 = []; tra3 = []; tes3 = []; tra4 = []; tes4 = []

    for ite in range(10):
        tra1.append(mean[ite,6]); tra2.append(q1[ite,6]); tra3.append(q2[ite,6]); tra4.append(q3[ite,6])
        tes1.append(mean[ite,7]); tes2.append(q1[ite,7]); tes3.append(q2[ite,7]); tes4.append(q3[ite,7])
        tra1.append(mean[ite,22]); tra2.append(q1[ite,22]); tra3.append(q2[ite,22]); tra4.append(q3[ite,22])
        tes1.append(mean[ite,23]); tes2.append(q1[ite,23]); tes3.append(q2[ite,23]); tes4.append(q3[ite,23])
    tra1 = np.array(tra1); tra2 = np.array(tra2)
    tes1 = np.array(tes1); tes2 = np.array(tes2)

    fig, ax = plt.subplots(figsize=(10, 6)); cmap = plt.get_cmap('jet')
    ax.plot(range(1,20+1), tra1, lw=1, color='r', label=r"${\rm error}_{\rm train}$")
    ax.scatter(range(1,20+1), tra1, s=50, marker="D", edgecolor='r', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tra3[k-1]*np.ones(10), color='r')
        if tra2[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra2[k-1]*np.ones(10), color='r')
        if tra4[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra4[k-1]*np.ones(10), color='r')
        ax.plot(k*np.ones(10), np.linspace(tra2[k-1],tra4[k-1],10), color='r')
        if tra1[k-1]==np.max(tra1): ax.scatter(k, np.max(tra1), marker="D", s=50, color='r', zorder=3)
    ax.plot(range(1,20+1), tes1, lw=1, color='b', label=r"${\rm error}_{\rm test}$")
    ax.scatter(range(1,20+1), tes1, s=50, edgecolor='b', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tes3[k-1]*np.ones(10), color='b', linestyle="--")
        if tes2[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes2[k-1]*np.ones(10), color='b', linestyle="--")
        if tes4[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes4[k-1]*np.ones(10), color='b', linestyle="--")
        ax.plot(k*np.ones(10), np.linspace(tes2[k-1],tes4[k-1],10), color='b', linestyle="--")
        if tes1[k-1]==np.max(tes1): ax.scatter(k, np.max(tes1), marker="o", s=50, color='b', zorder=3)
    ax.set_xticks([1,2,3,4,8,12,16,20])
    ax.set_xticklabels(["0,1","1,1","1,2","2,2","4,4","6,6","8,8","10,10"])
    ax.set_xlabel(r'\#update: $s,t$ of $(\hat{\sigma}^{[s]},(\hat{r}_i^{[t]})_{i\in[n]})$')
    ax.set_ylabel(r"Spearman's Rho")
    ax.set_xlim(0,21); ax.grid(True)
    plt.savefig("./Base/sq-rank2/%f.png"%r, bbox_inches="tight", pad_inches=.02, facecolor=fig.get_facecolor(), dpi=100, edgecolor='none', metadata={'Software': None}, pil_kwargs={'optimize': True}); plt.close()#,format='webp', 

    mean = np.mean((1-res), axis=0)
    q1 = np.quantile((1-res), q=0.25, axis=0)
    q2 = np.quantile((1-res), q=0.50, axis=0)
    q3 = np.quantile((1-res), q=0.75, axis=0)
    tra1 = []; tes1 = []; tra2 = []; tes2 = []; tra3 = []; tes3 = []; tra4 = []; tes4 = []

    for ite in range(10):
        tra1.append(mean[ite,8]); tra2.append(q1[ite,8]); tra3.append(q2[ite,8]); tra4.append(q3[ite,8])
        tes1.append(mean[ite,9]); tes2.append(q1[ite,9]); tes3.append(q2[ite,9]); tes4.append(q3[ite,9])
        tra1.append(mean[ite,24]); tra2.append(q1[ite,24]); tra3.append(q2[ite,24]); tra4.append(q3[ite,24])
        tes1.append(mean[ite,25]); tes2.append(q1[ite,25]); tes3.append(q2[ite,25]); tes4.append(q3[ite,25])
    tra1 = np.array(tra1); tra2 = np.array(tra2)
    tes1 = np.array(tes1); tes2 = np.array(tes2)

    fig, ax = plt.subplots(figsize=(10, 6)); cmap = plt.get_cmap('jet');  plt.yscale('log')
    ax.plot(range(1,20+1), tra1, lw=1, color='r', label=r"${\rm error}_{\rm train}$")
    ax.scatter(range(1,20+1), tra1, s=50, marker="D", edgecolor='r', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tra3[k-1]*np.ones(10), color='r')
        if tra2[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra2[k-1]*np.ones(10), color='r')
        if tra4[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra4[k-1]*np.ones(10), color='r')
        ax.plot(k*np.ones(10), np.linspace(tra2[k-1],tra4[k-1],10), color='r')
        if tra1[k-1]==np.min(tra1): ax.scatter(k, np.min(tra1), marker="D", s=50, color='r', zorder=3)
    ax.plot(range(1,20+1), tes1, lw=1, color='b', label=r"${\rm error}_{\rm test}$")
    ax.scatter(range(1,20+1), tes1, s=50, edgecolor='b', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tes3[k-1]*np.ones(10), color='b', linestyle="--")
        if tes2[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes2[k-1]*np.ones(10), color='b', linestyle="--")
        if tes4[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes4[k-1]*np.ones(10), color='b', linestyle="--")
        ax.plot(k*np.ones(10), np.linspace(tes2[k-1],tes4[k-1],10), color='b', linestyle="--")
        if tes1[k-1]==np.min(tes1): ax.scatter(k, np.min(tes1), marker="o", s=50, color='b', zorder=3)
    ax.set_xticks([1,2,3,4,8,12,16,20])
    ax.set_xticklabels(["0,1","1,1","1,2","2,2","4,4","6,6","8,8","10,10"])
    ax.set_xlabel(r'\#update: $s,t$ of $(\hat{\sigma}^{[s]},(\hat{r}_i^{[t]})_{i\in[n]})$')
    ax.set_ylabel(r'Tie criterion')
    ax.set_xlim(0,21); ax.grid(True)
    plt.savefig("./Base/sq-rank3/%f.png"%r, bbox_inches="tight", pad_inches=.02, facecolor=fig.get_facecolor(), dpi=100, edgecolor='none', metadata={'Software': None}, pil_kwargs={'optimize': True}); plt.close()#,format='webp', 


    mean = np.mean(res, axis=0)
    q1 = np.quantile(res, q=0.25, axis=0)
    q2 = np.quantile(res, q=0.50, axis=0)
    q3 = np.quantile(res, q=0.75, axis=0)
    tra1 = []; tes1 = []; tra2 = []; tes2 = []; tra3 = []; tes3 = []; tra4 = []; tes4 = []

    for ite in range(10):
        tra1.append(mean[ite,10]); tra2.append(q1[ite,10]); tra3.append(q2[ite,10]); tra4.append(q3[ite,10])
        tes1.append(mean[ite,11]); tes2.append(q1[ite,11]); tes3.append(q2[ite,11]); tes4.append(q3[ite,11])
        tra1.append(mean[ite,26]); tra2.append(q1[ite,26]); tra3.append(q2[ite,26]); tra4.append(q3[ite,26])
        tes1.append(mean[ite,27]); tes2.append(q1[ite,27]); tes3.append(q2[ite,27]); tes4.append(q3[ite,27])
    tra1 = np.array(tra1); tra2 = np.array(tra2)
    tes1 = np.array(tes1); tes2 = np.array(tes2)

    fig, ax = plt.subplots(figsize=(10, 6)); cmap = plt.get_cmap('jet')
    ax.plot(range(1,20+1), tra1, lw=1, color='r', label=r"${\rm error}_{\rm train}$")
    ax.scatter(range(1,20+1), tra1, s=50, marker="D", edgecolor='r', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tra3[k-1]*np.ones(10), color='r')
        if tra2[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra2[k-1]*np.ones(10), color='r')
        if tra4[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra4[k-1]*np.ones(10), color='r')
        ax.plot(k*np.ones(10), np.linspace(tra2[k-1],tra4[k-1],10), color='r')
        if tra1[k-1]==np.max(tra1): ax.scatter(k, np.max(tra1), marker="D", s=50, color='r', zorder=3)
    ax.plot(range(1,20+1), tes1, lw=1, color='b', label=r"${\rm error}_{\rm test}$")
    ax.scatter(range(1,20+1), tes1, s=50, edgecolor='b', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tes3[k-1]*np.ones(10), color='b', linestyle="--")
        if tes2[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes2[k-1]*np.ones(10), color='b', linestyle="--")
        if tes4[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes4[k-1]*np.ones(10), color='b', linestyle="--")
        ax.plot(k*np.ones(10), np.linspace(tes2[k-1],tes4[k-1],10), color='b', linestyle="--")
        if tes1[k-1]==np.max(tes1): ax.scatter(k, np.max(tes1), marker="o", s=50, color='b', zorder=3)
    ax.set_xticks([1,2,3,4,8,12,16,20])
    ax.set_xticklabels(["0,1","1,1","1,2","2,2","4,4","6,6","8,8","10,10"])
    ax.set_xlabel(r'\#update: $s,t$ of $(\hat{\sigma}^{[s]},(\hat{r}_i^{[t]})_{i\in[n]})$')
    ax.set_ylabel(r"Kendall's Tau")
    ax.set_xlim(0,21); ax.grid(True)
    plt.savefig("./Base/sq-rank4/%f.png"%r, bbox_inches="tight", pad_inches=.02, facecolor=fig.get_facecolor(), dpi=100, edgecolor='none', metadata={'Software': None}, pil_kwargs={'optimize': True}); plt.close()#,format='webp', 


    mean = np.mean(res, axis=0)
    q1 = np.quantile(res, q=0.25, axis=0)
    q2 = np.quantile(res, q=0.50, axis=0)
    q3 = np.quantile(res, q=0.75, axis=0)
    tra1 = []; tes1 = []; tra2 = []; tes2 = []; tra3 = []; tes3 = []; tra4 = []; tes4 = []

    for ite in range(10):
        tra1.append(mean[ite,12]); tra2.append(q1[ite,12]); tra3.append(q2[ite,12]); tra4.append(q3[ite,12])
        tes1.append(mean[ite,13]); tes2.append(q1[ite,13]); tes3.append(q2[ite,13]); tes4.append(q3[ite,13])
        tra1.append(mean[ite,28]); tra2.append(q1[ite,28]); tra3.append(q2[ite,28]); tra4.append(q3[ite,28])
        tes1.append(mean[ite,29]); tes2.append(q1[ite,29]); tes3.append(q2[ite,29]); tes4.append(q3[ite,29])
    tra1 = np.array(tra1); tra2 = np.array(tra2)
    tes1 = np.array(tes1); tes2 = np.array(tes2)

    fig, ax = plt.subplots(figsize=(10, 6)); cmap = plt.get_cmap('jet')
    ax.plot(range(1,20+1), tra1, lw=1, color='r', label=r"${\rm error}_{\rm train}$")
    ax.scatter(range(1,20+1), tra1, s=50, marker="D", edgecolor='r', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tra3[k-1]*np.ones(10), color='r')
        if tra2[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra2[k-1]*np.ones(10), color='r')
        if tra4[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra4[k-1]*np.ones(10), color='r')
        ax.plot(k*np.ones(10), np.linspace(tra2[k-1],tra4[k-1],10), color='r')
        if tra1[k-1]==np.max(tra1): ax.scatter(k, np.max(tra1), marker="D", s=50, color='r', zorder=3)
    ax.plot(range(1,20+1), tes1, lw=1, color='b', label=r"${\rm error}_{\rm test}$")
    ax.scatter(range(1,20+1), tes1, s=50, edgecolor='b', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tes3[k-1]*np.ones(10), color='b', linestyle="--")
        if tes2[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes2[k-1]*np.ones(10), color='b', linestyle="--")
        if tes4[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes4[k-1]*np.ones(10), color='b', linestyle="--")
        ax.plot(k*np.ones(10), np.linspace(tes2[k-1],tes4[k-1],10), color='b', linestyle="--")
        if tes1[k-1]==np.max(tes1): ax.scatter(k, np.max(tes1), marker="o", s=50, color='b', zorder=3)
    ax.set_xticks([1,2,3,4,8,12,16,20])
    ax.set_xticklabels(["0,1","1,1","1,2","2,2","4,4","6,6","8,8","10,10"])
    ax.set_xlabel(r'\#update: $s,t$ of $(\hat{\sigma}^{[s]},(\hat{r}_i^{[t]})_{i\in[n]})$')
    ax.set_ylabel(r"Spearman's Rho")
    ax.set_xlim(0,21); ax.grid(True)
    plt.savefig("./Base/sq-rank5/%f.png"%r, bbox_inches="tight", pad_inches=.02, facecolor=fig.get_facecolor(), dpi=100, edgecolor='none', metadata={'Software': None}, pil_kwargs={'optimize': True}); plt.close()#,format='webp', 


    mean = np.mean((1-res), axis=0)
    q1 = np.quantile((1-res), q=0.25, axis=0)
    q2 = np.quantile((1-res), q=0.50, axis=0)
    q3 = np.quantile((1-res), q=0.75, axis=0)
    tra1 = []; tes1 = []; tra2 = []; tes2 = []; tra3 = []; tes3 = []; tra4 = []; tes4 = []

    for ite in range(10):
        tra1.append(mean[ite,14]); tra2.append(q1[ite,14]); tra3.append(q2[ite,14]); tra4.append(q3[ite,14])
        tes1.append(mean[ite,15]); tes2.append(q1[ite,15]); tes3.append(q2[ite,15]); tes4.append(q3[ite,15])
        tra1.append(mean[ite,30]); tra2.append(q1[ite,30]); tra3.append(q2[ite,30]); tra4.append(q3[ite,30])
        tes1.append(mean[ite,31]); tes2.append(q1[ite,31]); tes3.append(q2[ite,31]); tes4.append(q3[ite,31])
    tra1 = np.array(tra1); tra2 = np.array(tra2)
    tes1 = np.array(tes1); tes2 = np.array(tes2)

    fig, ax = plt.subplots(figsize=(10, 6)); cmap = plt.get_cmap('jet');  plt.yscale('log')
    ax.plot(range(1,20+1), tra1, lw=1, color='r', label=r"${\rm error}_{\rm train}$")
    ax.scatter(range(1,20+1), tra1, s=50, marker="D", edgecolor='r', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tra3[k-1]*np.ones(10), color='r')
        if tra2[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra2[k-1]*np.ones(10), color='r')
        if tra4[k-1]!=tra3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tra4[k-1]*np.ones(10), color='r')
        ax.plot(k*np.ones(10), np.linspace(tra2[k-1],tra4[k-1],10), color='r')
        if tra1[k-1]==np.min(tra1): ax.scatter(k, np.min(tra1), marker="D", s=50, color='r', zorder=3)
    ax.plot(range(1,20+1), tes1, lw=1, color='b', label=r"${\rm error}_{\rm test}$")
    ax.scatter(range(1,20+1), tes1, s=50, edgecolor='b', facecolor='none', zorder=2)
    for k in range(1,20+1):
        ax.plot(np.linspace(k-0.4,k+0.4,10), tes3[k-1]*np.ones(10), color='b', linestyle="--")
        if tes2[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes2[k-1]*np.ones(10), color='b', linestyle="--")
        if tes4[k-1]!=tes3[k-1]: ax.plot(np.linspace(k-0.2,k+0.2,10), tes4[k-1]*np.ones(10), color='b', linestyle="--")
        ax.plot(k*np.ones(10), np.linspace(tes2[k-1],tes4[k-1],10), color='b', linestyle="--")
        if tes1[k-1]==np.min(tes1): ax.scatter(k, np.min(tes1), marker="o", s=50, color='b', zorder=3)
    ax.set_xticks([1,2,3,4,8,12,16,20])
    ax.set_xticklabels(["0,1","1,1","1,2","2,2","4,4","6,6","8,8","10,10"])
    ax.set_xlabel(r'\#update: $s,t$ of $(\hat{\sigma}^{[s]},(\hat{r}_i^{[t]})_{i\in[n]})$')
    ax.set_ylabel(r'Tie criterion')
    ax.set_xlim(0,21); ax.grid(True)
    plt.savefig("./Base/sq-rank6/%f.png"%r, bbox_inches="tight", pad_inches=.02, facecolor=fig.get_facecolor(), dpi=100, edgecolor='none', metadata={'Software': None}, pil_kwargs={'optimize': True}); plt.close()#,format='webp', 
