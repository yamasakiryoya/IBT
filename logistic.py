import numpy as np
from scipy.special import logit, expit, log_expit, erf
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 25
plt.rcParams['text.usetex'] = True
cmap = plt.get_cmap('jet')


# plot
fig = plt.figure(figsize=(10,6)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111)
fx = np.linspace(-10,10,10000)
ax.plot(fx, expit(fx), lw=2, color="k")
ax.set_xlabel(r'$u$'); ax.set_ylabel(r'$\sigma_{\rm Log.}(u)=1/(1+e^{-u})$'); ax.grid(True); plt.tight_layout()
plt.savefig("./logistic.png", bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
