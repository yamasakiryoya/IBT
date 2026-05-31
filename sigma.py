import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma
from scipy.integrate import quad

def sigma_bradley_terry(u):
    return 1 / (1 + np.exp(-u))
def sigma_thurstone_mosteller(u):
    return norm.cdf(u)
def sigma_stern(u, s=2.0):
    def single_stern(ui):
        scale_i = np.exp(-ui)
        scale_j = 1.0
        integrand = lambda xj: gamma.cdf(xj, a=s, scale=scale_i) * gamma.pdf(xj, a=s, scale=scale_j)
        val, _ = quad(integrand, 0, np.inf)
        return val
    return np.vectorize(single_stern)(u)
def sigma_cauchy(u):
    return np.arctan(u)/np.pi+0.5
def sigma_interp(u):
    xp = [
        -5.0, -4.0,
        -3.0, -2.0,
        -1.5, -0.9,
        -0.9, -0.5,
        -0.2,  0.2,
         0.5,  0.9,
         0.9,  1.5,
         2.0,  3.0,
         4.0,  5.0
    ]
    fp = [
        0.02, 0.02,
        0.15, 0.15,
        0.30, 0.30,
        0.40, 0.40,
        0.50, 0.50,
        0.60, 0.60,
        0.70, 0.70,
        0.85, 0.85,
        0.98, 0.98
    ]
    return np.interp(u, xp, fp)
def main():
    u_vals = np.linspace(-6, 6, 10000)
    y_bt = sigma_bradley_terry(u_vals)
    y_tm = sigma_thurstone_mosteller(u_vals)
    y_stern_s1 = sigma_stern(u_vals, s=0.1)
    y_stern_s2 = sigma_stern(u_vals, s=1)
    y_stern_s3 = sigma_stern(u_vals, s=10)
    y_cau = sigma_cauchy(u_vals)
    y_interp = sigma_interp(u_vals)
    fig = plt.figure(figsize=(10, 4))
    plt.plot(u_vals, y_bt, label='Bradley-Terry '+r'$\sigma_{\rm Logistic}$', color='r', linewidth=2)
    plt.plot(u_vals, y_tm, label='Thurstone-Mosteller', color='g', linewidth=2)
    plt.plot(u_vals, y_stern_s1, label='Stern (s=0.1)', color='b', linewidth=2)
    plt.plot(u_vals, y_stern_s2, label='Stern (s=1)', color='c', linestyle='--', linewidth=2)
    plt.plot(u_vals, y_stern_s3, label='Stern (s=10)', color='m', linewidth=2)
    plt.plot(u_vals, y_cau, label=r'$\sigma_{\rm Cauchy}$', color='gold', linewidth=2)
    plt.plot(u_vals, y_interp, label='Polyline', color='k', linewidth=2)
    plt.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle=':', linewidth=0.8)
    plt.title(r'Inverse Link Functions $\sigma$', fontsize=14)
    plt.xlabel(r'$u$', fontsize=14)
    plt.ylabel(r'$\sigma(u)$', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)
    plt.savefig("sigma.png", bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()

if __name__ == '__main__':
    main()