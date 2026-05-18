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
def sigma_step(u):
    condlist = [
        u < -5.0,
        (-5.0 <= u) & (u < -4.0),
        (-4.0 <= u) & (u < -3.0),
        (-3.0 <= u) & (u < -2.0),
        (-2.0 <= u) & (u < -1.5),
        (-1.5 <= u) & (u < -1.0),
        (-1.0 <= u) & (u < -0.5),
        (-0.5 <= u) & (u <= 0.5),
        ( 0.5 <  u) & (u <= 1.0),
        ( 1.0 <  u) & (u <= 1.5),
        ( 1.5 <  u) & (u <= 2.0),
        ( 2.0 <  u) & (u <= 3.0),
        ( 3.0 <  u) & (u <= 4.0),
        ( 4.0 <  u) & (u <= 5.0),
        5.0 < u
    ]
    funclist = [0.00,0.01,0.03,0.08,0.15,0.22,0.32,0.50,0.68,0.78,0.85,0.92,0.97,0.99,1.00]
    return np.piecewise(u, condlist, funclist)
def main():
    u_vals = np.linspace(-6, 6, 10000)
    y_bt = sigma_bradley_terry(u_vals)
    y_tm = sigma_thurstone_mosteller(u_vals)
    y_stern_s1 = sigma_stern(u_vals, s=1)
    y_stern_s2 = sigma_stern(u_vals, s=2)
    y_stern_s4 = sigma_stern(u_vals, s=4)
    y_step = sigma_step(u_vals)
    fig = plt.figure(figsize=(10, 4))
    plt.plot(u_vals, y_bt, label='Bradley-Terry', color='r', linewidth=2)
    plt.plot(u_vals, y_tm, label='Thurstone-Mosteller', color='g', linewidth=2)
    plt.plot(u_vals, y_stern_s1, label='Stern (s=1)', color='b', linestyle=':', linewidth=2)
    plt.plot(u_vals, y_stern_s2, label='Stern (s=2)', color='c', linewidth=2)
    plt.plot(u_vals, y_stern_s4, label='Stern (s=4)', color='m', linewidth=2)
    plt.plot(u_vals, y_step, label='Step', color='gold', linewidth=2)
    plt.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle=':', linewidth=0.8)
    plt.title(r'Inverse Link Functions $\sigma$', fontsize=16)
    plt.xlabel(r'$u$', fontsize=16)
    plt.ylabel(r'$\sigma(u)$', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16)
    plt.savefig("sigma.png", bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()

if __name__ == '__main__':
    main()