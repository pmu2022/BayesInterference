import numba
import numpy as np
from matplotlib import pyplot as plt
from numpy import trapz
from scipy.integrate import simpson as integrate
from scipy.integrate import simps

data = [10, 20, 15, 100, 110, 50]


@numba.vectorize('f8(f8, f8, f8)', nopython=True)
def likelihood(data, alpha, beta):
    return 1 / np.pi * beta / (beta ** 2 + (data - alpha) ** 2)


n = 400
min = 1e-10
max = 50

alpha = np.linspace(min, max, n)
beta = np.linspace(min, max, n)

X, Y = np.meshgrid(alpha, beta)

posterior = np.zeros((n, n))

for i, a in enumerate(alpha):
    for j, b in enumerate(beta):
        posterior[i, j] = np.prod(likelihood(data, a, b)) / (50 * 50)

norm = trapz([trapz(zz_x, alpha) for zz_x in posterior], beta)

fig, axes = plt.subplots(nrows=2)

con = axes[0].contourf(X, Y, posterior)
fig.colorbar(con)

posterior /= norm

con = axes[1].contourf(X, Y, posterior)
fig.colorbar(con)

for ax in axes:
    ax.set_aspect('equal')
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")

plt.show()
