import numba
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import laplace
from scipy.stats import norm
from scipy.stats import cauchy

numbers = 1000000

fig, axes = plt.subplots(nrows=3)


# Cauchy

@numba.vectorize('f8(f8)', nopython=True)
def convert_to_cauchy(u1):
    _x = np.tan(np.pi * (u1 - 0.5))
    return _x


# Gaussian

@numba.vectorize('f8(f8, f8)', nopython=True)
def convert_to_gaussian(u1, u2):
    return np.cos(2 * np.pi * u1) * np.sqrt(2.0 * np.log(1 / u2))


# Laplacian

@numba.vectorize('f8(f8, f8)', nopython=True)
def convert_to_laplacian(u1, u2):
    _x = - np.log(u1)

    if u2 < 0.5:
        _x = -_x

    return _x




rng = np.random.default_rng()

u1 = rng.uniform(0.0, 1.0, numbers)
u2 = rng.uniform(0.0, 1.0, numbers)

x = np.linspace(norm.ppf(0.001),
                norm.ppf(0.999), 100)

for ax in axes:
    ax.set_xlim(np.min(x), np.max(x))

z = convert_to_gaussian(u1, u2)
axes[0].hist(z, bins=200, color="red", density=True)
axes[0].plot(x, norm.pdf(x), 'b--')

z = convert_to_laplacian(u1, u2)
axes[1].hist(z, bins=1000, color="red", density=True)
axes[1].plot(x, laplace.pdf(x), 'b--')

z = convert_to_cauchy(u1)
axes[2].hist(z, bins=5000, range=(-100, 100), color="red", density=True)
axes[2].plot(x, cauchy.pdf(x), 'b--')

plt.show()
