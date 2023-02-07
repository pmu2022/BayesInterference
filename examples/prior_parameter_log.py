import numba
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simpson as integrate

"""
Doesn't work with likelhood one has to work with log likleihood
"""

dataset = [
    [2.668, 0.729, 1.697, 1.738, 2.559, 4.964],
    [3.002, 3.035, 1.998, 0.393, 7.12, 4.57],
    [133.43, 3.21, 12.3, 3.987, -1.238, -56.433],
]


@numba.vectorize('f8(f8, f8, f8)', nopython=True)
def log_likelihood(data, mu, sigma):
    return np.log(1 / (np.sqrt(2.0 * np.pi) * sigma)) + (-1 / (sigma ** 2 * 2.0) * (data - mu) ** 2)


figs, axes = plt.subplots(nrows=3)

mins = [0, 1, 10]
maxs = [5, 6, 20]

for idx, data in enumerate(dataset):

    mu = np.linspace(mins[idx], maxs[idx], 20000)

    sigma = 1.0

    posterior = np.zeros_like(mu)

    for i in range(len(posterior)):
        posterior[i] = np.sum(log_likelihood(data, mu[i], sigma))

    posterior -= np.max(posterior)

    posterior = np.exp(posterior)

    norm = integrate(posterior, mu)

    posterior /= norm

    axes[idx].plot(mu, posterior, ls='-')

    arg = np.argmax(posterior)

    axes[idx].axvline(mu[arg], color='red', ls='-', lw=4)

    data_mu = np.mean(data)

    axes[idx].axvline(data_mu, color='blue', ls='--', lw=1)

plt.show()
