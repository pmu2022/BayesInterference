import matplotlib.pyplot as plt
import numba
import numpy as np
from numba import prange

fig, axes = plt.subplots(nrows=3)

x = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
data = np.array([7.5, 12.0, 15.0, 19.0, 22.0])


@numba.vectorize('f8(f8, f8, f8)', nopython=True)
def model(x, a, b):
    return np.sqrt(a * b * x) + np.log(a * x) + 0.5 * np.sqrt(b) * x


@numba.vectorize('f8(f8, f8, f8)', nopython=True)
def likelihood(data, mu, sigma):
    return 1.0 / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-1.0 / (sigma ** 2 * 2.0) * (data - mu) ** 2)


n = 200
a_vals = np.linspace(0.5, 30, n)
b_vals = np.linspace(0.2, 20, n)
X, Y = np.meshgrid(a_vals, b_vals)

posterior = np.zeros((n, n))


for i in prange(n):
    for j in prange(n):
        m_vals = model(x, X[i, j], Y[i, j])
        posterior[i, j] = np.prod(likelihood(data, m_vals, 1.0))

norm = np.trapz([np.trapz(zz_x, a_vals) for zz_x in posterior], b_vals)
posterior /= norm

axes[0].contourf(a_vals, b_vals, posterior)
axes[0].set_aspect('equal')

pa = np.trapz(posterior, b_vals, axis=0)
pb = np.trapz(posterior, a_vals, axis=1)

max_points = np.argmax(posterior)
max_index = np.unravel_index(max_points, posterior.shape)

print(X[max_index], Y[max_index])

x_lin = np.linspace(0.1, 10, 100)
axes[2].plot(x_lin, model(x_lin, X[max_index], Y[max_index]))
axes[2].plot(x, data, 'rx')

axes[0].scatter(X[max_index], Y[max_index], color='red')

axes[1].plot(a_vals, pa, color='red')
axes[1].plot(b_vals, pb, color='blue')

a_mean = np.trapz(pa * a_vals, a_vals)
b_mean = np.trapz(pb * b_vals, b_vals)

a_var = np.trapz(pa * a_vals ** 2, a_vals) - a_mean ** 2
b_var = np.trapz(pb * b_vals ** 2, b_vals) - b_mean ** 2

print(a_var)
print(b_var)

axes[1].axvline(a_mean, color='red', ls='--')
axes[1].axvline(b_mean, color='blue', ls='--')

plt.show()
