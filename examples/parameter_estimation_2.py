import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import numba
from scipy.integrate import simps

fig, axes = plt.subplots(nrows=2)

a_vals = np.linspace(-10, 10, 10000)


@numba.vectorize('f8(f8, f8)', nopython=True)
def model(x, a):
    return a * np.cos(x)


x = np.array([0, 1, 2, 3, 4, 5, 6])
data = np.array([3.5, 0.6, 3.61, -0.7, 0.45, 2.9, 2.57])


@numba.vectorize('f8(f8, f8, f8)', nopython=True)
def likelihood(d, m, b):
    if (d - m) > 0:
        return 1 / b * np.exp(-(d - m) / b)
    else:
        return 0


posterior = np.empty_like(a_vals)

for i, a in enumerate(a_vals):
    m_vals = model(x, a)

    posterior[i] = np.prod(likelihood(data, m_vals, 1.1))

norm = simps(posterior, a_vals)

posterior /= norm

axes[0].plot(a_vals, posterior)
axes[0].set_xlim(0.0, 2.0)

x_val = np.linspace(-1, 7, 100)

X, Y = np.meshgrid(x_val, a_vals)

diff = np.abs(np.gradient(posterior, a_vals))

args = np.argsort(-diff)

axt = axes[0].twinx()

axt.plot(a_vals, diff, color='red', ls='--')
axt.axvline(a_vals[args[0]])
axt.axvline(a_vals[args[2]])

minval = a_vals[args[0]]
maxval = a_vals[args[2]]

axes[1].fill_between(x_val, model(x_val, minval), model(x_val, maxval))

axes[1].plot(x, data, 'k*', label='data')

axes[1].legend()

plt.show()
