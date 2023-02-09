import matplotlib.pyplot as plt
import numba
import numpy as np


@numba.vectorize('f8(f8, f8, f8)', nopython=True)
def likelihood(data, mu, sigma):
    return 1.0 / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-1.0 / (sigma ** 2 * 2.0) * (data - mu) ** 2)


@numba.vectorize('f8(f8, f8, f8)', nopython=True)
def log_likelihood(data, mu, sigma):
    return np.log(1 / (np.sqrt(2.0 * np.pi) * sigma)) + (-1 / (sigma ** 2 * 2.0) * (data - mu) ** 2)


@numba.vectorize('f8(f8)', nopython=True)
def model_1(c):
    return c


@numba.vectorize('f8(f8, f8, f8)', nopython=True)
def model_2(x, b, c):
    return b * x + c


@numba.vectorize('f8(f8, f8, f8, f8)', nopython=True)
def model_3(x, a, b, c):
    return a * x ** 2 + b * x + c


x_data = np.array([0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1, 9.0])
d_data = np.array([0.8, 0.9, 1.4, 1.25, 1.1, 0.5, 0.77, 0.69, 1.2, 1.25])
sigma_data = np.array([0.12, 0.25, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.5])

n = 400

vals = np.linspace(-5, 5, n)

posterior_1 = np.zeros(n)
posterior_2 = np.zeros((n, n))
posterior_3 = np.zeros((n, n, n))

fig, axes = plt.subplots(nrows=2)

for i in range(n):
    m_vals_1 = model_1(vals[i])
    posterior_1[i] = np.sum(log_likelihood(d_data, m_vals_1, sigma_data))

posterior_1 = np.exp(posterior_1)
evidence_1 = np.trapz(posterior_1, vals) * 1 / 10
posterior_1 /= evidence_1

axes[0].plot(vals, posterior_1)

X, Y = np.meshgrid(vals, vals)


@numba.jit(nopython=True)
def posterior_2d(n, X, Y, x_d, d_d, sigma_d):
    posterior = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            x = X[i, j]
            y = Y[i, j]

            m_vals_2 = model_2(x_d, x, y)
            posterior[i, j] = np.sum(log_likelihood(d_d, m_vals_2, sigma_d))

    return posterior


posterior_2 = posterior_2d(n, X, Y, x_data, d_data, sigma_data)

posterior_2 = np.exp(posterior_2)
evidence_2 = np.trapz([np.trapz(zz_x, vals) for zz_x in posterior_2], vals) * (1 / 10) ** 2
posterior_2 /= evidence_2

axes[1].contourf(vals, vals, posterior_2)
axes[1].set_aspect('equal')

XX, YY, ZZ = np.meshgrid(vals, vals, vals, indexing='ij')


@numba.jit(nopython=True)
def posterior_3d(n, XX, YY, ZZ, x_d, d_d, sigma_d):
    posterior = np.zeros((n, n, n))

    for i in range(n):
        for j in range(n):
            for k in range(n):
                x = XX[i, j, k]
                y = YY[i, j, k]
                z = ZZ[i, j, k]

                m_vals_3 = model_3(x_d, x, y, z)
                posterior[i, j, k] = np.sum(log_likelihood(d_d, m_vals_3, sigma_d))

    return posterior


posterior_3 = posterior_3d(n, XX, YY, ZZ, x_data, d_data, sigma_data)

posterior_3 = np.exp(posterior_3)
res = [np.trapz(zz_x, vals) for zz_x in posterior_3]
evidence_3 = np.trapz([np.trapz(zz_x, vals) for zz_x in res], vals) * (1 / 10) ** 3
posterior_3 /= evidence_3

c1 = vals[np.argmax(posterior_1)]

max_points = np.argmax(posterior_2)
max_index = np.unravel_index(max_points, posterior_2.shape)

b2 = X[max_index]
c2 = Y[max_index]

max_points = np.argmax(posterior_3)
max_index = np.unravel_index(max_points, posterior_3.shape)

a3 = XX[max_index]
b3 = YY[max_index]
c3 = ZZ[max_index]

fig1, axes1 = plt.subplots()

axes1.errorbar(x_data, d_data, yerr=sigma_data, marker='s', ls='none')

x_lin = np.linspace(np.min(x_data), np.max(x_data), 100)

axes1.plot(x_lin, model_1(c1) * np.ones_like(x_lin), 'r')
axes1.plot(x_lin, model_2(x_lin, b2, c2), 'g')
axes1.plot(x_lin, model_3(x_lin, a3, b3, c3), 'b')

evidence = np.array([
    evidence_1,
    evidence_2,
    evidence_3])

evidence /= np.max(evidence)

print(evidence)

plt.show()
