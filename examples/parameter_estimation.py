import numpy as np

from scipy.stats import norm

data_a = np.array([2.67, 0.73, 1.7, 1.79, 2.56, 4.59])

sigma_sample = np.std(data_a)
mean_sample = np.std(data_a)


def prior(mu, mu_0, S):
    return np.exp(-(mu - mu_0) ** 2 / (2 * S ** 2)) / (S * np.sqrt(2 * np.pi))

@np.vectorize
def gaussian(d, mu, sigma):
    return np.exp(-(d - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2.0 * np.pi))

mu = 1.1223
sigma = 1.0

likelihood = gaussian(data_a, mu, sigma)

print(likelihood)