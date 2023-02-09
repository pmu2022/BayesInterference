import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln

"""
It is easier too prove that the game is not fair then the other way around 
"""

random_generator = np.random.default_rng()

n = 2000

nha = np.cumsum(np.random.choice([0, 1], size=n, p=[0.5, 0.5]))
nhb = np.cumsum(np.random.choice([0, 1], size=n, p=[0.45, 0.55]))

log_odds = np.zeros((2, n))
i_s = np.arange(0, n)

log_odds[0, :] = i_s * np.log(0.5) + gammaln(i_s + 2) - gammaln(nha + 1) - gammaln(i_s - nha + 1)
log_odds[1, :] = i_s * np.log(0.5) + gammaln(i_s + 2) - gammaln(nhb + 1) - gammaln(i_s - nhb + 1)

plt.plot(log_odds[0, :], 'r+', label="0.5")
plt.plot(log_odds[1, :], 'b+', label="0.6")

plt.legend()

plt.show()
