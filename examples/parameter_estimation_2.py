import numpy as np

x = np.array([0, 1, 2, 3, 4, 5, 6])
data = np.array([3.5, 0.6, 3.61, -0.7, 0.45, 2.9, 2.57])

fig, axes = plt.subplots()

axes.plot(x, data, 'r*')

plt.show()
