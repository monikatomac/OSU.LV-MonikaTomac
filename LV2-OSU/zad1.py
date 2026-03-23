import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 2, 1, 3, 2])

plt.plot(x, y, color='red', linewidth=3, linestyle='--')

plt.scatter(x, y, color='blue') 
plt.title("Primjer slike")
plt.xlabel("X os")
plt.ylabel("Y os")

plt.show()