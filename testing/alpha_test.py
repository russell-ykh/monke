import numpy as np

a1 = np.arange(0, 20).reshape((5, 4))
a2 = np.array([0,0 ,0, 1, 1])

print(a1)

print(a1[a2 != 0])