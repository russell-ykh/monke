import numpy as np

alpha = np.array(["a", "b", "c", "d", "e", "f"])

print(np.where(np.isin(alpha, ["c", "e", "a"])))