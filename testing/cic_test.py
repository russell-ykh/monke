import numpy as np
import os.path as path
from pathlib import Path

cd = Path(__file__).parent

data = np.random.randint(-10, 11, (30, 3))

frames = data.shape[0]
features = data.shape[1]
seconds = frames // 10

threshold = 3
num_windows = frames - 10 + 1
changes_in_changes = []

for i in range(num_windows):
    window = data[i:i+10-1]
    window_next = data[i+1:i+10]
    sign_changes = np.sign(window_next) != np.sign(window)
    print(sign_changes)

    masked = window * sign_changes
    print(masked)

    counts = []
    for feature in range(features):
        differences = np.diff(masked[:, feature][masked[:, feature] != 0])
        count = np.count_nonzero(np.abs(differences) > threshold, axis=0)
        counts.append(count)

    changes_in_changes.append(counts)

changes_in_changes = np.array(changes_in_changes)
print(changes_in_changes)