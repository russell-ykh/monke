# Testing if supervised UMAP reduction can visualise unlabelled data well!

import os.path as path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

cd = Path(__file__).parent

data = np.genfromtxt(path.join(cd, "acceleration/boba_apr11_accel.csv"), skip_header=1, delimiter=",")[300:700, :]
names_part = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
xyz = ["_x", "_y", "_z"]

names = []

for name in names_part:
    for axis in xyz:
        names.append(f"{name}{axis}")

frames = data[:, 0]
features = data.shape[1] - 1

# tremours = np.genfromtxt(path.join(cd, "raw/boba_apr11_tremours.csv"), skip_header=1, delimiter=",")
# tremour_rows = tremours[tremours[:, 2] == 1]
# tremour_times = tremour_rows[:, :2]

for f in range(1, features):
    feature = data[:, f]
    feature_ymin = np.min(feature)
    feature_ymax = np.max(feature)
    plt.plot(frames, feature, label=names[f-1])
    # plt.vlines(tremour_times, ymin=0, ymax=feature_ymax, colors="red")
    plt.vlines([450, 690], ymin=feature_ymin, ymax=feature_ymax, colors="red")
    plt.legend()
    plt.xlabel("Frame")
    plt.ylabel("Acceleration")
    plt.savefig(path.join(cd, f"timeline/acceleration/boba_apr11_{names[f-1]}_acceleration.png"), dpi=300)
    plt.show()