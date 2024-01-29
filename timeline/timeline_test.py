# Testing if supervised UMAP reduction can visualise unlabelled data well!

import os.path as path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

cd = "Y2S2/Monke/"

data = np.genfromtxt(path.join(cd, "ang_change/boba_apr11_ang_changes.csv"), skip_header=1, delimiter=",")[:900, :]
names = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
frames = data[:900, 0]

features = data.shape[1]
for f in range(1, features):
    feature = data[:, f]
    plt.plot(frames, feature, label=names[f-1])
    plt.vlines([450, 690], ymin=0, ymax=3, colors="red")
    plt.legend()
    plt.savefig(path.join(cd, f"timeline/boba_apr11_{names[f-1]}_ang_changes.png"), dpi=300)
    plt.show()