import os.path as path
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

cd = Path(__file__).parent

data = np.genfromtxt(path.join(cd, "raw", "boba_apr11.csv"), skip_header=3, delimiter=",")[:, 1:]
tremours = np.genfromtxt(path.join(cd, "acceleration", "boba_apr11_labels.csv"), skip_header=1, delimiter=",")[:, 2]

frames = data.shape[0]
body_parts = data.shape[1] // 3

positions = data.reshape((frames, body_parts, 3))
x, y, z = positions[:, :, 0], positions[:, :, 1], positions[:, :, 2]

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
                     
body_labels = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
body_colors = []

skeleton = ax.scatter(x[0], y[0], z[0])#, labels=body_labels, c=body_colors)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

def anim(frame):
    ax.clear()
    ax.set_title(f"{frame} / {frames}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.set_xlim(-50, 50)
    # ax.set_ylim(-50, 50)
    # ax.set_zlim(-50, 50)
    tremouring = tremours[frame] == 1 if (frame < len(tremours)) else 0
    ax.scatter(x[frame], y[frame], z[frame], c="red" if tremouring else "blue")
    
fa = FuncAnimation(fig, anim, interval=30)

plt.show()