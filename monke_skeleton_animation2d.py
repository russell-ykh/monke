import os.path as path
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

import cv2

cd = Path(__file__).parent

data = np.genfromtxt(path.join(cd, "raw", "2d", "boba_apr11_camera1.csv"), skip_header=3, delimiter=",")[:, 1:]
tremours = np.genfromtxt(path.join(cd, "acceleration", "boba_apr11_labels.csv"), skip_header=1, delimiter=",")[:, 2]

frames = data.shape[0]
body_parts = data.shape[1] // 3

positions = data.reshape((frames, body_parts, 3))
x, y = positions[:, :, 0], positions[:, :, 1]

video_file = path.join(cd, "raw", "2d", "boba_apr11_camera1.mp4")
cap = cv2.VideoCapture(video_file)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_file = path.join(cd, "skeleton", "boba_apr11_skeleton.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

fig = plt.figure()
ax = fig.add_subplot()

body_labels = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
body_colors = []

skeleton = ax.scatter(x[0], y[0], s=1)#, labels=body_labels, c=body_colors)

def anim(frame):
    ret, frame_img = cap.read()
    if not ret:
        return
    
    ax.clear()
    # ax.imshow(frame_img)
    ax.set_title(f"{frame} / {frames}")
    ax.set_xlim(750, 1750)
    ax.set_ylim(800, 0)

    ax.scatter(x[frame], y[frame])
    # tremouring = tremours[frame] == 1 if (frame < len(tremours)) else 0
    # ax.scatter(x[frame], y[frame], c="red" if tremouring else "blue", s=1)

    fig.canvas.draw()

    # Convert the plot to an array
    # img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # # img = img.reshape((height, width, 3))  # Correct shape

    # # Convert RGB to BGR (required by OpenCV)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # out.write(img)
    
fa = FuncAnimation(fig, anim, interval=30)

plt.show()
#fa.save(output_file, fps=fps, extra_args=['-vcodec', 'libx264'])

cap.release()
out.release()