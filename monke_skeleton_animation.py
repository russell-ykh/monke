import os.path as path
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

import cv2

from monke_features import get_indices, generate_labelled_frames

cd = Path(__file__).parent

# data_2d = np.genfromtxt(path.join(cd, "raw", "2d", "boba_apr11_camera1.csv"), skip_header=3, delimiter=",")[:, 1:]
# tremours = generate_labelled_frames(data_2d, pd.read_csv(path.join(cd, "raw", "boba_apr11_tremours.csv")))
# video_file = path.join(cd, "raw", "2d", "boba_apr11_camera1.mp4")
# output_file = path.join(cd, "skeleton", "boba_apr11_skeleton.mp4")

data_2d = np.genfromtxt(path.join(cd, "raw", "2d", "bandung_mar27_3_camera1.csv"), skip_header=3, delimiter=",")[:, 1:]
tremours = generate_labelled_frames(data_2d, pd.read_csv(path.join(cd, "raw", "bandung_mar27_3_tremours.csv")))
video_file = path.join(cd, "raw", "2d", "bandung_mar27_3_camera1.mp4")

class Frame:
    def __init__(self):
        self.value = 0

    def set(self, value):
        self.value = value

    def get(self):
        return self.value

def animated2d(data, tremours, video_file, output_file=None, skeleton=None):
    frames = data.shape[0]
    body_parts = data.shape[1] // 3

    positions = data.reshape((frames, body_parts, 3))
    x, y = positions[:, :, 0], positions[:, :, 1]

    cap = cv2.VideoCapture(video_file)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if output_file is not None:
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    fig = plt.figure()
    ax = fig.add_subplot()

    #DELETE THIS LATER
    # target = np.array([5, 7, 9])
    #JUST FOR ERIC TESTING

    current_frame = Frame()

    def anim(frame):
        ret, frame_img = cap.read()
        if not ret:
            return
        ax.clear()
        ax.imshow(frame_img)

        tremoring = tremours[frame] == 1 if (frame < len(tremours)) else 0

        ax.set_title(f"{frame} / {frames}", c="tab:red" if tremoring else "black")
        ax.set_axis_off()
        # ax.set_xlim(1000, 1750)
        # ax.set_ylim(800, 100)
        ax.set_xlim(1000, 1700)
        ax.set_ylim(800, 100)
        
        ax.scatter(x[frame], y[frame], c="tab:red" if tremoring else "tab:blue", s=2, zorder=3)
        
        tremouring_colours = ["tab:orange", "tab:olive"]
        section_colours = ["tab:pink", "tab:green"]
        i = 0
        for section in skeleton:
            ax.plot(x[frame, section], y[frame, section], c=tremouring_colours[i] if tremoring else section_colours[i], linewidth=2, zorder=1)
            i+=1

        current_frame.set(frame+1)
        fig.canvas.draw()

    def update_fa(val):
        anim(current_frame.value)

    if output_file is not None:
        # Convert the plot to an array
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

        # # Convert RGB to BGR (required by OpenCV)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)
    
    def toggle_pause(event):
        if pause_button.label.get_text() == "Play":
            fa.resume()
            pause_button.label.set_text("Pause")
        else:
            fa.pause()
            pause_button.label.set_text("Play")

    # Function to handle slider changes
    def on_slider_change(val):
        fa.pause()
        frame = int(val)
        current_frame.set(frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame-1)
        pause_button.label.set_text("Play")
        anim(frame)

    pause_ax = plt.axes([0.7, 0.01, 0.1, 0.05])
    pause_button = Button(pause_ax, "Pause")
    pause_button.on_clicked(toggle_pause)

    slider_ax = plt.axes([0.2, 0.01, 0.4, 0.03])
    slider = Slider(slider_ax, 'Frame', 0, frames-1, valinit=1, valstep=1)
    slider.on_changed(on_slider_change)
    
    fa = FuncAnimation(fig, update_fa, interval=10)

    plt.show()
    #fa.save(output_file, fps=fps, extra_args=['-vcodec', 'libx264'])

    cap.release()
    out.release()

def animated3d(data, tremours):
    frames = data.shape[0]
    body_parts = data.shape[1] // 3

    positions = data.reshape((frames, body_parts, 3))
    x, y, z = positions[:, :, 0], positions[:, :, 1], positions[:, :, 2]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(x[0], y[0], z[0])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    def anim(frame):
        ax.clear()
        ax.set_title(f"{frame} / {frames}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_zlim(-30, 40)
        tremouring = tremours[frame] == 1 if (frame < len(tremours)) else 0
        ax.scatter(x[frame], y[frame], z[frame], c="red" if tremouring else "blue")
        
    fa = FuncAnimation(fig, anim, interval=1000/30)

    plt.show()

def screenshot_annotated(screenshot, x, y, labels, skeleton_sections):
    plt.imshow(screenshot)

    # for i in range(len(labels)):
    #     label = labels[i]
    #     plt.annotate(label, (x[i], y[i]), c="tab:blue", fontsize=6)

    for indices in skeleton_sections:
        plt.plot(x[indices], y[indices], linewidth=5, c="tab:blue", zorder=1)
    
    plt.scatter(x, y, s=10, c="tab:orange", zorder=3)
    plt.axis('off')

    plt.show()

# frame = 1173

# cap = cv2.VideoCapture(path.join(cd, "raw", "2d", "boba_apr11_camera1.mp4"))
# cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
# ret, screenshot = cap.read()
# screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)

# data_2d = np.genfromtxt(path.join(cd, "raw", "2d", "boba_apr11_camera1.csv"), skip_header=3, delimiter=",")[:, 1:]
# coordinates = data_2d[frame]
# coordinates = np.reshape(coordinates, (coordinates.shape[0]//3, 3))
# x = coordinates[:, 0]
# y = coordinates[:, 1]

labels = np.genfromtxt(path.join(cd, "raw", "2d", "bandung_mar27_3_camera1.csv"), delimiter=",", dtype=str)[1, 1:][::3]
bottom_indices = get_indices(labels, ["right_ankle", "right_knee", "right_hip", "left_hip", "left_knee", "left_ankle"])
top_indices = get_indices(labels, ["right_wrist", "right_elbow", "right_shoulder", "left_shoulder", "left_elbow", "left_wrist"])
# face_indices = get_indices(labels, ["right_ear", "right_eye", "nose", "left_eye", "left_ear"])
skeleton_sections = [bottom_indices, top_indices]#, face_indices]

# screenshot_annotated(screenshot_rgb, x, y, labels, skeleton_sections)
# cap.release()

def animated3d_upgraded(data, tremours, skeleton):
    frames = data.shape[0]
    body_parts = data.shape[1] // 3

    positions = data.reshape((frames, body_parts, 3))
    x, y, z = positions[:, :, 0], positions[:, :, 1], positions[:, :, 2]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(x[0], y[0], z[0], c="tab:blue")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    def anim(frame):
        ax.clear()
        ax.set_title(f"{frame} / {frames}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        # ax.set_xlim(0, 10)
        # ax.set_ylim(0, 10)
        # ax.set_zlim(-30, 40)

        tremouring = tremours[frame] == 1 if (frame < len(tremours)) else 0
        ax.scatter(x[frame], y[frame], z[frame], c="tab:red" if tremouring else "tab:blue")

        section_colours = ["tab:pink", "tab:green", "tab:orange"]
        i = 0
        for section in skeleton:
            ax.plot(x[frame, section], y[frame, section], z[frame, section], c="tab:red" if tremouring else section_colours[i])
            i+=1

    def toggle_pause(event):
        if pause_button.label.get_text() == "Play":
            fa.resume()
            pause_button.label.set_text("Pause")
        else:
            fa.pause()
            pause_button.label.set_text("Play")

    # Function to handle slider changes
    def on_slider_change(val):
        fa.pause()
        pause_button.label.set_text("Play")
        anim(val)

    pause_ax = plt.axes([0.7, 0.01, 0.1, 0.05])
    pause_button = Button(pause_ax, "Pause")
    pause_button.on_clicked(toggle_pause)

    slider_ax = plt.axes([0.2, 0.01, 0.4, 0.03])
    slider = Slider(slider_ax, 'Frame', 0, frames-1, valinit=1, valstep=1)
    slider.on_changed(on_slider_change)

    fa = FuncAnimation(fig, anim, interval=1000/30)

    plt.show()

def anim3d_test():
    data_3d = np.genfromtxt(path.join(cd, "raw", "boba_apr11.csv"), skip_header=3, delimiter=",")[:, 1:]
    labels = np.genfromtxt(path.join(cd, "raw", "boba_apr11.csv"), delimiter=",", dtype=str)[1, 1:][::3]
    bottom_indices = get_indices(labels, ["right_ankle", "right_knee", "right_hip", "left_hip", "left_knee", "left_ankle"])
    top_indices = get_indices(labels, ["right_wrist", "right_elbow", "right_shoulder", "left_shoulder", "left_elbow", "left_wrist"])
    #face_indices = get_indices(labels, ["right_ear", "right_eye", "nose", "left_eye", "left_ear"])
    skeleton = [bottom_indices, top_indices]#, face_indices]

    animated3d_upgraded(data_3d, tremours, skeleton)

animated2d(data_2d, tremours, video_file, skeleton=skeleton_sections)