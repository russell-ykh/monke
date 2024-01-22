import os.path as path
import numpy as np
import pandas as pd
from pathlib import Path

cd = Path(__file__).parent

def monke_accel():
    data = np.genfromtxt("Y2S2/Monke/boba_apr11.csv", skip_header=3, delimiter=",")

    accel = np.diff(data, n=2, axis=0)

    df = pd.DataFrame(accel)
    df.to_csv("Y2S2/Monke/boba_apr11_accel.csv")

def monke_vel(data, file_name):
    vel = np.diff(data, axis=0)

    df = pd.DataFrame(vel)
    df.to_csv(file_name)

# data = np.genfromtxt("Y2S2/Monke/boba_apr11.csv", skip_header=3, delimiter=",")
# monke_vel(data, "Y2S2/Monke/boba_apr11_vel.csv")
    
# Returns a new file with the number of times the sign changes in a second
def monke_dir_changes(data, file_name, fps=30):
    dir_changes_raw = np.sign(data[:-1, 1:]) != np.sign(data[1:, 1:])

    frames = dir_changes_raw.shape[0]
    features = dir_changes_raw.shape[1]

    seconds_ceil = -(frames // -fps)
    frames_to_add = seconds_ceil*fps - frames

    dir_changes_raw = np.pad(dir_changes_raw, ((0, frames_to_add), (0,0)), constant_values=False)

    frames = frames + frames_to_add

    dc_reshaped = dir_changes_raw.reshape((frames//fps, fps, features))
    dir_changes = dc_reshaped.sum(axis=1)

    df = pd.DataFrame(dir_changes)
    df.to_csv(file_name)

# data = np.genfromtxt("Y2S2/Monke/dir_change/boba_apr11_vel.csv", skip_header=1, delimiter=",")
# monke_dir_changes(data, "Y2S2/Monke/dir_change/boba_apr11_dir_changes.csv")
    
def monke_angchanges(data, file_name):
    vel = np.diff(data[:, 1:], axis=0)
    
    frames = vel.shape[0]
    features = vel.shape[1]
    body_parts = features//3

    vel_reshaped = vel.reshape((frames, body_parts, 3))

    # u is the velocities (or maybe you could think of them as vectors?) of the current frame
    # v is the velocities of the next frame
    ux, uy, uz = vel_reshaped[:-1, :, 0], vel_reshaped[:-1, :, 1], vel_reshaped[:-1, :, 2]
    vx, vy, vz = vel_reshaped[1:, :, 0], vel_reshaped[1:, :, 1], vel_reshaped[1:, :, 2]

    uv = np.add(np.multiply(ux, vx), np.multiply(uy, vy), np.multiply(uz, vz))
    mag_u = np.sqrt(np.add(ux**2, uy**2, uz**2))
    mag_v = np.sqrt(np.add(vx**2, vy**2, vz**2))

    ang_changes = np.nan_to_num(np.arccos(np.nan_to_num(np.divide(uv, np.multiply(mag_u, mag_v)))))

    df = pd.DataFrame(ang_changes)
    df.to_csv(file_name)

# data_path = path.join(cd, "raw/boba_apr11.csv")
# monke_angchanges(np.genfromtxt(data_path, skip_header=3, delimiter=","), path.join(cd, "ang_change/boba_apr11_ang_changes.csv"))

def generate_labelled_frames(pose_data, tremour_data_raw, file_name, fps=30):
    labels = []
    for index, row in tremour_data_raw.iterrows():
        start = int(row["temporal_segment_start"]) * fps
        end = int(row["temporal_segment_end"]) * fps
        label = int(row["label"])

        for i in range(start, end):
            labels.append(label)
    
    frames = pose_data["frame"]
    last_frame = int(frames.tail(1))
    
    remainder = last_frame - len(labels)

    for a in range(remainder+1):
        labels.append(0)

    tremours = pd.DataFrame({"frame":frames, "label":labels})
    tremours.to_csv(file_name)

# pose_data = pd.read_csv("Y2S2/Monke/boba_apr11_accel.csv")
# tremour_data_raw = pd.read_csv("Y2S2/Monke/boba_apr11_tremours.csv")
# generate_labelled_frames(pose_data, tremour_data_raw, "Y2S2/Monke/boba_apr11_frames_annotated.csv")

def generate_labelled_seconds(pose_data, tremour_data_raw, file_name):
    labels = []
    for index, row in tremour_data_raw.iterrows():
        start = int(row["temporal_segment_start"])
        end = int(row["temporal_segment_end"])
        label = int(row["label"])

        for i in range(start, end):
            labels.append(label)
    
    seconds = pose_data.shape[0] - 1
    remainder = seconds - len(labels)

    for a in range(remainder+1):
        labels.append(0)

    tremours = pd.DataFrame({"label":labels})
    tremours.to_csv(file_name)

# pose_data = pd.read_csv(path.join(Path(__file__).parent, "dir_change/boba_apr11_dir_changes.csv"))
# tremour_data_raw = pd.read_csv(path.join(Path(__file__).parent, "raw/boba_apr11_tremours.csv"))
# file_name = path.join(Path(__file__).parent, "dir_change/boba_apr11_labels.csv")
# generate_labelled_seconds(pose_data, tremour_data_raw, file_name)