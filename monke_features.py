import os.path as path
import numpy as np
import pandas as pd
from pathlib import Path

cd = Path(__file__).parent
data = np.genfromtxt(path.join(cd, "raw/boba_apr11.csv"), skip_header=3, delimiter=",")

# General-purpose function for processing DLC 3D coordinate data CSV files
# data: Input data, usually as a Numpy array
# process: Function for processing input data, must return a new Numpy array
# save_as (optional): File name to save new processed array under
def monke_process(data, process, save_as=None):
    processed = process(data)

    if save_as is not None:
        pd.DataFrame(processed).to_csv(save_as)
    
    return processed

def vel(data):
    return np.diff(data, axis=0)

# data = np.genfromtxt("Y2S2/Monke/boba_apr11.csv", skip_header=3, delimiter=",")
# monke_vel(data, "Y2S2/Monke/boba_apr11_vel.csv")

def accel(data):
    return np.diff(data, n=2, axis=0)

# file_name = path.join(cd, "accel/boba_apr11_accel.csv")
# monke_accel()

def jerk(data):
    return np.diff(data, n=3, axis=0)

def diff(n):
    def diff_internal(data):
        return np.diff(data, n=n, axis=0)
    return diff_internal

def ang_of(a1, b1):
    if (type(a1) == float) or (type(b1) == int) or (type(a1) == np.float64):
        if (a1 >= 0 and b1 >= 0) :
            return np.arctan(b1/a1)
        elif a1 < 0:
            return np.pi + np.arctan(b1/a1)
        elif (a1 >= 0 and b1 < 0) :
            return 2 * np.pi + np.arctan(b1/a1)
    else:
        temp = []
        for i in range(len(a1)):
            temp.append(ang_of(a1[i], b1[i]))
        return temp

def unsign_to_sign_ang_change(c1):
    if (type(c1) == float) or (type(c1) == int) or (type(c1) == np.float64):
        d1 = c1
        if c1 > np.pi:
            d1 = c1 - 2 * np.pi
        elif c1 <= -np.pi:
            d1 = c1 +  2 * np.pi
        return d1
    else:
        temp = []
        for i in range(len(c1)):
            temp.append(unsign_to_sign_ang_change(c1[i]))
        return temp
    
# The number of times the sign changes in a second
def dir_changes(fps=30):
    def dir_changes(data):
        vel = vel(data)
        dir_changes_raw = np.sign(vel[:-1, 1:]) != np.sign(vel[1:, 1:])

        frames = dir_changes_raw.shape[0]
        features = dir_changes_raw.shape[1]

        seconds_ceil = -(frames // -fps)
        frames_to_add = seconds_ceil*fps - frames

        dir_changes_raw = np.pad(dir_changes_raw, ((0, frames_to_add), (0,0)), constant_values=False)

        frames = frames + frames_to_add

        dc_reshaped = dir_changes_raw.reshape((frames//fps, fps, features))
        dir_changes_final = dc_reshaped.sum(axis=1)

        return dir_changes_final
    return dir_changes

# data = np.genfromtxt("Y2S2/Monke/dir_change/boba_apr11_vel.csv", skip_header=1, delimiter=",")
# monke_dir_changes(data, "Y2S2/Monke/dir_change/boba_apr11_dir_changes.csv")

# The number of times the sign in velocity changes in a second, across all three axes
# data: monke_dir_changes
def dir_changes_summed(data):
    seconds = data.shape[0]
    feature_axes = data.shape[1]
    features = feature_axes // 3
    reshaped = data.reshape((seconds, features, 3))
    dir_changes_summed = np.sum(reshaped, axis=2)
    return dir_changes_summed

# data = np.genfromtxt(path.join(cd, "dir_change/boba_apr11_dir_changes.csv"), skip_header=1, delimiter=",")[:, 1:]
# monke_process(data, monke_dir_changes_summed, save_as=path.join(cd, "dir_change_summed/boba_apr11.csv"))

def ang_changes(data):
    vel = vel(data)
    
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

    return ang_changes

def phi_changes(data):
    vel = vel(data)
    
    frames = vel.shape[0]
    features = vel.shape[1]
    body_parts = features//3

    vel_reshaped = vel.reshape((frames, body_parts, 3))

    # u is the velocities (or maybe you could think of them as vectors?) of the current frame
    # v is the velocities of the next frame
    ux, uy, uz = vel_reshaped[:-1, :, 0], vel_reshaped[:-1, :, 1], vel_reshaped[:-1, :, 2]
    vx, vy, vz = vel_reshaped[1:, :, 0], vel_reshaped[1:, :, 1], vel_reshaped[1:, :, 2]

    a0 = (ux, uy)
    for a1 in range(2):
        a2 = ("ux", "uy")
        print(f'the first 20 values of {a2[a1]} is {a0[a1][:20]}')

    ang_u = np.array(ang_of(ux, uy))
    ang_v = np.array(ang_of(vx, vy))

    temp = np.subtract(ang_v, ang_u)
    phi_changes = unsign_to_sign_ang_change(temp)
    #print(type(phi_changes[0]))
    #print(type(phi_changes[0][0]))
    return phi_changes

# data_path = path.join(cd, "raw/boba_apr11.csv")
# monke_ang_changes(np.genfromtxt(data_path, skip_header=3, delimiter=","), path.join(cd, "ang_change/boba_apr11_ang_changes.csv"))

# The 3D angle between velocities in two frames
def ang3d_changes(data):
    pass

## ------------------- LABEL MAKING ------------------- ##

def generate_labelled_frames(pose_data, tremour_data_raw, save_as=None, fps=30):
    labels = []
    for index, row in tremour_data_raw.iterrows():
        start = int(row["temporal_segment_start"]) * fps
        end = int(row["temporal_segment_end"]) * fps
        label = int(row["label"])

        for i in range(start, end):
            labels.append(label)
    
    last_frame = pose_data.shape[0]
    
    remainder = last_frame - len(labels)

    for _ in range(remainder+1):
        labels.append(0)

    if save_as is not None:
        tremours = pd.DataFrame({"label":labels})
        tremours.to_csv(save_as)

    return labels

# pose_data = pd.read_csv("Y2S2/Monke/boba_apr11_accel.csv")
# tremour_data_raw = pd.read_csv("Y2S2/Monke/boba_apr11_tremours.csv")
# generate_labelled_frames(pose_data, tremour_data_raw, "Y2S2/Monke/boba_apr11_frames_annotated.csv")

def generate_labelled_seconds(pose_data, tremour_data_raw, save_as=None):
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
    
    if save_as is not None:
        tremours = pd.DataFrame({"label":labels})
        tremours.to_csv(save_as)
    
    return labels

# pose_data = pd.read_csv(path.join(Path(__file__).parent, "dir_change/boba_apr11_dir_changes.csv"))
# tremour_data_raw = pd.read_csv(path.join(Path(__file__).parent, "raw/boba_apr11_tremours.csv"))
# file_name = path.join(Path(__file__).parent, "dir_change/boba_apr11_labels.csv")
# generate_labelled_seconds(pose_data, tremour_data_raw, file_name)