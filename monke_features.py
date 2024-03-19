import os.path as path
import numpy as np
import pandas as pd
from pathlib import Path

# cd = Path(__file__).parent
# data = np.genfromtxt(path.join(cd, "raw", "boba_apr11.csv"), skip_header=3, delimiter=",")[:, 1:]

# General-purpose function for processing DLC 3D coordinate data CSV files
# data: Input data, usually as a Numpy array
# process: Function for processing input data, must return a new Numpy array
# save_as (optional): File name to save new processed array under
def monke_process(data, process, save_as=None):
    processed = process(data)

    if save_as is not None:
        pd.DataFrame(processed).to_csv(save_as)
    
    return processed

# Calculates velocity
def vel(data):
    return np.diff(data, axis=0)

# Calculates acceleration
def accel(data):
    return np.diff(data, n=2, axis=0)

# Calculates jerk
def jerk(data):
    return np.diff(data, n=3, axis=0)

# General purpose function for n-th derivative of the pose data
def diff(n):
    def diff_internal(data):
        return np.diff(data, n=n, axis=0)
    return diff_internal
    
# The number of times the sign changes in a second
def dir_changes(fps=30):
    def dir_changes(data):
        velocity = vel(data)
        dir_changes_raw = np.sign(velocity[:-1, 1:]) != np.sign(velocity[1:, 1:])

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

# The number of times the sign in velocity changes in a second, across all three axes
# data: monke_dir_changes
def dir_changes_summed(data):
    seconds = data.shape[0]
    feature_axes = data.shape[1]
    features = feature_axes // 3
    reshaped = data.reshape((seconds, features, 3))
    dir_changes_summed = np.sum(reshaped, axis=2)
    return dir_changes_summed

# The angular velocity.
# Angular velocity is the angle between the velocity vectors at frame i and frame i+1.
def ang_changes(data):
    velocity = vel(data)
    
    frames = velocity.shape[0]
    features = velocity.shape[1]
    body_parts = features//3

    vel_reshaped = velocity.reshape((frames, body_parts, 3))

    # u is the velocities (or maybe you could think of them as vectors?) of the current frame
    # v is the velocities of the next frame
    ux, uy, uz = vel_reshaped[:-1, :, 0], vel_reshaped[:-1, :, 1], vel_reshaped[:-1, :, 2]
    vx, vy, vz = vel_reshaped[1:, :, 0], vel_reshaped[1:, :, 1], vel_reshaped[1:, :, 2]

    uv = np.add(np.multiply(ux, vx), np.multiply(uy, vy), np.multiply(uz, vz))
    mag_u = np.sqrt(np.add(ux**2, uy**2, uz**2))
    mag_v = np.sqrt(np.add(vx**2, vy**2, vz**2))

    ang_changes = np.nan_to_num(np.arccos(np.nan_to_num(np.divide(uv, np.multiply(mag_u, mag_v)))))

    return ang_changes

# --- HELPER FUNCTIONS FOR 3D ANGULAR VELOCITY FEATURE ---
def phi_of(x, y):
    temp = np.arctan2(y, x)
    temp[(y <= 0)] += 2*np.pi
    #temp[np.logical_and(x >= 0, y < 0)] += 2*np.pi
    return temp

def theta_of(x, y):
    temp = np.arctan2(y, x)
    #temp[x < 0] += np.pi
    return temp

def unsign_to_sign_ang_change(ang):
    ang_change = np.copy(ang)
    ang_change[ang_change > np.pi] -= 2*np.pi
    ang_change[ang_change <= -np.pi] += 2*np.pi
    return ang_change

def phi_changes(data):
    velocity = vel(data)
    
    frames = velocity.shape[0]
    features = velocity.shape[1]
    body_parts = features//3

    vel_reshaped = velocity.reshape((frames, body_parts, 3))

    # u is the velocities (or maybe you could think of them as vectors?) of the current frame
    # v is the velocities of the next frame
    ux, uy = vel_reshaped[:-1, :, 0], vel_reshaped[:-1, :, 1]
    vx, vy = vel_reshaped[1:, :, 0], vel_reshaped[1:, :, 1]

    ang_u = phi_of(ux, uy)
    ang_v = phi_of(vx, vy)

    temp = np.subtract(ang_v, ang_u)
    phi_changes = unsign_to_sign_ang_change(temp)
    return phi_changes

def theta_changes(data):
    velocity = vel(data)
    
    frames = velocity.shape[0]
    features = velocity.shape[1]
    body_parts = features//3

    vel_reshaped = velocity.reshape((frames, body_parts, 3))

    # u is the velocities (or maybe you could think of them as vectors?) of the current frame
    # v is the velocities of the next frame
    ux, uy, uz = vel_reshaped[:-1, :, 0], vel_reshaped[:-1, :, 1], vel_reshaped[:-1, :, 2]
    vx, vy, vz = vel_reshaped[1:, :, 0], vel_reshaped[1:, :, 1], vel_reshaped[1:, :, 2]

    hyp_u = np.sqrt(np.add(ux**2, uy**2))

    hyp_v = np.sqrt(np.add(vx**2, vy**2))

    ang_u = np.array(theta_of(uz, hyp_u))
    ang_v = np.array(theta_of(vz, hyp_v))

    theta_changes = np.nan_to_num(np.subtract(ang_v, ang_u))
    return theta_changes
# --- END OF HELPER FUNCTIONS ---

# The change in 3D angular velocity.
# Unlike the previous ang_changes feature, this calculates the phi and theta angles of the velocity vector at frame i.
# It then compares these to the phi and theta angles at frame i+1.
def ang3d_changes(data):
    phi = phi_changes(data)
    theta = theta_changes(data)
    ang3d = np.concatenate((phi, theta), axis=1)
    return ang3d

# data = np.genfromtxt(path.join(cd, "raw/boba_apr11.csv"), skip_header=3, delimiter=",")[:, 1:]

# ---- HELPER FUNCTIONS FOR JOINTS FEATURES ----

# Returns the indices of the elements with the given names
# a: Target array
# names: Elements to get
def get_indices(a, elements):
    return [np.where(a == element)[0] for element in elements]

# A function to create a user-defined list
def get_joints(headers):
    joints_total = []
    joints_total.append(get_indices(headers, ["right_ankle", "right_knee", "right_hip"]))
    joints_total.append(get_indices(headers, ["left_ankle", "left_knee", "left_hip"]))
    joints_total.append(get_indices(headers, ["right_wrist", "right_elbow", "right_shoulder"]))
    joints_total.append(get_indices(headers, ["left_wrist", "left_elbow", "left_shoulder"]))
    return joints_total

# headers = np.genfromtxt(path.join(cd, "raw", "boba_apr11.csv"), delimiter=",", dtype=str)[1, 1:][::3]
# joints = get_joints(headers)

# The change in 2D angle of joints, manually identified
# data: raw input positional data
# joints: indices of connected body parts!
def change_in_joint_angle(joints):
    def change_in_joint_angle_internal(data):
        frames = data.shape[0]
        features = data.shape[1]
        body_parts = features//3

        reshaped = data.reshape((frames, 3, body_parts))
        angles_total = []

        for joint in joints:
            joint_a = reshaped[:, :, joint[0]]
            joint_b = reshaped[:, :, joint[1]]
            joint_c = reshaped[:, :, joint[2]]
            ab = np.subtract(joint_b, joint_a)
            bc = np.subtract(joint_c, joint_b)
            dot_product = np.sum(ab * bc, axis=1)
            mag_ab = np.linalg.norm(ab, axis=1)
            mag_bc = np.linalg.norm(bc, axis=1)
            angles = np.arccos(dot_product / (mag_ab * mag_bc))
            angles_total.append(angles)
        
        angles_merged = np.stack(angles_total, axis=1)

        return np.diff(angles_merged, axis=0)

    return change_in_joint_angle_internal

# The angular acceleration of joints, manually identified
# data: raw input positional data
# joints: indices of connected body parts!
def change_in_change_in_joint_angle(joints):
    def change_in_change_in_joint_angle_internal(data):
        return np.diff(change_in_joint_angle(joints)(data), axis=0)

    return change_in_change_in_joint_angle_internal

# monke_process(data, change_in_change_in_joint_angle(joints), save_as=path.join(cd, "joints", "boba_apr11_accel.csv"))

# --- ADVANCED FEATURE PROCESSING FUNCTIONS ---

# The number of times the given feature changes SIGNIFICANTLY in a specified window of time
# window: number of frames to consider for changes
# threshold: change in change needed to register as a proper change
def changes_in_changes(raw, window_size, threshold):
    cic = np.diff(raw, axis=0)
    frames = cic.shape[0]
    features = cic.shape[1]

    num_windows = frames - window_size + 1
    results = []

    for i in range(num_windows):
        window_raw = raw[i:i+window_size-1]
        window = cic[i:i+window_size-1]
        window_next = cic[i+1:i+window_size]
        sign_changes = np.sign(window_next) != np.sign(window)

        masked = window_raw * sign_changes
        counts = []
        for feature in range(features):
            differences = np.diff(masked[:, feature][masked[:, feature] != 0])
            count = np.count_nonzero(np.abs(differences) > threshold, axis=0)
            counts.append(count)

        results.append(counts)

    return np.array(results)

def changes_in_changes_in_phi_theta(data):
    velocity = vel(data)
    
    frames = velocity.shape[0]
    features = velocity.shape[1]
    body_parts = features//3

    vel_reshaped = velocity.reshape((frames, body_parts, 3))

    # u is the velocities (or maybe you could think of them as vectors?) of the current frame
    # v is the velocities of the next frame
    ux, uy, uz = vel_reshaped[:-1, :, 0], vel_reshaped[:-1, :, 1], vel_reshaped[:-1, :, 2]
    vx, vy, vz = vel_reshaped[1:, :, 0], vel_reshaped[1:, :, 1], vel_reshaped[1:, :, 2]

    ang_u = phi_of(ux, uy)
    ang_v = phi_of(vx, vy)

    temp = np.subtract(ang_v, ang_u)
    
    phi_changes = unsign_to_sign_ang_change(temp)

    hyp_u = np.sqrt(np.add(ux**2, uy**2))

    hyp_v = np.sqrt(np.add(vx**2, vy**2))

    ang_u = np.array(theta_of(uz, hyp_u))
    ang_v = np.array(theta_of(vz, hyp_v))

    theta_changes = np.nan_to_num(np.subtract(ang_v, ang_u))
    phi_cic = changes_in_changes(phi_changes, 30, 0.2)
    theta_cic = changes_in_changes(theta_changes, 30, 0.2)
    results = np.concatenate((phi_cic, theta_cic), axis=1)
    return results

#accel_new = monke_process(vel(data), lambda x : changes_in_changes(x, 30, 0.2), save_as=path.join(cd, "features", "cic_vel", "boba_apr11_accel_processed.csv"))

# --- END OF ADVANCED FEATURE PROCESSING FUNCTIONS ---

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

    if remainder > 0:
        for _ in range(remainder):
            labels.append(0)
    elif remainder < 0:
        labels = labels[:remainder]

    if save_as is not None:
        tremours = pd.DataFrame({"label":labels})
        tremours.to_csv(save_as)

    return labels

# pose_data = pd.read_csv(path.join(cd, "features", "joints", "boba_apr11_accel.csv"))
# tremour_data_raw = pd.read_csv(path.join(cd, "raw", "boba_apr11_tremours.csv"))
# generate_labelled_frames(accel_new, tremour_data_raw, path.join(cd, "features", "cic_vel", "boba_apr11_accel_labels.csv"))

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

# pose_data = pd.read_csv(path.join(cd, "features", "dir_change", "boba_apr11_dir_changes.csv"))
# tremour_data_raw = pd.read_csv(path.join(cd, "raw", "boba_apr11_tremours.csv"))
# file_name = path.join(cd, "features", "dir_change", "boba_apr11_labels.csv")
# generate_labelled_seconds(pose_data, tremour_data_raw, file_name)