import numpy as np
import os.path as path
import pandas as pd
from pathlib import Path

# General utility module for reading / writing different kinds of monke data
# This uses the __file__ trick to work so it may not on all systems

cd = Path(__file__).parent

# --- STANDARD RAW DATA ---

# id: name and date of the desired file
# full_path (optional): Indicates that the id is the full path of the file rather than the name and date
# Returns the headers of a pose data file
def read_header(id, full_path=False):
    headers_raw = np.genfromtxt(id if full_path else path.join(cd, "raw", "pose", f"{id}.csv"), delimiter=",", dtype=str)[1, 1:]
    return headers_raw[::3]

# id: name and date of the desired file
# full_path (optional): Indicates that the id is the full path of the file rather than the name and date
# Returns the raw pose data without headers or index column
def read_pose(id, full_path=False):
    return np.genfromtxt(id if full_path else path.join(cd, "raw", "pose", f"{id}.csv"), delimiter=",", skip_header=3)[:, 1:]

# ids: names and dates of the desired files
# full_path (optional): Indicates that the ids are the full paths of the files rather than the names and dates
# Returns a dictionary of raw pose data without headers or index column
def read_poses(ids, full_path=False):
    poses = {}
    for id in ids:
        poses[id] = read_pose(id, full_path=full_path)
    return poses

# id: name and date of the desired file
# full_path (optional): Indicates that the id is the full path of the file rather than the name and date
# Returns the raw tremour data as a Pandas DataFrame
def read_tremors(id, full_path=False):
    return pd.read_csv(id if full_path else path.join(cd, "raw", "tremors", f"{id}_tremors.csv"))

# ids: names and dates of the desired files
# full_path (optional): Indicates that the ids are the full paths of the files rather than the names and dates
# Returns the raw tremour data as a Pandas DataFrame
def read_tremors_multi(ids, full_path=False):
    tremors_multi = {}
    for id in ids:
        tremors_multi[id] = read_tremors(id, full_path=full_path)
    return tremors_multi

# id: name and date of the desired files
# full_path (optional): Indicates that the id is the full path of the file rather than the name and date
# Returns the average likelihood of each DLC pose point without headers or index column
def read_weights(id, full_path=False):
    pose2d1 = read_pose2d(f"{id}_camera1", full_path=full_path)
    pose2d2 = read_pose2d(f"{id}_camera2", full_path=full_path)

    rs1 = pose2d1.reshape((pose2d1.shape[0], pose2d1.shape[1]//3, 3))
    likelihood1 = rs1[:, :, 2]

    rs2 = pose2d2.reshape((pose2d2.shape[0], pose2d2.shape[1]//3, 3))
    likelihood2 = rs2[:, :, 2]

    frames1 = likelihood1.shape[0]
    frames2 = likelihood2.shape[0]

    if frames1 > frames2:
        likelihood1 = likelihood1[:frames2, :]
    elif frames2 > frames1:
        likelihood2 = likelihood2[:frames1, :]
    
    return np.mean((likelihood1 + likelihood2)/2, axis=1)

# ids: names and dates of the desired files
# full_path (optional): Indicates that the ids are the full paths of the files rather than the names and dates
# Returns the average likelihood of each DLC pose point without headers or index column
def read_weights_multi(ids, full_path=False):
    weights_multi = {}
    for id in ids:
        weights_multi[id] = read_weights(id, full_path=full_path)
    return weights_multi

# --- 2D DATA FOR VIDEOS AND SKELETONS ---

# id: name and date of the desired file
# full_path (optional): Indicates that the id is the full path of the file rather than the name and date
# Returns the 2D DLC pose data
def read_pose2d(id, full_path=False):
    return np.genfromtxt(id if full_path else path.join(cd, "raw", "2d", f"{id}.csv"), skip_header=3, delimiter=",")[:, 1:]

# id: name and date of the desired file
# Returns the path to the video file
def read_video_data(id):
    return path.join(cd, "raw", "2d", "{id}.mp4")

# --- PROCESSED FEATURE DATA ---

# TBD or maybe not at all