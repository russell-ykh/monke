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
# Returns the raw pose data without headers or index column
def read_pose(id, full_path=False):
    return np.genfromtxt(id if full_path else path.join(cd, "raw", "pose", f"{id}.csv"), delimiter=",", skip_header=3)[:, 1:]

# id: name and date of the desired file
# full_path (optional): Indicates that the id is the full path of the file rather than the name and date
# Returns the raw tremour data as a Pandas DataFrame
def read_tremors(id, full_path=False):
    return pd.read_csv(id if full_path else path.join(cd, "raw", "tremors", f"{id}_tremors.csv"))

# --- 2D DATA FOR VIDEOS AND SKELETONS ---

# id: name and date of the desired file
# full_path (optional): Indicates that the id is the full path of the file rather than the name and date
# Returns the 2D DLC pose data
def read_pose2d(id, full_path=False):
    return np.genfromtxt(id if full_path else path.join(cd, "raw", "2d", "{id}.csv"), skip_header=3, delimiter=",")[:, 1:]

# id: name and date of the desired file
# Returns the path to the video file
def read_video_data(id):
    return path.join(cd, "raw", "2d", "{id}.mp4")

# --- PROCESSED FEATURE DATA ---

# TBD or maybe not at all