import pandas as pd
import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import numpy as np
import os.path as path
from pathlib import Path

cd = Path(__file__).parent

def umap_reduce(data, labels=None, n_neighbors=15, n_components=2):
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components)
    reducer.fit(data, y=labels)
    return reducer.embedding_, reducer

def visualise_reduced(reduced, labels, three_dimensional=False, title=None, save_fig=None, hide=-1, s=1):
    fig = plt.figure()
    
    colours = ["tab:blue", "red"]

    ax = fig.add_subplot(projection="3d" if three_dimensional else None)

    if hide == 0 or hide == 1:
        to_plot = reduced[labels != hide]
        to_label = labels[labels != hide]

        if hide == 0:
            colours.remove("tab:blue")
    else:
        to_plot = reduced
        to_label = labels

    cmap = mplc.ListedColormap(colours)
    
    if three_dimensional:
        ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2], c=to_label, s=s, cmap=cmap)
    else:
        ax.scatter(to_plot[:, 0], to_plot[:, 1], c=to_label, s=s, cmap=cmap)

    ax.set_ylabel("UMAP Y")
    ax.set_xlabel("UMAP X")
    ax.set_xbound((-9, 15))
    ax.set_ybound((-9, 14))

    if title is not None:
        ax.set_title(title)

    if save_fig is not None:
        plt.savefig(save_fig, dpi=300)

    plt.show()

def pipeline(data, labels, three_dimensional=False, title=None, save_as=None, save_fig=None, search_for_nan=False):
    if(save_as is not None):
        if search_for_nan:
            nan_indices = np.where(np.isnan(data))
            print("Row indices of NaN values:", nan_indices[0])
            print("Column indices of NaN values:", nan_indices[1])

        reduced, reducer = umap_reduce(data, n_components=3 if three_dimensional else 2)
        pd.DataFrame(reduced).to_csv(save_as)

    if(save_fig is not None):
        if (save_as is None):
            reduced = data
        visualise_reduced(reduced, labels, three_dimensional=three_dimensional, title=title, save_fig=save_fig)

# data = np.genfromtxt(path.join(cd, "features", "joints", "boba_apr11_accel.csv"), skip_header=1, delimiter=",")[:, 1:]
# labels = pd.read_csv(path.join(cd, "features", "joints", "boba_apr11_accel_labels.csv"))["label"]
# save_as_path = path.join(cd, "features", "joints", "boba_apr11_accel_umap3.csv")
# save_fig_path = path.join(cd, "features", "joints", "boba_apr11_accel_umap3.png")

# pipeline(data, labels, three_dimensional=True, title="Angular Acceleration of Joints", save_as=save_as_path, save_fig=save_fig_path)

reduced = np.genfromtxt(path.join(cd, "features", "acceleration", "boba_apr11_umap2_unsupervised.csv"), skip_header=1, delimiter=",")[:, 1:]
labels = np.genfromtxt(path.join(cd, "features", "acceleration", "boba_apr11_labels.csv"), skip_header=1, delimiter=",")[:, -1]

for i in [-1, 0, 1]:
    save_fig_path = path.join(cd, "features", "acceleration", f"boba_apr11_accel{'_tremor' if i == 0 else ('_normal' if i == 1 else '')}.png")
    visualise_reduced(reduced, labels, title="Acceleration", save_fig=save_fig_path, hide=i, s=2)

def quick_umap_results(data, process, labels, n_neighbors=15, n_components=2):
    processed = process(data)
    reduced, mapper = umap_reduce(processed)
    visualise_reduced(reduced, labels)

# data_path = path.join(cd, "raw/boba_apr11.csv")
# data = np.genfromtxt(data_path, skip_header=3, delimiter=",")[:, 1:]

# tremour_path = path.join(cd, "raw/boba_apr11_tremours.csv")
# tremours_raw = pd.read_csv(tremour_path)

# labels = mf.generate_labelled_frames(data[:-21, :], tremours_raw)

# quick_umap_results(data, mf.diff(20), labels)