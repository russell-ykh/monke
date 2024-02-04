import pandas as pd
import umap.umap_ as umap
import matplotlib.pyplot as plt
import numpy as np
import os.path as path
import monke_features as mf
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

cd = Path(__file__).parent

def umap_reduce(data, labels=None, n_neighbors=15, n_components=2):
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components)
    reducer.fit(data, y=labels)
    return reducer.embedding_, reducer

def visualise_reduced(reduced, labels, three_dimensional=False, title=None, save_as=None, hide=-1):
    fig = plt.figure()

    if hide == 0 or hide == 1:
        ncolors = 256
        color_array = plt.get_cmap('rainbow')(range(ncolors))
        if hide == 0:
            color_array[:,-1] = np.linspace(0.0,1.0,ncolors)
        elif hide == 1:
            color_array[:,-1] = np.linspace(1.0,0.0,ncolors)
        map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)
        plt.colormaps.register(cmap=map_object)

    ax = fig.add_subplot(projection="3d" if three_dimensional else None)
    
    if three_dimensional:
        ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=labels, s=1, cmap="rainbow" if hide == -1  else "rainbow_alpha")
    else:
        ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, s=1, cmap="RdYlGn_r" if hide == -1  else "rainbow_alpha")

    ax.set_ylabel("UMAP Y")
    ax.set_xlabel("UMAP X")

    if title is not None:
        ax.set_title(title)

    if save_as is not None:
        plt.savefig(save_as, dpi=300)

    plt.show()

def pipeline(data, labels, visualising, save_as=None, search_for_nan=False):
    writing = (save_as is not None)

    if(writing):
        if search_for_nan:
            nan_indices = np.where(np.isnan(data))
            print("Row indices of NaN values:", nan_indices[0])
            print("Column indices of NaN values:", nan_indices[1])

        reduced, reducer = umap_reduce(data)
        df = pd.DataFrame(reduced)
        df.to_csv(save_as)

    if(visualising):
        if not(writing):
            df = data
        visualise_reduced(df, labels, save_as=save_as)

# data = np.genfromtxt(path.join(cd, "ang3d_change/boba_apr11_ang3d_change.csv"), skip_header=1, delimiter=",")[:, 1:]
# labels = pd.read_csv(path.join(cd, "ang3d_change/boba_apr11_labels.csv"))["label"]
# save_as = path.join(cd, "ang3d_change/boba_apr11_umap2_unsupervised.png")

# pipeline(data, labels, True, save_as=save_as)
        
reduced = np.genfromtxt(path.join(cd, "ang3d_change/boba_apr11_umap2_unsupervised.csv"), skip_header=1, delimiter=",")[:, 1:]
labels = pd.read_csv(path.join(cd, "ang3d_change/boba_apr11_labels.csv"))["label"]
save_as = path.join(cd, "ang3d_change/boba_apr11_umap2_unsupervised.png")
visualise_reduced(reduced, labels, title="Phi and Theta Changes", save_as=save_as)

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