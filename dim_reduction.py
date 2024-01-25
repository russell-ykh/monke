import pandas as pd
import umap.umap_ as umap
import matplotlib.pyplot as plt
import numpy as np
import os.path as path
import glob
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

def umap_reduce(data, labels=None, n_neighbors=15, n_components=2):
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components)
    reducer.fit(data, y=labels)
    return reducer.embedding_, reducer

def visualise_reduced(reduced, labels, three_dimensional=False, title=None, save_as=None, hide=-1):
    fig = plt.figure()

    if hide == 0 or hide == 1:
        ncolors = 256
        color_array = plt.get_cmap('viridis')(range(ncolors))
        if hide == 0:
            color_array[:,-1] = np.linspace(0.0,1.0,ncolors)
        elif hide == 1:
            color_array[:,-1] = np.linspace(1.0,0.0,ncolors)
        map_object = LinearSegmentedColormap.from_list(name='viridis_alpha',colors=color_array)
        plt.colormaps.register(cmap=map_object)

    ax = fig.add_subplot(projection="3d" if three_dimensional else None)
    
    if three_dimensional:
        ax.scatter(reduced["0"], reduced["1"], reduced["2"], c=labels, cmap="viridis" if hide == -1  else "viridis_alpha")
    else:
        ax.scatter(reduced["0"], reduced["1"], c=labels, cmap="viridis" if hide == -1  else "viridis_alpha")

    ax.set_ylabel("UMAP Y")
    ax.set_xlabel("UMAP X")

    if title is not None:
        ax.set_title(title)

    if save_as is not None:
        plt.savefig(save_as, dpi=300)

    plt.show()

cd = Path(__file__).parent
labels = pd.read_csv(path.join(cd, "ang_change/boba_apr11_labels.csv"))["label"]
writing = False
visualising = False
search_for_nan = False

if(writing):
    data = np.genfromtxt(path.join(cd, "ang_change/boba_apr11_ang_changes.csv"), skip_header=1, delimiter=",")[:, 1:]

    if search_for_nan:
        nan_indices = np.where(np.isnan(data))
        print("Row indices of NaN values:", nan_indices[0])
        print("Column indices of NaN values:", nan_indices[1])

    reduced, reducer = umap_reduce(data)
    df = pd.DataFrame(reduced)
    df.to_csv(path.join(cd, "ang_change/boba_apr11_umap2_unsupervised.csv"))

if(visualising):
    reduced = pd.read_csv(path.join(cd, "ang_change/boba_apr11_umap2_unsupervised.csv"))
    visualise_reduced(reduced, labels, save_as=path.join(cd, "ang_change/boba_apr11_umap2_unsupervised.png"))

# visualise_reduced(reduced, labels, save_as=path.join(cd, "ang_change/boba_apr11_umap2_supervised.png"))

# umap_results = glob.glob(path.join(cd, "dir_change/boba_apr11_umap*.csv"))

# for umap in umap_results:
#     data = pd.read_csv(umap)
#     name = str(umap)
#     head, tail = path.split(umap)
#     title = tail.replace(".csv", "")
#     new_name = name.replace(".csv", ".png")
#     visualise_reduced(data, labels, three_dimensional="3" in name, title=title, save_as=new_name)