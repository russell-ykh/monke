# Testing if supervised UMAP reduction can visualise unlabelled data well!

import monke_features as mf
import dim_reduction as dr
import os.path as path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

cd = Path(__file__).parent

train_data = np.genfromtxt(path.join(cd, "ang_change/boba_apr11_ang_changes.csv"), skip_header=1, delimiter=",")[:, 1:]
train_labels = np.genfromtxt(path.join(cd, "ang_change/boba_apr11_labels.csv"), skip_header=1, delimiter=",")[:, 2]

embedding, reducer = dr.umap_reduce(train_data, train_labels)

test_raw = np.genfromtxt(path.join(cd, "raw/boba_apr21.csv"), skip_header=3, delimiter=",")[:, 1:]
test_data = mf.monke_process(test_raw, mf.monke_ang_changes)

test_labels_raw = pd.read_csv(path.join(cd, "raw/boba_apr21_tremours.csv"))
test_labels = mf.generate_labelled_frames(test_data, test_labels_raw)[:-1]

test_embedding = reducer.transform(test_data)

plt.scatter(embedding[:, 0], embedding[:, 1], c=train_labels, cmap="viridis", s=1, alpha=0.5)
plt.savefig(path.join(cd, "ang_change/bobavboba_apr11_supervised_test.png"), dpi=300)
plt.show()

plt.scatter(test_embedding[:, 0], test_embedding[:, 1], c=test_labels, cmap="viridis", s=1, alpha=0.5)
plt.savefig(path.join(cd, "ang_change/bobavboba_apr21_supervised_test.png"), dpi=300)
plt.show()

plt.scatter(embedding[:, 0], embedding[:, 1], c=train_labels, cmap="viridis", s=1, alpha=0.5)
plt.scatter(test_embedding[:, 0], test_embedding[:, 1], c=test_labels, cmap="plasma", s=1, alpha=0.5)
plt.savefig(path.join(cd, "ang_change/bobavboba_supervised_test.png"), dpi=300)
plt.show()