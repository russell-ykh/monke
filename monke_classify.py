import numpy as np
import os.path as path
from pathlib import Path
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

cd = Path(__file__).parent
features = sorted(glob.glob(path.join(cd, "features", "*", "*processed.csv")))
labels = sorted(glob.glob(path.join(cd, "features", "*", "*labels.csv")))

# summary = {}

clf = RandomForestClassifier()

for i in range(len(features)):
    print(features[i])
    print(labels[i])
    print("---\n")
    processed_feature = np.genfromtxt(features[i], skip_header=1, delimiter=",")[:, 1:]
    feature_labels = np.genfromtxt(labels[i], skip_header=1, delimiter=",")[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(processed_feature, feature_labels, test_size=0.2, random_state=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    feature_name = path.split(Path(features[i]))[-1].split("boba_apr11_")[-1].split("_processed.csv")[0]
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Normal Movement", "Tremor"], ax=ax)
    ax.set_title(feature_name)
    fig.savefig(path.join(cd, "classification", f"{feature_name}.png"), dpi=300)

    # print(feature_name)
    # print(classification_report(y_test, y_pred))
    # print("\n")

    # summary[feature_name] = {}
    # summary[feature_name]["precision"] = precision_score(y_test, y_pred)
    # summary[feature_name]["recall"] = recall_score(y_test, y_pred)
    # summary[feature_name]["f1"] = f1_score(y_test, y_pred)

# print(summary)

# summary_to = path.join(cd, "classification", "rf_results_summary.csv")
# pd.DataFrame(summary).to_csv(summary_to)