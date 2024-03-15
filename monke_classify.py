import numpy as np
import os.path as path
from pathlib import Path
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

# Classify a given set of test data
# train_data: The training data
# train_labels: The training labels (0 or 1) corresponding with the given training data
# test_data: The testing data, in a dictionary with names for each sample of testing data
# test_labels: The corresponding base truth labels for the testing data
# train_weights: Weights for the training data. Default is None.
# classifier (optional): The classifier to use. Default is Random Forest Classifier.
# metrics (optional): The metrics used to analyse the prediction results, in a dictionary with names for each metric.
# process (optional): The monke_process function to use on the training and testing data. Default is None.
def classify(train_data, train_labels, test_data, test_labels, train_weights=None, classifier=None, 
             metrics={"precision":precision_score, "recall":recall_score, "f1":f1_score, "accuracy":accuracy_score}, 
             process=None):
    predictions = {}
    metric_results = {}

    if process != None:
        train_data = process(train_data)

        for name in test_data:
            test_data[name] = process(test_data[name])
    
    if classifier == None:
        classifier = RandomForestClassifier()

    classifier.fit(train_data, train_labels, sample_weight=train_weights)

    for name in test_data:
        predict_labels = classifier.predict(test_data[name])
        truth = test_labels[name]

        predictions[name] = predict_labels
        metric_results[name] = {}
        
        for metric in metrics:
            metric_results[name][metric] = metrics[metric](truth, predict_labels)

    return predictions, metric_results

def test_features():
    cd = Path(__file__).parent

    features = sorted(glob.glob(path.join(cd, "features", "*", "*processed.csv")))
    labels = sorted(glob.glob(path.join(cd, "features", "*", "*labels.csv")))

    summary = {}

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

        print(feature_name)
        print(classification_report(y_test, y_pred))
        print("\n")

        summary[feature_name] = {}
        summary[feature_name]["precision"] = precision_score(y_test, y_pred)
        summary[feature_name]["recall"] = recall_score(y_test, y_pred)
        summary[feature_name]["f1"] = f1_score(y_test, y_pred)

    print(summary)

    summary_to = path.join(cd, "classification", "rf_results_summary.csv")
    pd.DataFrame(summary).to_csv(summary_to)