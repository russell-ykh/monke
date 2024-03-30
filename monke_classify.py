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

def prep_train_test_data(pose_data, labels, train_names, test_names=None, weights=None, test_size=0.2):
    training_data = []
    training_labels = []

    if weights is not None:
        training_weights = []

    testing_data = {}
    testing_labels = {}

    for name in train_names:
        pose_train = pose_data[name]
        labels_train = labels[name]

        if weights is not None:
            weights_train = weights[name]
            X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(pose_train, labels_train, weights_train, test_size=test_size)
            training_weights.append(z_train)
        else:
            X_train, X_test, y_train, y_test = train_test_split(pose_train, labels_train, test_size=test_size)

        training_data.append(X_train)
        training_labels.append(y_train)

        if test_names is not None:
            if name in test_names:
                testing_data[name] = X_test
                testing_labels[name] = y_test
        else:
            testing_data[name] = X_test
            testing_labels[name] = y_test
    
    if test_names is not None:
        for name in test_names:
            if name not in testing_data:
                pose_test = pose_data[name]
                labels_test = labels[name]
                _, X_test, _, y_test = train_test_split(pose_test, labels_test, test_size=test_size)
                testing_data[name] = X_test
                testing_labels[name] = y_test
        
    if(len(training_data) > 1):
        training_data = np.concatenate(training_data)
        training_labels = np.concatenate(training_labels)
        if weights is not None:
            training_weights = np.concatenate(training_weights)
    else:
        training_data = training_data[0]
        training_labels = training_labels[0]
        if weights is not None:
            training_weights = training_weights[0]

    if weights is None:
        return training_data, testing_data, training_labels, testing_labels
    else:
        return training_data, testing_data, training_labels, testing_labels, training_weights

def prep_multi_train_test_data(pose_data, labels, train_names, test_names=None, weights=None, test_size=0.2):
    training_data = {}
    training_labels = {}

    if weights is not None:
        training_weights = {}

    testing_data = {}
    testing_labels = {}

    for name in train_names:
        pose_train = pose_data[name]
        labels_train = labels[name]

        if weights is not None:
            weights_train = weights[name]
            X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(pose_train, labels_train, weights_train, test_size=test_size)
            training_weights[name] = z_train
        else:
            X_train, X_test, y_train, y_test = train_test_split(pose_train, labels_train, test_size=test_size)

        training_data[name] = X_train
        training_labels[name] = y_train
        
        if test_names is not None:
            if name in test_names:
                testing_data[name] = X_test
                testing_labels[name] = y_test
        else:
            testing_data[name] = X_test
            testing_labels[name] = y_test

    if weights is None:
        return training_data, testing_data, training_labels, testing_labels
    else:
        return training_data, testing_data, training_labels, testing_labels, training_weights

def process_data(pose_data, labels, process):
    processed_data = {}
    processed_labels = {}

    for name in pose_data:
        processed_data[name] = process(pose_data[name])
        processed_labels[name] = labels[name][:processed_data[name].shape[0]]
    
    return processed_data, processed_labels

def test_classify(clf, test_data, test_labels):
    predicted_labels = clf.predict(test_data)
    mcc = matthews_corrcoef(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)
    acc = accuracy_score(test_labels, predicted_labels)
    return {"predictions":predicted_labels, "mcc":mcc, "f1":f1, "accuracy":acc}