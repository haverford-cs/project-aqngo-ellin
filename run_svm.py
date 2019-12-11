"""
Contents: Train and test support vectors for regression
Authors: Jason Ngo and Emily Lin
Date:
"""

# imports from python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import utils

# imports from our libraries
from utils import *


def main():
    file_path = "ghcnd_hcn/pair_final.csv"
    df = pd.read_csv(file_path)

    category = "PRCP2"
    # Toggle these values to test different functions
    ONE_STATION, NEARBY_STATION = True, False

    if ONE_STATION:
        print("Run SVM on one station")
        X_train, X_test, y_train, y_test, bins = one_station_split(
            df, 'USC00057936', category)
    elif NEARBY_STATION:
        print("Run SVM on nearby stations")
        X_train, X_test, y_train, y_test = nearby_station_split(
            df, 'USC00057936', category)

    # fit regression models for SVC
    svc_rbf = SVC(kernel='rbf', C=100, gamma=0.1)  # categorical
    svcs = [svc_rbf]
    matrix = svc_matrix(svcs, X_train, X_test, y_train, y_test)
    svc_heat(svcs, X_train, X_test, y_train, y_test)


def svc_matrix(svcs, X_train, X_test, y_train, y_test):
    print('Support Vector Confusion Matrix')
    svc = svcs[0]
    y_pred = svc.fit(X_train, y_train).predict(X_test)
    results = confusion_matrix(y_test, y_pred)  # normalize='true')
    return results

def svc_heat(svcs, X_train, X_test, y_train, y_test):
    svc = svcs[0]
    classifier = svc.fit(X_train, y_train)
    class_names = (set(y_test))
    titles_options = [#("Confusion matrix, without normalization", None),
                  ("Normalized Confusion Matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix
    plt.show()

def svc_plot(matrix):
    sns.heatmap(matrix, annot=True, cbar=False)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    main()
