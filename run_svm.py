"""
Contents: Train and test support vectors for regression
Authors: Jason Ngo and Emily Lin
Date:
"""

# imports from python libraries
import numpy as np
import pandas as pd

from sklearn.svm import SVR, LinearSVR, SVC
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
        X_train, X_test, y_train, y_test = one_station_split(
            df, 'USC00057936', category)
    elif NEARBY_STATION:
        print("Run SVM on nearby stations")
        X_train, X_test, y_train, y_test = nearby_station_split(
            df, 'USC00057936', category)

    # fit regression models for SVC
    svc_rbf = SVC(kernel='rbf', C=100, gamma=0.1).fit(X_train, y_train)
    heatmap(svc_rbf, X_test, y_test)


if __name__ == "__main__":
    main()
