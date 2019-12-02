"""
Contents: Train and test support vectors for regression
Authors: Jason Ngo and Emily Lin
Date:
"""

#imports from python libraries
import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn import utils

#imports from our libraries
from utils import one_station

#see lab 7
def main():
    file = 'ghcnd_hcn/USC00447338_rem.csv'
    category = 'PRCP'
    X_train, X_test, y_train, y_test = one_station(file, category)
    #fit regression model
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    #predict
    predictions = svr_rbf.fit(X_train, y_train).predict(X_test)
    print('predictions', predictions)

if __name__ == "__main__":
    main()
