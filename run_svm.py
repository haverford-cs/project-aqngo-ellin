"""
Contents: Train and test support vectors for regression
Authors: Jason Ngo and Emily Lin
Date:
"""

#imports from python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVR, LinearSVR
from sklearn import utils

#imports from our libraries
from utils import one_station

#see lab 7
def main():
    file = 'ghcnd_hcn/USC00447338_rem.csv'
    category = 'PRCP'
    X_train, X_test, y_train, y_test = one_station(file, category)
    #fit regression models
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr_lin = LinearSVR(C=100)
    svr_poly = SVR(kernel='poly',C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
    svrs = [svr_rbf, svr_lin, svr_poly]
    plot(svrs, X_train[:20], X_test[:2], y_train[:20], y_test[:2])

def plot(svrs, X_train, X_test, y_train, y_test):
    print('plotting')
    lw = 2
    kernel_label = [RBF', 'Linear', 'Polynomial']
    model_color = ['m', 'c', 'g']

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
    for ix, svr in enumerate(svrs):
        axes[ix].scatter(y_test, svr.fit(X_train, y_train).predict(X_test), color=model_color[ix], lw=lw,
                      label='{} model'.format(kernel_label[ix]))
        axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                        ncol=1, fancybox=True, shadow=True)

        fig.text(0.5, 0.04, 'Actual', ha='center', va='center')
        fig.text(0.06, 0.5, 'Predicted', ha='center', va='center', rotation='vertical')
        fig.suptitle("Support Vector Regression", fontsize=14)
        plt.show()

if __name__ == "__main__":
    main()
