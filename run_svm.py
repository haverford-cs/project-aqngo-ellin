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
    #fit regression models
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr_lin = SVR(kernel='linear', C=100, gamma='auto')
    svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)
    svrs = [svr_rbf, svr_lin, svr_poly]
    plot(svrs, X_train, X_test, y_train, y_test)

def plot(svrs, X_train, X_test, y_train, y_test):
    lw = 2
    kernel_label = ['RBF', 'Linear', 'Polynomial']
    model_color = ['m', 'c', 'g']

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
    for ix, svr in enumerate(svrs):
        axes[ix].plot(X_test, svr.fit(X_train, y_train).predict(X_test), color=model_color[ix], lw=lw,
                      label='{} model'.format(kernel_label[ix]))
        axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor="none",
                         edgecolor=model_color[ix], s=50,
                         label='{} support vectors'.format(kernel_label[ix]))
        axes[ix].scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
                         y[np.setdiff1d(np.arange(len(X)), svr.support_)],
                         facecolor="none", edgecolor="k", s=50,
                         label='other training data')
        axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                        ncol=1, fancybox=True, shadow=True)

    fig.text(0.5, 0.04, 'data', ha='center', va='center')
    fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
    fig.suptitle("Support Vector Regression", fontsize=14)
    plt.show()

if __name__ == "__main__":
    main()
