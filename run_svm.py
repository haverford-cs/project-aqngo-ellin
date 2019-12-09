"""
Contents: Train and test support vectors for regression
Authors: Jason Ngo and Emily Lin
Date:
"""

#imports from python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVR, LinearSVR, SVC
from sklearn.metrics import confusion_matrix
from sklearn import utils

#imports from our libraries
from utils import one_station

#see lab 7
def main():
    file = 'ghcnd_hcn/USC00057936_unpaired.csv'
    #file = 'ghcnd_hcn/USC00057936_CO_pair_data.csv' #paired days
    #file = 'ghcnd_hcn/USC00447338_rem.csv'
    category = 'PRCP' #'PRCP2' for unpaired
    #one_station(file, category) #test conversion to categorical
    X_train, X_test, y_train, y_test = one_station(file, category)
    #fit regression models for SVC
    svc_rbf = SVC(kernel='rbf', C=100, gamma=0.1) #categorical
    svcs = [svc_rbf]
    svc_acc(svcs, X_train, X_test, y_train, y_test)

    """
    #fit regression models for SVR
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr_lin = LinearSVR(C=100)
    svr_poly = SVR(kernel='poly',Cscatter=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
    svrs = [svr_rbf] #[svr_rbf, svr_lin, svr_poly]
    svr_plot(svrs, X_train[:20], X_test[:2], y_train[:20], y_test[:2])
    """
def svc_acc(svcs, X_train, X_test, y_train, y_test):
    print('svc plot')
    svc = svcs[0]
    y_pred = svc.fit(X_train, y_train).predict(X_test)
    results = confusion_matrix(y_test, y_pred) #normalize='true')
    print(results)
    return results

"""
[[594   0   0   0   0   0   0   0   0   0]
 [  0 197   0   0   0   0   0   0   0   0]
 [  0   0  51   0   0   0   0   0   0   0]
 [  0   0   0  39   0   0   0   0   0   0]
 [  0   0   0   0  15   0   0   0   0   0]
 [  0   0   0   0   0   4   0   0   0   0]
 [  0   0   0   0   0   0  28   0   0   0]
 [  0   0   0   0   0   0   0  54   0   0]
 [  0   0   0   0   0   0   0   0  15   0]
 [  0   0   0   0   0   0   0   0   0  34]]
"""

def svr_plot(svrs, X_train, X_test, y_train, y_test):
    print('plotting')
    lw = 2
    kernel_label = ['RBF'] #['RBF', 'Linear', 'Polynomial']
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
