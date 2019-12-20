"""
Contents: Train and test support vectors for regression
Authors: Jason Ngo and Emily Lin
Date: 12/20/19
"""

# imports from python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, \
     validation_curve
from sklearn.metrics import confusion_matrix
from collections import OrderedDict

# imports from our libraries
from utils import *


def main():
    file_path = "ghcnd_hcn/pair_final.csv"
    df = pd.read_csv(file_path)

    category = "PRCP2"
    # Toggle these values to test different functions
    ONE_STATION, NEARBY_STATION = False, True

    if ONE_STATION:
        print("Run SVM on one station")
        X_train, X_test, y_train, y_test, map_dict = one_station_split(
            df, 'USC00057936', category)

        #for cross validation
        X,y = one_station(df, 'USC00057936', category)

    elif NEARBY_STATION:
        print("Run SVM on nearby stations")
        X_train, X_test, y_train, y_test, map_dict = nearby_station_split(
            df, 'USC00057936', 'Precipitation')

        #for cross validation
        X,y = nearby_station(df, 'USC00057936', category)

    elif ALL_STATIONS:
        #TODO
        pass

    # fit regression models for SVC
    #accuracy(X.values, y.values) #get best hyperparameters
    #params = accuracy(X.values, y.values)
    #svc_clf = SVC(params)

    svc_clf = SVC(kernel='rbf', C=1, gamma=0.001).fit(X_train, y_train)
    train_score = accuracy_score(y_train, svc_clf.predict(X_train), sample_weight=None)
    test_score = accuracy_score(y_test, svc_clf.predict(X_test), sample_weight=None)
    print(train_score)
    print(test_score)
    #heatmap(svc_clf, X_test, y_test)
    """
    #create normalized confusion matrix
    od = OrderedDict(sorted(map_dict.items()))
    class_names = [str(entry) for entry in list(od.values())]
    y_pred = svc_clf.fit(X_train, y_train).predict(X_test)
    c_matrix = confusion_matrix(y_test, y_pred, normalize='true')
    for i in range(len(c_matrix)):
        for j in range(len(c_matrix)):
            c_matrix[i][j] = round(c_matrix[i][j], 2)
    print_confusion_matrix(c_matrix, class_names)
    """
    plot_curves(X,y)



#from lab 7

def plot_curves(X,y):
    """plot validation curves"""
    train_accuracy = {}
    test_accuracy = {}
    param_name='gamma'
    svm_param_range = np.logspace(-5, 1, 7)
    train_scores, test_scores = \
        validation_curve(SVC(), X, y, \
                        param_name=param_name, param_range=svm_param_range, \
                        scoring='accuracy')
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    train_accuracy['SVM'] = train_scores_mean
    test_accuracy['SVM'] = test_scores_mean
    generate('SVM', \
        param_name, svm_param_range, train_accuracy, test_accuracy)

    print('\nSVM')
    print('Gamma, Train Accuracy, Test Accuracy')
    for i in range(len(svm_param_range)):
       print(svm_param_range[i], str(train_accuracy['SVM'][i]) + \
       ', ' + str(test_accuracy['SVM'][i]))

def generate(method, param_name, param_range, \
        train_accuracy, test_accuracy):
    """generate learning curves for given arguments (method, etc.)"""
    plt.title('Validation Curve for '+method)
    plt.xlabel(param_name)
    plt.ylabel('Score')
    #use plt.semilogx when plotting for svm
    plt.semilogx(param_range, train_accuracy[method], label='Training score', \
        color='blue', marker='o')
    plt.semilogx(param_range, test_accuracy[method], label='Testing score', \
        color='red', marker='o')
    plt.legend(loc='best')
    plt.show()

def accuracy(X, y):
    train_accuracy = {}
    test_accuracy = {}
    parameters = {}

    gamma = np.logspace(-4, 0, 5)
    #default kernel is Gaussian/rbf
    svc_parameters = {'C': [1, 10, 100, 1000], 'gamma': gamma}
    svc_clf = SVC()
    train_scores, test_scores, params = \
        runTuneTest(svc_clf, svc_parameters, X, y)
    train_accuracy['svc'] = train_scores
    test_accuracy['svc'] = test_scores
    parameters['svc'] = params
    print('params', params) #development

    for i in range(5):
        print('\nFold ' + str(i+1) + ':')
        #print('rf:', parameters['rf'][i])
        print('svc:', parameters['svc'][i])
        print('Training Score (svc): ' + \
                #str(train_accuracy['rf'][i]) + ", " + \
                str(train_accuracy['svc'][i]))
    print('\nFold, SVC Test Accuracy')
    sum = 0
    #rf_total = 0
    svc_total = 0
    for i in range(5):
        #rf_acc = test_accuracy['rf'][i]
        #rf_total += rf_acc
        svc_acc = test_accuracy['svc'][i]
        svc_total += svc_acc
        print(str(i+1) + ', ' + str(svc_acc))
    #rf_avg = rf_total/5
    svc_avg = svc_total/5
    print('\nAverage test accuracy (svc): \n'+ \
             str(svc_avg))

    return parameters['svc']

def runTuneTest(learner, params, X, y):
    """
    takes in base learner, hyperparameters to tune, and all data;
    creates train, tune, and test sets and runs the pipeline
    """
    #divide data
    skf = StratifiedKFold(5, True, 42)

    test_scores = []
    train_scores = []
    best_params = []

    i = 0
    for train_index, test_index in skf.split(X, y):
        #print('X', X)
        #print('y', y)
        print('i', i)
        i += 1
        """
        print('train index', train_index)
        X_train = X.loc[:, train_index]
        print('X[train_index]', X_train)
        print('X[test_index]', X.loc[test_index, :])
        """
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print('gridsearch')
        clf = GridSearchCV(learner, params, cv=3)

        #fit classifier to data and record training score
        print('fit')
        clf.fit(X_train, y_train)
        best_params.append(clf.best_params_)
        print('train scores')
        train_score = clf.score(X_train, y_train)
        train_scores.append(train_score)

        print('test scores')
        #get test accuracy
        test_score = clf.score(X_test, y_test)
        test_scores.append(test_score)

    return train_scores, test_scores, best_params

if __name__ == "__main__":
    main()
