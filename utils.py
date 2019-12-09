"""
Contents: Utils for parsing data
Authors: Jason Ngo and Emily Lin
Date:
"""
#imports from python libraries
import pandas as pd
from sklearn.model_selection import train_test_split

def one_station(file, category):
    data = pd.read_csv(file)
    y = to_categorical(data)
    #y = data.PRCP
    X = data.drop(category, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def to_categorical(data):
    precip = data.PRCP
    converted = list()
    for x in precip:
        if x == 0:
            converted.append('0')
        elif 0<x<=30:
            converted.append('0<x<=30')
        elif 30<x<=70:
            converted.append('30<x<=70')
        elif 70<x<=100:
            converted.append('70<x<=100')
        elif 100<x<=140:
            converted.append('100<x<=140')
        elif 140<x<=200:
            converted.append('140<x<=200')
        elif 200<x<=250:
            converted.append('200<x<=250')
        elif 250<x<=300:
            converted.append('250<x<=300')
        elif 300<x<=400:
            converted.append('300<x<=400')
        else:
            converted.append('400<x')
    return converted
