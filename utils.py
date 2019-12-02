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
    y = data.PRCP
    X = data.drop(category, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test
