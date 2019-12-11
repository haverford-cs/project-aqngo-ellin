"""
Contents: Utils for parsing data
Authors: Jason Ngo and Emily Lin
Date:
"""
# imports from python libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def one_station_split(df, station, category):
    # y = to_categorical(df, category)
    df = df.query("STATION == '{}'".format(station))
    df = clean_data(df, category)   # convert continuous values to bins

    y = df[category]
    X = df.drop(category, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test, bins

def nearby_station_split(df, station, category):
    # TODO: WORK ON THIS
    pass


def clean_data(df, category):
    df[category] = pd.qcut(df[category], q=45, duplicates='drop').astype(str)
    if 'STATION' in df.columns:
        df = df.drop(['STATION'], axis=1)
    return df
