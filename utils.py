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

    df, map_dict = clean_data(df, category)   # convert continuous values to bins

    y = df[category]
    X = df.drop(category, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test, map_dict

def nearby_station_split(df, station, category):
    # TODO: WORK ON THIS
    lon, lat = 
    pass


def clean_data(df, category):
    df[category] = pd.qcut(df[category], q=45, duplicates='drop').astype('category')
    map_dict = dict(zip(df[category].cat.codes, df[category]))
    df[category] = df[category].cat.codes

    if 'STATION' in df.columns:
        df = df.drop(['STATION'], axis=1)
    return df, map_dict