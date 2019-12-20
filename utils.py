"""
Contents: Utils for parsing data and plotting confusion matrices
Authors: Jason Ngo and Emily Lin
Date: 12/20/19
"""
# imports from python libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn import utils


def one_station_split(df, station, category):
    """
    Parse data for one station prediction
    Returns train and test sets as well as a dictionary of labels
    """
    df = df.query("STATION == '{}'".format(station))
    df, map_dict = clean_data(df, category)
    y = df[category]
    X = df.drop(category, axis=1)
    X, y = utils.shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test, map_dict


def nearby_station_split(df, station, category):
    """
    Parse data for nearby stations prediction
    Returns train and test sets as well as a dictionary of labels
    """
    k_nearest_stations = get_k_nearest_stations(df, station, 3)
    nearby_df = merge_k_nearest_stations(df, k_nearest_stations, station)
    nearby_df, map_dict = clean_data_nearby(nearby_df, category)
    y = nearby_df[category]
    X = nearby_df.drop(category, axis=1)
    X, y = utils.shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test, map_dict


def one_station(df, station, category):
    """
    Parse data for one station cross validation
    Returns examples and labels
    """
    df = df.query("STATION == '{}'".format(station))
    df, map_dict = clean_data(df, category)

    y = df[category]
    X = df.drop(category, axis=1)
    X, y = utils.shuffle(X, y)
    return X, y


def nearby_station(df, station, category):
    """
    Parse data for nearby stations cross validation
    Returns examples and labels
    """
    k_nearest_stations = get_k_nearest_stations(df, station, 3)
    nearby_df = merge_k_nearest_stations(df, k_nearest_stations, station)
    nearby_df, map_dict = clean_data_nearby(nearby_df, category)
    y = nearby_df[category]
    X = nearby_df.drop(category, axis=1)
    X, y = utils.shuffle(X, y)
    return X, y


def get_k_nearest_stations(df, station, k):
    """
    Get k nearest stations to use for nearby stations prediction
    """
    distance_arr = []
    test_df = df.query("STATION == '{}'".format(station))
    test_location = np.array([test_df[element].values[0]
                              for element in ['LON', 'LAT', 'ELEV']])
    for station_id, station_name in enumerate(df['STATION'].unique()):
        nearby_df = df.query("STATION == '{}'".format(station_name))
        nearby_location = np.array(
            [nearby_df[element].values[0] for element in ['LON', 'LAT', 'ELEV']])
        nearby_distance = np.linalg.norm(test_location - nearby_location)

        distance_arr.append(nearby_distance)

    k_nearest_stations = df['STATION'].unique(
    )[np.argpartition(distance_arr, k)[:k]]

    return k_nearest_stations


def merge_k_nearest_stations(df, k_nearest_stations, test_station):
    """
    Merge data for k nearest stations to use for nearby stations prediction
    """
    df_train = []
    df_test = []
    for station in k_nearest_stations:
        print(station)
        if station != test_station:
            df_temp = df.query("STATION == '{}'".format(station))
            df_temp = df_temp.set_index(['DATE'])
            df_train.append(df_temp)
        else:
            df_test = df.query("STATION == '{}'".format(station))
            df_test = df_test.set_index(['DATE'])
    Y = pd.DataFrame(df_test['PRCP2'])
    Y.columns = ['Precipitation']

    df_train[1].columns = ['2STATION', '2LAT', '2LON', '2ELEV', '2PRCP1',
                           '2TMAX1', '2TMIN1', '2SNOW1', '2SNOWD1', '2PRCP2',
                           'TMAX2', 'TMIN2', 'SNOW2', 'SNOWD2']
    df_train.append(Y)

    nearby_final = pd.concat(df_train, axis=1, join='inner', sort=False)
    nearby_final = nearby_final.reset_index()
    nearby_final = nearby_final.select_dtypes(exclude=['object'])
    return nearby_final


def clean_data(df, category):
    """
    Convert labels from continous to categorical for one station prediction
    """
    if 'STATION' in df.columns:
        df = df.drop(['STATION'], axis=1)

    df[category] = pd.qcut(
        df[category], q=45, duplicates='drop').astype('category')
    map_dict = dict(zip(df[category].cat.codes, df[category]))
    df[category] = df[category].cat.codes
    return df, map_dict


def clean_data_nearby(df, category):
    """
    Convert labels from continous to categorical for nearby stations prediction
    """
    df[category] = pd.qcut(
        df[category], q=45, duplicates='drop').astype('category')
    map_dict = dict(zip(df[category].cat.codes, df[category]))
    df[category] = df[category].cat.codes
    return df, map_dict


def train_normalize(x, train_stats):
    """Normalize data"""
    return (x - train_stats['mean']) / train_stats['std']


def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7),
                           fontsize=14):
    """
    Prints a confusion matrix as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object from using sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order of the confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size
        of the ouputted figure, the second determining the vertical size.
        Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    sns.set()
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize, dpi=65)
    try:
        heatmap = sns.heatmap(df_cm, cmap=plt.cm.Blues, annot=True)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',
        fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',
        fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Normalized Confusion Matrix')
    plt.show()


def plot_train_validation_curve(history):
    """Plot training and validation curves

    Arguments:
        history {tensorflow history object} -- Training history info
    """
    fig = plt.figure(figsize=(14, 8), dpi=200)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('FC Training Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
