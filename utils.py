"""
Contents: Utils for parsing data
Authors: Jason Ngo and Emily Lin
Date:
"""
# imports from python libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


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
    k_nearest_stations = get_k_nearest_stations(df, station, 3)
    nearby_df = merge_k_nearest_stations(df, k_nearest_stations, station)
    y = nearby_df[category]
    X = nearby_df.drop(category, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def get_k_nearest_stations(df, station, k):
    distance_arr = []
    test_df = df.query("STATION == 'USC00057936'")
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
                           '2TMAX1', '2TMIN1', '2SNOW1',
                           '2SNOWD1', '2PRCP2', 'TMAX2', 'TMIN2', 'SNOW2', 'SNOWD2']
    df_train.append(Y)

    nearby_final = pd.concat(df_train, axis=1, join='inner', sort=False)
    nearby_final = nearby_final.reset_index()
    nearby_final = nearby_final.select_dtypes(exclude=['object'])
    return nearby_final


def clean_data(df, category):
    df[category] = pd.qcut(df[category], q=45, duplicates='drop').astype(str)
    if 'STATION' in df.columns:
        df = df.drop(['STATION'], axis=1)
    return df


"""
For evaluation
- classifier must be fit already
"""


def matrix(classifier, X_test, y_test):
    """Create confusion matrix"""
    print('Confusion Matrix')
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    results = confusion_matrix(y_test, y_pred)  # normalize='true')
    return results


def heatmap(classifier, X_test, y_test):
    """Create heatmap"""
    class_names = (set(y_test))
    titles_options = [  # ("Confusion matrix, without normalization", None),
        ("Normalized Confusion Matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)

    plt.show()


def heatmap2(matrix):
    """Create heatmap using sns"""
    sns.heatmap(matrix, annot=True, cbar=False)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()
