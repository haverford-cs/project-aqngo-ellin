"""
Contents: Utils for parsing data
Authors: Jason Ngo and Emily Lin
Date:
"""
# imports from python libraries
import pandas as pd
import seaborn as sns
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
    # TODO: WORK ON THIS
    pass

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
    titles_options = [#("Confusion matrix, without normalization", None),
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
