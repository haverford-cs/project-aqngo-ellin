#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from sklearn.preprocessing import normalize
from collections import OrderedDict


from utils import *

ONE_STATION, NEARBY_STATION = True, False


def main():
    file_path = "ghcnd_hcn/pair_final.csv"
    df = pd.read_csv(file_path)
    if ONE_STATION:
        df = df.drop(['LON', 'LAT', 'ELEV'], axis=1)
        df = df.sample(frac=1).reset_index(drop=True)
        X_train, X_test, y_train, y_test, map_dict = one_station_split(
            df, 'USC00057936', 'PRCP2')

    train_stats = X_train.describe()
    train_stats = train_stats.transpose()
    normed_train_data = train_normalize(X_train, train_stats)
    normed_test_data = train_normalize(X_test, train_stats)

    model = build_model(len(map_dict))

    history = model.fit(normed_train_data.values, y_train.values,
                        epochs=5, validation_split=0.2,
                        verbose=0, batch_size=16,
                        callbacks=[tfdocs.modeling.EpochDots()])

    test_loss, test_acc = model.evaluate(
        normed_test_data.values,  y_test.values, verbose=2)

    print("\nTest Accuracy: ", test_acc)

    predictions = model.predict(normed_test_data.values)
    predicted_labels = np.array([np.argmax(row) for row in predictions])

    matrix = tf.math.confusion_matrix(
        y_test,
        predicted_labels,
        num_classes=None,
        weights=None,
        dtype=tf.dtypes.int32,
        name=None).numpy()
    normed_matrix = normalize(matrix, axis=1, norm='l1')
    normed_matrix = np.around(normed_matrix, decimals=2)

    plot_train_validation_curve(history)

    od = OrderedDict(sorted(map_dict.items()))
    class_names = [str(entry) for entry in list(od.values())]
    print_confusion_matrix(normed_matrix, class_names)



def build_model(num_classes):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    main()
