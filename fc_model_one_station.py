#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from utils import *


# In[45]:


df = pd.read_csv("ghcnd_hcn/pair_final.csv")
df = df.drop(['LON', 'LAT', 'ELEV'], axis=1)
df = df.sample(frac=1).reset_index(drop=True)

category = 'PRCP2'
X_train, X_test, y_train, y_test, map_dict = one_station_split(
            df, 'USC00057936', category)


# In[46]:


train_stats = X_train.describe()
train_stats = train_stats.transpose()
train_stats


# In[47]:


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(X_train)
normed_test_data = norm(X_test)


# In[48]:


def build_model():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(len(map_dict), activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


# In[59]:


model = build_model()

EPOCHS = 150

history = model.fit(normed_train_data.values, y_train.values,
  epochs=EPOCHS, validation_split = 0.2, verbose=0, batch_size=16,
  callbacks=[tfdocs.modeling.EpochDots()])


# In[78]:


test_loss, test_acc = model.evaluate(X_test.values,  y_test.values, verbose=2)


# In[83]:


len(y_test)


# In[91]:


predictions = model.predict(X_test)
matrix = tf.math.confusion_matrix(
    y_test,
    [0]*1102,
    num_classes=None,
    weights=None,
    dtype=tf.dtypes.int32,
    name=None
)


# In[77]:


# Plot training & validation accuracy values
plt.figure(figsize=(14,8), dpi=200)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('FC Training Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[102]:


a = matrix.numpy()
norm = np.linalg.norm(a, axis = 1, keepdims = True)
for i_, row in enumerate(a):
    a[i_] = row/norm[i_]

plt.figure(figsize=(14,8), dpi=200)
sns.heatmap(a, cmap="YlGnBu")


# In[ ]:




