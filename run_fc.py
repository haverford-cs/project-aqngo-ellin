"""
Contents: Train and test fully connected neural network
Authors: Jason Ngo and Emily Lin
Date:
"""

#imports from python libraries
import numpy as np
from tensorflow.keras.losses import SparseCategoricalCrossentropy
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow import GradientTape

#imports from our libraries
from fc_nn import FCmodel

#see lab 8
def main():
    model = FCmodel()
    model.compile(loss='mse',
            optimizer=optimizer,
            metrics=['mae', 'mse'])
    run_training(fc_model, train_dset)

def run_training(model, train_dset):
    loss_fn = SparseCategoricalCrossentropy()
    optimizer = RMSprop(0.001)
    #optimizer = Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.\
            SparseCategoricalAccuracy(name='train-accuracy')
    for epoch in range(10):
        for images, labels in train_dset:
            loss, predictions = \
            train_step(model, images, labels, optimizer, loss_fn)

def train_step(model, images, labels, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, modle.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions

#see https://www.tensorflow.org/tutorials/keras/regression
def evaluate():
    hist = pd.DataFrame(history.history)
    mse_values = hist['val_mse']

if __name__ == "__main__":
    main()
