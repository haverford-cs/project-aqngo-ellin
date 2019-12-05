"""
Contents: Fully connected neural network class
Authors: Jason Ngo and Emily Lin
Date:
"""

#imports from python libraries
import numpy as np

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model
from tensorflow.keras.activations import relu, softmax

class FCmodel(Model):
    def __init__(self):
        super(FCmodel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(4000, activation='relu')
        #number of units should be number of classes
        self.d2 = Dense(1)

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x
