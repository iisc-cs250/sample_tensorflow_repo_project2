"""
Starter Code in Pytorch for training a multi layer neural network.

** Takes around 30 minutes to train.
"""

import numpy as np
import pdb
import os
from tqdm import tqdm

from matplotlib import pyplot as plt

from utils import AverageMeter

import tensorflow as tf
from tensorflow import keras


def create_MLP():
    """Create a model of multi-layer-neural-net
    """

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28, 1)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":

    number_epochs = 5

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    model = create_MLP()

    model.fit(train_images, train_labels, epochs=number_epochs)
    model.save('./models/MLP_model.h5')
