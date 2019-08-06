#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from json import load

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.utils import to_categorical

MODEL_FILE = 'dense_model.h5'

def prepare_dataset():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    print(train_images.shape)
    print(test_images.shape)

    train_images = train_images.reshape((train_images.shape[0], train_images.shape[1] * train_images.shape[2]))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2]))
    test_images = test_images.astype('float32') / 255

    print(train_images.shape)
    print(test_images.shape)

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    print(train_labels.shape)
    print(train_labels[0])

    print(test_labels.shape)
    print(test_labels[0])
    return train_images, train_labels, test_images, test_labels

def train_network():
    train_images, train_labels, test_images, test_labels = prepare_dataset()

    network = Sequential()

    network.add(Dense(512, activation='relu', input_shape=(train_images.shape[1], )))
    network.add(Dense(10, activation='softmax'))

    network.summary()

    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    network.fit(train_images, train_labels, epochs=5, batch_size=128)

    test_loss = network.evaluate(test_images, test_labels)
    print(test_loss)

    network.save(MODEL_FILE)
    return network

def load_network():
    return load_model(MODEL_FILE)

# network = train_network()

train_images, train_labels, test_images, test_labels = prepare_dataset()
network = load_network()

image = train_images[0:1]
# print(image)
output = network.predict(image)

# print(list(range(10)), output[0])
plt.bar(list(range(10)), output[0])
plt.show()
