#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical


def train_dense_network():
    # Prepare dataset
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

    # Create model
    network = Sequential()

    network.add(Dense(512, activation='relu', input_shape=(train_images.shape[1],)))
    network.add(Dense(10, activation='softmax'))

    network.summary()

    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    network.fit(train_images, train_labels, epochs=5, batch_size=128)

    # Evaluate model
    test_loss = network.evaluate(test_images, test_labels)
    print(test_loss)

    # Save model
    # network.save('model_dense.h5')
    return network


def train_conv_network():
    # Prepare dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # np.set_printoptions(linewidth=200)
    # print(train_images[0])

    print(train_images.shape)
    print(test_images.shape)

    train_images = train_images.reshape((train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))
    test_images = test_images.astype('float32') / 255

    # print(train_images[0])

    # return

    print(train_images.shape)
    print(test_images.shape)

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Create model
    network = Sequential()
    network.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    network.add(MaxPooling2D((2, 2)))
    network.add(Conv2D(64, (3, 3), activation='relu'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Conv2D(64, (3, 3), activation='relu'))
    network.add(Flatten())
    network.add(Dense(64, activation='relu'))
    network.add(Dense(10, activation='softmax'))

    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    network.summary()

    # Train model
    history = network.fit(train_images, train_labels, epochs=5, batch_size=64)

    acc = history.history['accuracy']
    loss = history.history['loss']
    plt.plot(range(len(acc)), acc, label='Accuracy')
    plt.plot(range(len(loss)), loss, label='Loss')
    plt.legend()
    plt.show()

    # Evaluate model
    test_loss, test_acc = network.evaluate(test_images, test_labels, batch_size=64)
    print("Accuracy: " + str(test_acc))

    # Save model
    # network.save('model_conv.h5')

    return network


def load_network(filename):
    return load_model(filename)


network = train_dense_network()
# network = train_conv_network()
