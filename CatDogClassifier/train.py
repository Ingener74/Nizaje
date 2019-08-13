#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import shutil
from zipfile import ZipFile
from pprint import pprint
import click

from kaggle import api
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def train():
    COMPETITION = 'dogs-vs-cats'
    DATASET_PATH = 'dataset'
    SAMPLE_CVS = 'sampleSubmission.csv'
    TRAIN_ZIP = 'train.zip'
    TEST_ZIP = 'test1.zip'

    TRAIN_SET = 'train'
    VALIDATION_SET = 'valid'
    TEST_SET = 'test'
    CAT = 'cat'
    DOG = 'dog'

    LOCAL_SAMPLE_CVS = os.path.join(DATASET_PATH, SAMPLE_CVS)
    LOCAL_TRAIN_ZIP = os.path.join(DATASET_PATH, TRAIN_ZIP)
    LOCAL_TEST_ZIP = os.path.join(DATASET_PATH, TEST_ZIP)

    TRAIN_DIR = os.path.join(DATASET_PATH, TRAIN_ZIP[:-4])
    TEST_DIR = os.path.join(DATASET_PATH, TEST_ZIP[:-4])

    LOCAL_TRAIN_SET = os.path.join(TRAIN_DIR, 'train')
    LOCAL_VALIDATION_SET = os.path.join(TRAIN_DIR, 'valid')
    LOCAL_TEST_SET = os.path.join(TRAIN_DIR, 'test')

    LOCAL_TRAIN_SET_CAT = os.path.join(LOCAL_TRAIN_SET, CAT)
    LOCAL_VALIDATION_SET_CAT = os.path.join(LOCAL_VALIDATION_SET, CAT)
    LOCAL_TEST_SET_CAT = os.path.join(LOCAL_TEST_SET, CAT)

    LOCAL_TRAIN_SET_DOG = os.path.join(LOCAL_TRAIN_SET, DOG)
    LOCAL_VALIDATION_SET_DOG = os.path.join(LOCAL_VALIDATION_SET, DOG)
    LOCAL_TEST_SET_DOG = os.path.join(LOCAL_TEST_SET, DOG)

    print('Make dataset dir...')
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH, exist_ok=True)
    print('    Done')

    print('Kaggle api...')
    api.authenticate()
    print('    Done')
    if not os.path.exists(LOCAL_SAMPLE_CVS):
        api.competition_download_file(competition=COMPETITION, file_name=SAMPLE_CVS, path=DATASET_PATH)
    print('    Done')
    print('Download train.zip...')
    if not os.path.exists(LOCAL_TRAIN_ZIP):
        api.competition_download_file(competition=COMPETITION, file_name=TRAIN_ZIP, path=DATASET_PATH)
    print('    Done')
    print('Download test.zip...')
    if not os.path.exists(LOCAL_TEST_ZIP):
        api.competition_download_file(competition=COMPETITION, file_name=TEST_ZIP, path=DATASET_PATH)
    print('    Done')

    print('Extract train.zip...')
    if not os.path.exists(TRAIN_DIR):
        z = ZipFile(LOCAL_TRAIN_ZIP)
        z.extractall(DATASET_PATH)
    print('    Done')

    print('Prepare train dir...')
    if not os.path.exists(LOCAL_TRAIN_SET):
        print('Make dirs for train...')
        os.makedirs(LOCAL_TRAIN_SET_CAT)
        os.makedirs(LOCAL_VALIDATION_SET_CAT)
        os.makedirs(LOCAL_TEST_SET_CAT)
        os.makedirs(LOCAL_TRAIN_SET_DOG)
        os.makedirs(LOCAL_VALIDATION_SET_DOG)
        os.makedirs(LOCAL_TEST_SET_DOG)
        print('    Done')

        m = 12500

        train_set_count = 11000
        valid_set_count = 1400
        test_set_count = 100

        print('Check dataset sizes')
        assert(train_set_count + valid_set_count + test_set_count <= m)
        print('    Done')

        c = 0

        print('Copy images for train')
        for i in range(0, train_set_count):
            shutil.copyfile(
                os.path.join(TRAIN_DIR, f'cat.{c + i}.jpg'),
                os.path.join(LOCAL_TRAIN_SET_CAT, f'cat.{c + i}.jpg')
            )
            shutil.copyfile(
                os.path.join(TRAIN_DIR, f'dog.{c + i}.jpg'),
                os.path.join(LOCAL_TRAIN_SET_DOG, f'dog.{c + i}.jpg')
            )
        print('    Done')

        c += train_set_count

        print('Copy images for validation')
        for i in range(0, valid_set_count):
            shutil.copyfile(
                os.path.join(TRAIN_DIR, f'cat.{c + i}.jpg'),
                os.path.join(LOCAL_VALIDATION_SET_CAT, f'cat.{c + i}.jpg')
            )
            shutil.copyfile(
                os.path.join(TRAIN_DIR, f'dog.{i}.jpg'),
                os.path.join(LOCAL_VALIDATION_SET_DOG, f'dog.{c + i}.jpg')
            )
        print('    Done')

        c += valid_set_count

        print('Copy images for test')
        for i in range(0, test_set_count):
            shutil.copyfile(
                os.path.join(TRAIN_DIR, f'cat.{c + i}.jpg'),
                os.path.join(LOCAL_TEST_SET_CAT, f'cat.{c + i}.jpg')
            )
            shutil.copyfile(
                os.path.join(TRAIN_DIR, f'dog.{c + i}.jpg'),
                os.path.join(LOCAL_TEST_SET_DOG, f'dog.{c + i}.jpg')
            )
        print('    Done')

        print('Files for Train ' + str(len(os.listdir(LOCAL_TRAIN_SET))))
        print('Files for Validation ' + str(len(os.listdir(LOCAL_VALIDATION_SET))))
        print('Files for Test ' + str(len(os.listdir(LOCAL_TEST_SET))))
    print('    Done')

    if not os.path.exists(TEST_DIR):
        z = ZipFile(LOCAL_TEST_ZIP)
        z.extractall(DATASET_PATH)

    print('Create model...')
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

    print('    Done')

    model.summary()

    train_data_generator = ImageDataGenerator(rescale=1./255)
    train_generator = train_data_generator.flow_from_directory(
        LOCAL_TRAIN_SET, 
        target_size=(150, 150), 
        batch_size=100, 
        class_mode='binary'
    )

    validation_data_generator = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_data_generator.flow_from_directory(
        LOCAL_VALIDATION_SET,
        target_size=(150, 150),
        batch_size=100,
        class_mode='binary'
    )

    history = model.fit_generator(
        train_generator, 
        steps_per_epoch=110, 
        epochs=20, 
        validation_data=validation_generator,
        validation_steps=5)

    with open('log', 'w') as f:
        json.dump(history.history, f, indent=4, separators=(',', ':'))
    
    pprint(history.history)

    plot_history(history.history)

    model.save('cat_dog.h5')

def plot_history(history):
    acc = history['acc']
    loss = history['loss']
    val_acc = history['val_acc']
    val_loss = history['val_loss']

    plt.subplot(2, 1, 1)
    plt.plot(range(len(acc)), acc, label='Training')
    plt.plot(range(len(val_acc)), val_acc, label='Validation')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(range(len(loss)), loss, label='Trainig')
    plt.plot(range(len(val_loss)), val_loss, label='Validation')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)

    plt.show()

@click.command()
@click.option('--plot/--no-plot', default=False)
def main(plot):
    if plot:
        plot_history(json.load(open('log')))
    else:
        train()
        
if __name__ == "__main__":
    main()
