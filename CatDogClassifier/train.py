#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import shutil
from zipfile import ZipFile
from pprint import pprint
import click
import matplotlib.pyplot as plt


def create_model(v):
    from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
    from keras.models import Sequential
    from keras.applications import VGG16
    from keras.optimizers import RMSprop

    assert 0 < v < 6
    print(f'Selected model: {v}')
    if v == 1:
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

    elif v == 2:
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
        model.add(Dropout(0.3))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Dropout(0.3))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Dropout(0.3))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Dropout(0.3))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
    elif v == 3:
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

    elif v == 4:
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
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

    elif v == 5:
        print('Use transfer learning with VGG16 ImageNet weights base')

        conv = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

        conv.trainable = False

        set_train = False
        for layer in conv.layers:
            if layer.name == 'block5_conv1':
                set_train = True
            layer.trainable = set_train

        model = Sequential()
        model.add(conv)
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        conv.summary()

        # model.summary()
        for layer in conv.layers:
            print(f'{layer.name} is trainable {"+" if layer.trainable else "-"}')

        model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-5), metrics=['acc'])

    model.summary()
    return model


def train_internal(log_file, train_with_plot, train_size, validation_size, test_size, batch, epochs, model_version,
                   model_name, augment_data):
    from kaggle import api
    from keras.preprocessing.image import ImageDataGenerator

    print(f'Train size:      {train_size}')
    print(f'Validation size: {validation_size}')
    print(f'Test size:       {test_size}')
    print(f'Batch:           {batch}')
    print(f'Epochs:          {epochs}')
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

    ########################
    # Check files for training
    ########################

    def check_and_make_dirs(dir_name):
        print(f'Prepare {dir_name} dir...')
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print('    Done')

    check_and_make_dirs(LOCAL_TRAIN_SET)

    if os.path.exists(LOCAL_TRAIN_SET):
        shutil.rmtree(LOCAL_TRAIN_SET)
    if os.path.exists(LOCAL_VALIDATION_SET):
        shutil.rmtree(LOCAL_VALIDATION_SET)
    if os.path.exists(LOCAL_TEST_SET):
        shutil.rmtree(LOCAL_TEST_SET)

    os.makedirs(LOCAL_TRAIN_SET)
    os.makedirs(LOCAL_TRAIN_SET_CAT)
    os.makedirs(LOCAL_TRAIN_SET_DOG)
    os.makedirs(LOCAL_VALIDATION_SET_CAT)
    os.makedirs(LOCAL_VALIDATION_SET_DOG)
    os.makedirs(LOCAL_TEST_SET_CAT)
    os.makedirs(LOCAL_TEST_SET_DOG)

    m = 12500

    train_set_count = train_size
    valid_set_count = validation_size
    test_set_count = test_size

    print('Check dataset sizes')
    assert (train_set_count + valid_set_count + test_set_count <= m)
    print('    Done')

    print(f'Train + Validation + Test = {train_size + validation_size + test_size} < {m}')

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

    print('Files for Train ' + str(len(os.listdir(LOCAL_TRAIN_SET_CAT)) + len(os.listdir(LOCAL_TRAIN_SET_DOG))))
    print('Files for Validation ' + str(
        len(os.listdir(LOCAL_VALIDATION_SET_CAT)) + len(os.listdir(LOCAL_VALIDATION_SET_DOG))))
    print('Files for Test ' + str(len(os.listdir(LOCAL_TEST_SET_CAT)) + len(os.listdir(LOCAL_TEST_SET_DOG))))

    print('    Done')

    if not os.path.exists(TEST_DIR):
        z = ZipFile(LOCAL_TEST_ZIP)
        z.extractall(DATASET_PATH)

    print('Create model...')

    model = create_model(model_version)

    print('    Done')

    augment_config = {
        'rotation_range': 40,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'shear_range': 0.2,
        'zoom_range': 0.2,
        'horizontal_flip': True,
        'fill_mode': 'nearest'
    } if augment_data else {}

    train_data_generator = ImageDataGenerator(
        rescale=1. / 255,
        **augment_config
    )
    train_generator = train_data_generator.flow_from_directory(
        LOCAL_TRAIN_SET,
        target_size=(150, 150),
        batch_size=batch,
        class_mode='binary'
    )

    validation_data_generator = ImageDataGenerator(rescale=1. / 255)

    validation_generator = validation_data_generator.flow_from_directory(
        LOCAL_VALIDATION_SET,
        target_size=(150, 150),
        batch_size=batch,
        class_mode='binary'
    )

    print(f'Steps per epoch for training:   {train_size / batch}')
    print(f'Steps per epoch for validation: {validation_size / batch}')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_size / batch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_size / batch)

    model.save(model_name)

    with open(log_file, 'w') as f:
        json.dump(history.history, f, indent=4, separators=(',', ':'))

    pprint(history.history)

    if train_with_plot:
        plot_history(history.history, log_file)


def plot_history(history, plot_file):
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

    plt.savefig(plot_file + '.png')

    # plt.show()


@click.group()
def cli():
    pass


@click.command()
@click.option('--plot_file', type=str, default='log')
@click.option('--train_with_plot/--no-train_with_plot', default=False)
@click.option('--train_size', default=400)
@click.option('--validation_size', default=100)
@click.option('--test_size', default=200)
@click.option('--batch', default=100)
@click.option('--epochs', default=10)
@click.option('--model', default=1, help='1, 2, ...')
@click.option('--model_name', type=str, default='cat_dog.h5')
@click.option('--augment_data/--no-augment_data', default=False)
def train(plot_file, train_with_plot, train_size, validation_size, test_size, batch, epochs, model, model_name,
          augment_data):
    train_internal(plot_file, train_with_plot, train_size, validation_size, test_size, batch, epochs, model, model_name,
                   augment_data)


@click.command()
@click.option('--plot_file', type=str, default='log')
def plot(plot_file):
    plot_history(json.load(open(plot_file)), plot_file)


@click.command()
@click.argument('model', type=int)
def create(model):
    create_model(model)


cli.add_command(train)
cli.add_command(plot)
cli.add_command(create)

if __name__ == "__main__":
    cli()
