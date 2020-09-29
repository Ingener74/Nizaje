#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import shutil
from pprint import pprint
from zipfile import ZipFile

import click
import matplotlib.pyplot as plt

from Utils import setup_gpu_memory_growth


def create_model(v):
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.optimizers import RMSprop

    assert 0 < v < 6
    print(f'Selected model: {v}')
    if v == 1:
        model = Sequential(layers=[
            Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

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


def train_internal(log_file,
                   train_with_plot,
                   train_show_plot,
                   train_size,
                   validation_size,
                   test_size,
                   batch,
                   epochs,
                   model_version,
                   model_name,
                   augment_data):
    from kaggle import api
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    setup_gpu_memory_growth()

    print(f'Train size:      {train_size}')
    print(f'Validation size: {validation_size}')
    print(f'Test size:       {test_size}')
    print(f'Batch:           {batch}')
    print(f'Epochs:          {epochs}')
    competition: str = 'dogs-vs-cats'
    dataset_path = 'dataset'
    sample_cvs = 'sampleSubmission.csv'
    train_zip = 'train.zip'
    test_zip = 'test1.zip'

    train_set = 'train'
    validation_set = 'valid'
    test_set = 'test'
    cat = 'cat'
    dog = 'dog'

    local_sample_cvs = os.path.join(dataset_path, sample_cvs)
    local_train_zip = os.path.join(dataset_path, train_zip)
    local_test_zip = os.path.join(dataset_path, test_zip)

    train_dir = os.path.join(dataset_path, train_zip[:-4])
    test_dir = os.path.join(dataset_path, test_zip[:-4])

    local_train_set = os.path.join(train_dir, 'train')
    local_validation_set = os.path.join(train_dir, 'valid')
    local_test_set = os.path.join(train_dir, 'test')

    local_train_set_cat = os.path.join(local_train_set, cat)
    local_validation_set_cat = os.path.join(local_validation_set, cat)
    local_test_set_cat = os.path.join(local_test_set, cat)

    local_train_set_dog = os.path.join(local_train_set, dog)
    local_validation_set_dog = os.path.join(local_validation_set, dog)
    local_test_set_dog = os.path.join(local_test_set, dog)

    print('Make dataset dir...')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)
    print('    Done')

    print('Kaggle api...')
    api.authenticate()
    print('    Done')
    if not os.path.exists(local_sample_cvs):
        api.competition_download_file(competition=competition, file_name=sample_cvs, path=dataset_path)
    print('    Done')
    print('Download train.zip...')
    if not os.path.exists(local_train_zip):
        api.competition_download_file(competition=competition, file_name=train_zip, path=dataset_path)
    print('    Done')
    print('Download test.zip...')
    if not os.path.exists(local_test_zip):
        api.competition_download_file(competition=competition, file_name=test_zip, path=dataset_path)
    print('    Done')

    print('Extract train.zip...')
    if not os.path.exists(train_dir):
        z = ZipFile(local_train_zip)
        z.extractall(dataset_path)
    print('    Done')

    ########################
    # Check files for training
    ########################

    def check_and_make_dirs(dir_name):
        print(f'Prepare {dir_name} dir...')
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print('    Done')

    check_and_make_dirs(local_train_set)

    if os.path.exists(local_train_set):
        shutil.rmtree(local_train_set)
    if os.path.exists(local_validation_set):
        shutil.rmtree(local_validation_set)
    if os.path.exists(local_test_set):
        shutil.rmtree(local_test_set)

    os.makedirs(local_train_set)
    os.makedirs(local_train_set_cat)
    os.makedirs(local_train_set_dog)
    os.makedirs(local_validation_set_cat)
    os.makedirs(local_validation_set_dog)
    os.makedirs(local_test_set_cat)
    os.makedirs(local_test_set_dog)

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
            os.path.join(train_dir, f'cat.{c + i}.jpg'),
            os.path.join(local_train_set_cat, f'cat.{c + i}.jpg')
        )
        shutil.copyfile(
            os.path.join(train_dir, f'dog.{c + i}.jpg'),
            os.path.join(local_train_set_dog, f'dog.{c + i}.jpg')
        )
    print('    Done')

    c += train_set_count

    print('Copy images for validation')
    for i in range(0, valid_set_count):
        shutil.copyfile(
            os.path.join(train_dir, f'cat.{c + i}.jpg'),
            os.path.join(local_validation_set_cat, f'cat.{c + i}.jpg')
        )
        shutil.copyfile(
            os.path.join(train_dir, f'dog.{i}.jpg'),
            os.path.join(local_validation_set_dog, f'dog.{c + i}.jpg')
        )
    print('    Done')

    c += valid_set_count

    print('Copy images for test')
    for i in range(0, test_set_count):
        shutil.copyfile(
            os.path.join(train_dir, f'cat.{c + i}.jpg'),
            os.path.join(local_test_set_cat, f'cat.{c + i}.jpg')
        )
        shutil.copyfile(
            os.path.join(train_dir, f'dog.{c + i}.jpg'),
            os.path.join(local_test_set_dog, f'dog.{c + i}.jpg')
        )
    print('    Done')

    print('Files for Train ' + str(len(os.listdir(local_train_set_cat)) + len(os.listdir(local_train_set_dog))))
    print('Files for Validation ' + str(
        len(os.listdir(local_validation_set_cat)) + len(os.listdir(local_validation_set_dog))))
    print('Files for Test ' + str(len(os.listdir(local_test_set_cat)) + len(os.listdir(local_test_set_dog))))

    print('    Done')

    if not os.path.exists(test_dir):
        z = ZipFile(local_test_zip)
        z.extractall(dataset_path)

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
        local_train_set,
        target_size=(150, 150),
        batch_size=batch,
        class_mode='binary'
    )

    validation_data_generator = ImageDataGenerator(rescale=1. / 255)

    validation_generator = validation_data_generator.flow_from_directory(
        local_validation_set,
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

    new_history = {
        'acc': [float(v) for v in history.history['acc']],
        'loss': [float(v) for v in history.history['loss']],
        'val_acc': [float(v) for v in history.history['val_acc']],
        'val_loss': [float(v) for v in history.history['val_loss']],
    }

    with open(log_file, 'w') as f:
        json.dump(new_history, f, indent=4)

    pprint(history.history)

    if train_with_plot:
        plot_history(history.history, log_file, train_show_plot)


def plot_history(history, plot_file, show=False):
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
    if show:
        plt.show()


@click.group()
def cli():
    pass


@click.command()
@click.option('--plot_file', type=str, default='log')
@click.option('--train_with_plot/--no-train_with_plot', default=False)
@click.option('--train_show_plot/--no-train_show_plot', default=False)
@click.option('--train_size', default=400)
@click.option('--validation_size', default=100)
@click.option('--test_size', default=200)
@click.option('--batch', default=100)
@click.option('--epochs', default=10)
@click.option('--model', default=1, help='1, 2, ...')
@click.option('--model_name', type=str, default='cat_dog.h5')
@click.option('--augment_data/--no-augment_data', default=False)
def train(plot_file,
          train_with_plot,
          train_show_plot,
          train_size,
          validation_size,
          test_size,
          batch,
          epochs,
          model,
          model_name,
          augment_data):
    train_internal(plot_file,
                   train_with_plot,
                   train_show_plot,
                   train_size,
                   validation_size,
                   test_size,
                   batch,
                   epochs,
                   model,
                   model_name,
                   augment_data)


@click.command()
@click.option('--plot_file', type=str, default='log')
@click.option('--show_plot/--no-show_plot', default=False)
def plot(plot_file, show_plot):
    plot_history(json.load(open(plot_file)), plot_file, show_plot)


@click.command()
@click.argument('model', type=int)
def create(model):
    create_model(model)


@click.command()
def test_all():
    pass


cli.add_command(train)
cli.add_command(plot)
cli.add_command(create)
cli.add_command(test_all)

if __name__ == "__main__":
    cli()
