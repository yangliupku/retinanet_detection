"""
Copyright 2017-2018 yhenon (https://github.com/yhenon/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os

import keras
import keras.preprocessing.image

import tensorflow as tf

from keras_retinanet.models.resnet import ResNet18RetinaNet
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
import keras_retinanet
import pickle
import matplotlib.pyplot as plt


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_model(num_classes, num_features = 64):
    image = keras.layers.Input((None, None, 3))
    return ResNet18RetinaNet(image, num_classes=num_classes, features=num_features)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple training script for object detection from a CSV file.')
    
    parser.add_argument(
        '--batch-size', help='Size of the batches.', default=1, type=int)
    parser.add_argument(
            '--fsize', help='ResNet feature size', default=64, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    train_path = "/home/aragon/workspace/datasets/obj_detection_rdc2/train_linux.csv"
    classes = "/home/aragon/workspace/datasets/obj_detection_rdc2/classes.csv"
    val_path = "/home/aragon/workspace/datasets/obj_detection_rdc2/val_linux.csv"

    # create image data generator objects
    train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
    )

    # create a generator for training data
    train_generator = CSVGenerator(
        csv_data_file=train_path,
        csv_class_file=classes,
        image_data_generator=train_image_data_generator,
        batch_size=args.batch_size
    )

    test_image_data_generator = keras.preprocessing.image.ImageDataGenerator()

    # create a generator for testing data
    test_generator = CSVGenerator(
        csv_data_file=val_path,
        csv_class_file=classes,
        image_data_generator=test_image_data_generator,
        batch_size=args.batch_size
    )

    num_classes = train_generator.num_classes()

    # create the model
    print('Creating model, this may take a second...')
    model = create_model(num_classes=num_classes, num_features=args.fsize)

    # compile model (note: set loss to None since loss is added inside layer)
    model.compile(
        loss={
            'regression'    : keras_retinanet.losses.smooth_l1(),
            'classification': keras_retinanet.losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=2e-5, clipnorm=1e-4)
        # optimizer=keras.optimizers.RMSprop(lr=1e-5)
    )

    # print model summary
    print(model.summary())

    model_name = "resnet_f{}_s8_rdc2".format(args.fsize)
    model_dir = os.path.join("snapshots", model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    checkpoint_fname = os.path.join(model_dir, "{}_best.h5".format(model_name))
    final_save_fname = os.path.join(model_dir, "{}_final.h5".format(model_name))

    # start training
    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.size() // (args.batch_size),
        epochs=100,
        verbose=1,
        max_queue_size=20,
        validation_data=test_generator,
        validation_steps=test_generator.size() // (args.batch_size),
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                checkpoint_fname, monitor='val_loss', verbose=1, save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0),
            keras.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
        ],
    )
    with open(os.path.join(model_dir, "train.p"), "wb") as f:
        pickle.dump(history.history, f)

    model.save(final_save_fname)

    plt.plot(history.history["loss"], label='train_loss')
    plt.plot(history.history["val_loss"], label='val_loss')
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.grid(1)
    plt.savefig(os.path.join(model_dir, "training.png"))
