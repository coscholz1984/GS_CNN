#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a script to train a deeply convolutional network to recognize a type of
pattern from single image snapshots. Dynamics patterns are therefore not taken
explicitly into account.

A dataset was created using GSMain.py that creates various snapshots of patterns
with corresponding labels. This routine now has to devide this dataset into
training, verification and test sets and train the neural network to recognize
each pattern type from the snapshots.

__author__     = "Christian Scholz and Sandy Scholz"
__copyright__  = "Copyright 2021, authors"
__credits__    = ["Christian Scholz", "Sandy Scholz"]
__license__    = "GPLv3"
__version__    = "1.0"
__maintainer__ = "Christian Scholz and Sandy Scholz"
__email__      = "coscholz1984@gmail.com"
__status__     = "Development"
__summary__    = "Here we train the 2D convolutional neural network"

Usage
--------

python CNN_Train_2D.py

"""

# %% Dependencies
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import GStools as gst
import numpy as np
import pickle
import time

# %% Set global constants and options
NORMALIZE_DATA = False
Nepochs = 600
Bsize = 128

FILEPREFIX = "Dataset_2D_"
EXPORT_PATH_KERAS = "model_CNN_2D"

# %% Load datasets
dataset_train = np.array(pickle.load(open(FILEPREFIX + "train.p", "rb")))
dataset_val = np.array(pickle.load(open(FILEPREFIX + "val.p", "rb")))
dataset_test = np.array(pickle.load(open(FILEPREFIX + "test.p", "rb")))
labels_train = pickle.load(open(FILEPREFIX + "train_label.p", "rb"))
labels_val = pickle.load(open(FILEPREFIX + "val_label.p", "rb"))
labels_test = pickle.load(open(FILEPREFIX + "test_label.p", "rb"))

# Convert labels to integers and correct range
labels_train_int = gst.intLabels(labels_train)
labels_val_int = gst.intLabels(labels_val)
labels_test_int = gst.intLabels(labels_test)

# Add missing second dimension
labels_train_int = labels_train_int.reshape(-1, 1)

if NORMALIZE_DATA:
    dataset_train_norm = tf.keras.utils.normalize(dataset_train, axis=-1, order=2)
    dataset_val_norm = tf.keras.utils.normalize(dataset_val, axis=-1, order=2)
    dataset_test_norm = tf.keras.utils.normalize(dataset_test, axis=-1, order=2)
else:
    dataset_train_norm = dataset_train
    dataset_val_norm = dataset_val
    dataset_test_norm = dataset_test

# Here we generate a data generator that randomly rotates, zooms, shears, flips and shifts images before training
datagen_train = ImageDataGenerator(
    rotation_range=360,
    width_shift_range=1.0,
    height_shift_range=1.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=gst.randscale,
    fill_mode="wrap",
)

datagen_train.fit(dataset_train_norm)

# %% Define and compile neural network architecture
model = gst.define_2D_cnn_architecture((128, 128, 3), labels_train_int.max() + 1)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["sparse_categorical_accuracy"],
)
# %% Train network
STEPS_PER_EPOCH = len(dataset_train_norm) / Bsize
SAVE_PERIOD = 30

tic = time.perf_counter()
history_training = model.fit(
    datagen_train.flow(dataset_train_norm, labels_train_int, batch_size=Bsize),
    steps_per_epoch=len(dataset_train_norm) / Bsize,
    epochs=Nepochs,
    validation_data=(dataset_val_norm, labels_val_int),
)
print(
    "Training finished after "
    + str(time.perf_counter() - tic)
    + " seconds"
    + "\n"
)

# %% Evaluate trained network accuracy and loss on test data
print("Evaluate on test data")
results = model.evaluate(dataset_test_norm, labels_test_int, batch_size=Bsize)
print("test loss, test acc:", results)

# %% Save model incl. weights and training history
print("Saving model to " + EXPORT_PATH_KERAS)
model.save(EXPORT_PATH_KERAS)

print("Saving training history to " + "./" + EXPORT_PATH_KERAS + "/trainHistoryDict")
with open("./" + EXPORT_PATH_KERAS + "/trainHistoryDict", "wb") as file_target:
    pickle.dump(history_training.history, file_target)
