#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a script to evaluate the performance of a trained keras model.

__author__     = "Christian Scholz and Sandy Scholz"
__copyright__  = "Copyright 2021, authors"
__credits__    = ["Christian Scholz", "Sandy Scholz"]
__license__    = "GPLv3"
__version__    = "1.0"
__maintainer__ = "Christian Scholz and Sandy Scholz"
__email__      = "coscholz1984@gmail.com"
__status__     = "Development"
__summary__    = "Here we evaluate the performance of a convolutional neural 
                  network"

Usage
--------

python CNN_Evaluate.py [model path]

"""

# %% Dependencies
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import GStools as gst
import numpy as np
import pickle
import sys

# %% Global constants
NORMALIZE_DATA = False

# %% Font settings
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 16  # set font size for axes titles
plt.rcParams["legend.fontsize"] = 16

# %% check or set default input arguments, here which model is used for
# visualisation
#
if len(sys.argv) == 1:
    IMPORT_PATH_KERAS = "model_CNN_2D"
    print(
        "No input given, plotting for 'model_CNN_2D' case, for other models, "
        + "specify input path"
    )
elif len(sys.argv) == 2:
    if (
        sys.argv[1] == "model_CNN_2D"
        or sys.argv[1] == "model_CNN_2D_2nd"
        or sys.argv[1] == "model_CNN_3D"
    ):
        IMPORT_PATH_KERAS = sys.argv[1]
        print("Model " + IMPORT_PATH_KERAS + " used")
    else:
        IMPORT_PATH_KERAS = "model_CNN_2D"
        print("Unknown model path given, plotting for 'model_CNN_2D'.")
else:
    IMPORT_PATH_KERAS = sys.argv[1]
    raise NameError("Too many inputs given, plotting for 'model_CNN_2D'.")

# %% Load model and history
model = tf.keras.models.load_model(IMPORT_PATH_KERAS)
history = pickle.load(open("./" + IMPORT_PATH_KERAS + "/trainHistoryDict", "rb"))

# %% Load datasets
FILEPREFIX = "Dataset_2D_"
if IMPORT_PATH_KERAS == "model_CNN_3D":
    FILEPREFIX = "Dataset_3D_"
if IMPORT_PATH_KERAS == "model_CNN_2D_2nd":
    dataset_train = np.array(pickle.load(open(FILEPREFIX + "train2.p", "rb")))
    dataset_val = np.array(pickle.load(open(FILEPREFIX + "val2.p", "rb")))
    labels_val = pickle.load(open(FILEPREFIX + "val2_label.p", "rb"))
    labels_train = pickle.load(open(FILEPREFIX + "train2_label.p", "rb"))
else:        
    dataset_train = np.array(pickle.load(open(FILEPREFIX + "train.p", "rb")))
    dataset_val = np.array(pickle.load(open(FILEPREFIX + "val.p", "rb")))
    labels_val = pickle.load(open(FILEPREFIX + "val_label.p", "rb"))
    labels_train = pickle.load(open(FILEPREFIX + "train_label.p", "rb"))
dataset_test = np.array(pickle.load(open(FILEPREFIX + "test.p", "rb")))
labels_test = pickle.load(open(FILEPREFIX + "test_label.p", "rb"))

if NORMALIZE_DATA:
    dataset_train_norm = tf.keras.utils.normalize(dataset_train, axis=-1, order=2)
    dataset_val_norm = tf.keras.utils.normalize(dataset_val, axis=-1, order=2)
    dataset_test_norm = tf.keras.utils.normalize(dataset_test, axis=-1, order=2)
else:
    dataset_train_norm = dataset_train
    dataset_val_norm = dataset_val
    dataset_test_norm = dataset_test

# Adding empty channel dimension
if IMPORT_PATH_KERAS == "model_CNN_3D":
    dataset_train_norm = np.expand_dims(dataset_train_norm, axis=4)
    dataset_val_norm = np.expand_dims(dataset_val_norm, axis=4)
    dataset_test_norm = np.expand_dims(dataset_test_norm, axis=4)

# Convert labels to integers and correct range
labels_train_int = gst.intLabels(labels_train)
labels_val_int = gst.intLabels(labels_val)
labels_test_int = gst.intLabels(labels_test)

# Hack label index for l and h patterns [13,14] -> [14,15]
ticks_test = gst.get_greek_labels(labels_test_int.max() + 1) # stored required test labels
if IMPORT_PATH_KERAS == "model_CNN_2D_2nd":
    labels_test_int = np.where(labels_test_int==14, 15, labels_test_int)
    labels_test_int = np.where(labels_test_int==13, 14, labels_test_int)

# %% plot training history
plt.figure()
plt.plot(history["sparse_categorical_accuracy"])
plt.plot(history["val_sparse_categorical_accuracy"])
# plt.title('model accuracy')
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="lower right")
plt.show()

# %% Plot confusion matrix for train, val and test data
# predict classes
labels_train_pred = model.predict(dataset_train_norm)
labels_val_pred = model.predict(dataset_val_norm)
labels_test_pred = model.predict(dataset_test_norm)

# generate confusion matrices
confusion_matrix_train = confusion_matrix(
    labels_train_int, labels_train_pred.argmax(axis=1)
)
confusion_matrix_val = confusion_matrix(labels_val_int, labels_val_pred.argmax(axis=1))
confusion_matrix_test = confusion_matrix(
    labels_test_int, labels_test_pred.argmax(axis=1)
)

# generate figure
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))
axes[0].title.set_text("Training")
axes[1].title.set_text("Validation")
axes[2].title.set_text("Test")
ticks = gst.get_greek_labels(labels_train_int.max() + 1)

# plot all confusion matrices
for iIteration, confusion_matrix_list in enumerate(
    [confusion_matrix_train, confusion_matrix_val, confusion_matrix_test]
):
    cmD_hdl = ConfusionMatrixDisplay(
        confusion_matrix_list,
        display_labels=np.arange(24) + 1,
    )
    cmD_hdl = cmD_hdl.plot(
        include_values=True,
        cmap="Blues",
        ax=axes[iIteration],
        xticks_rotation="horizontal",
    )

# delete all colorbars
fig.delaxes(fig.axes[3])
fig.delaxes(fig.axes[3])
fig.delaxes(fig.axes[3])
for iIteration in range(2):
    axes[iIteration].set_xticklabels(ticks)
    axes[iIteration].set_yticklabels(ticks)
# test data can have different number of classes
axes[2].set_xticklabels(ticks_test)
axes[2].set_yticklabels(ticks_test)    

plt.tight_layout()
plt.show()
