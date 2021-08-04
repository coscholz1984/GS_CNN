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
__summary__    = "Here we evaluate the saliency maps of a convolutional neural 
                  network"

Usage
--------

python CNN_Saliency.py

"""

# %% Dependencies
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import tensorflow as tf
import GStools as gst
import numpy as np
import pickle

# %% Global constants
NORMALIZE_DATA = False

# %% Font settings
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 16  # set font size for axes titles
plt.rcParams["legend.fontsize"] = 16

# %% Load model and history
IMPORT_PATH_KERAS = "model_CNN_2D"
model = tf.keras.models.load_model(IMPORT_PATH_KERAS)

# %% Load datasets
FILEPREFIX = "Dataset_2D_"
dataset_train = np.array(pickle.load(open(FILEPREFIX + "train.p", "rb")))

if NORMALIZE_DATA:
    dataset_train_norm = tf.keras.utils.normalize(dataset_train, axis=-1, order=2)
else:
    dataset_train_norm = dataset_train

#%% plot Activation image for all classes
plt.figure()
titles = gst.get_greek_labels(15)
for iClass in range(15):
    smap = gst.get_saliency_map(
        model,
        tf.convert_to_tensor(dataset_train_norm[[iClass * 50 + 25], :, :, :]),
        iClass,
    )
    smap_filtered = ndimage.gaussian_filter(smap[0, :, :], sigma=2)
    axes = plt.subplot(3, 5, iClass + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(dataset_train_norm[iClass * 50 + 25, :, :, :])
    plt.imshow(smap_filtered, alpha=0.6, cmap=plt.get_cmap("viridis"))
    axes.set_title(titles[iClass])

plt.show()
