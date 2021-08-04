#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script contains helper functions to generate the patterns. 

__author__     = "Christian Scholz and Sandy Scholz"
__copyright__  = "Copyright 2021, authors"
__credits__    = ["Christian Scholz", "Sandy Scholz"]
__license__    = "GPLv3"
__version__    = "1.0"
__maintainer__ = "Christian Scholz and Sandy Scholz"
__email__      = "coscholz1984@gmail.com"
__status__     = "Development"
__summary__    = "Generate training and validation data for neural network, 
                  with parameters obtained in literature"
"""

__all__ = [
    "appendHist",
    "check",
    "chessboarddistance",
    "convert_labels_to_integer",
    "define_2D_cnn_architecture",
    "define_3D_cnn_architecture",
    "get_dataset_parameter",
    "get_greek_labels",
    "intLabels",
    "initPearsonNoise",
    "laplace",
    "nLI",
    "randscale",
    "runSimulation",
]

# %% Dependencies
from scipy import ndimage
import tensorflow as tf
import pandas as pd
import numpy as np

# %%
def appendHist(h1, h2):
    if h1 == {}:
        return h2
    else:
        dest = {}
        for key, value in h1.items():
            dest[key] = value + h2[key]
        return dest


# %%
def chessboarddistance(x1, y1, x2, y2):
    """
    Chessboard distance, i.e. maximum single coordinate difference of two
    points in 2D. Used to generate initial state.

    Parameters
    ----------
    x1 : float
        x-coordinate of point 1
    y1 : float
        y-coordinate of point 1
    x2 : float
        x-coordinate of point 2
    y2 : float
        y-coordinate of point 2

    Returns
    -------
    dist : float
        chess board distance

    """
    dist = np.max((np.abs(x1 - x2), np.abs(y1 - y2)))
    return dist


# %%
def check(value, min_value=0.2, max_value=0.4):
    """
    Helper funtion to return True if value is between min_value and max_value.
    Used to generate initial state.

    Parameters
    ----------
    value : float

    Returns
    -------
    bool

    """
    if min_value <= value <= max_value:
        return True
    return False


# %%
def convert_labels_to_integer(labels, categories):
    """
    Convert character labels for classes into integer

    Parameters
    ----------
    labels : 1D array
        Vector of labels in char or string format (not case sensitive).
    categories : list
        A list of categories that occur in labels given in the order in which
        they should be numbered (not case sensitive).

    Returns
    -------
    labels_integer : 1D array
        Return all labels as integers as defined by the categories list.

    """
    labels_integer = np.zeros(
        [
            len(labels),
        ],
        dtype="uint8",
    )
    for ind, label in enumerate(labels):
        for index_category, label_category in enumerate(categories):
            if label.lower() == label_category.lower():
                labels_integer[ind] = index_category
    return labels_integer


# %%
def define_2D_cnn_architecture(shape_input, num_categories):
    """
    Define the model architecture for 2D convolutional model

    Parameters
    ----------
    shape_input : tupel
        Shape of input data, i.e. (128, 128, 3).
    num_categories : integer
        Number of classes, e.g. 15.

    Returns
    -------
    model : keras model
        Neural network model.

    """
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=shape_input))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation="relu"))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.GlobalMaxPooling2D())
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.15))
    model.add(tf.keras.layers.Dense(num_categories))
    return model


# %%
def define_3D_cnn_architecture(shape_input, num_categories):
    """
    Define the model architecture for 3D convolutional model

    Parameters
    ----------
    shape_input : tupel
        Shape of input data, i.e. (128, 128, 10, 1).
    num_categories : integer
        Number of classes, e.g. 15.

    Returns
    -------
    model : keras model
        Neural network model.

    """
    sample_shape = (128, 128, 10, 1)
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(128, 128, 10, 1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(
        tf.keras.layers.Conv3D(
            32,
            kernel_size=(3, 3, 2),
            strides=2,
            activation="relu",
            input_shape=(sample_shape),
        )
    )
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 1), activation="relu"))
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1)))
    model.add(tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 1), activation="relu"))
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1)))
    model.add(tf.keras.layers.GlobalMaxPooling3D())
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.10))
    model.add(tf.keras.layers.Dense(num_categories, activation="softmax"))
    return model


# %%
def get_greek_labels(num_categories):
    """
    helper function to generate greek labels for plotting

    Returns
    -------
    ticks : list
        List of greek labels for plotting

    """
    if num_categories == 15:
        ticks = [
            r"$\alpha$",
            r"$\beta$",
            r"$\gamma$",
            r"$\delta$",
            r"$\epsilon$",
            r"$\zeta$",
            r"$\eta$",
            r"$\theta$",
            r"$\iota$",
            r"$\kappa$",
            r"$\lambda$",
            r"$\mu$",
            r"$\nu$",
            "l",
            "h",
        ]
    elif num_categories == 16:
        ticks = [
            r"$\alpha$",
            r"$\beta$",
            r"$\gamma$",
            r"$\delta$",
            r"$\epsilon$",
            r"$\zeta$",
            r"$\eta$",
            r"$\theta$",
            r"$\iota$",
            r"$\kappa$",
            r"$\lambda$",
            r"$\mu$",
            r"$\nu$",
            r"$\pi$",
            "l",
            "h",
        ]
    return ticks


# %%
def get_dataset_parameter(data_type):
    """
    get f and k parameters and according class

    Parameters
    ----------
    data_type : str
        desicion string on which data set to be generated:
            train, validation and test are the allowed inputs.

    Returns
    -------
     parameter_vector : list
        list of f, k
    vPearsonLabels : list
        list of related class name
    """
    if (
        (data_type != "test")
        and (data_type != "train")
        and (data_type != "val")
        and (data_type != "train2")
        and (data_type != "val2")
    ):
        raise NameError("Test type not know, cannot return valid parameters.")

    parameter_vector = []
    label_vector = []

    if data_type == "train" or data_type == "val":
        parameter_list = [
            ["alpha", 0.05, 0.015, 50],
            ["beta", 0.05, 0.025, 50],
            ["gamma", 0.055, 0.025, 50],
            ["delta", 0.055, 0.03, 50],
            ["epsilon", 0.055, 0.015, 50],
            ["zeta", 0.06, 0.025, 50],
            ["eta", 0.06, 0.03, 50],
            ["theta", 0.0565, 0.03, 50],
            ["iota", 0.06, 0.05, 50],
            ["kappa", 0.063, 0.046, 50],
            ["lambda", 0.065, 0.04, 50],
            ["mu", 0.065, 0.05, 50],
            ["nu", 0.0675, 0.05, 50],
            ["l", 0.055, 0.035, 50],
            ["h", 0.05, 0.005, 50],
        ]
    elif data_type == "test":
        parameter_list = [
            ["alpha", 0.045, 0.008, 16],
            ["alpha", 0.05, 0.02, 17],
            ["alpha", 0.05, 0.015, 17],
            ["beta", 0.04, 0.015, 25],
            ["beta", 0.05, 0.025, 25],
            ["gamma", 0.055, 0.025, 50],
            ["delta", 0.055, 0.03, 50],
            ["epsilon", 0.055, 0.02, 16],
            ["epsilon", 0.055, 0.015, 17],
            ["epsilon", 0.059, 0.02, 17],
            ["zeta", 0.06, 0.025, 50],
            ["eta", 0.06, 0.03, 25],
            ["eta", 0.0625, 0.035, 25],
            ["theta", 0.059, 0.04, 16],
            ["theta", 0.06, 0.045, 17],
            ["theta", 0.0565, 0.03, 17],
            ["iota", 0.06, 0.05, 50],
            ["kappa", 0.063, 0.046, 10],
            ["kappa", 0.0625, 0.0625, 10],
            ["kappa", 0.0625, 0.055, 10],
            ["kappa", 0.0625, 0.05, 10],
            ["kappa", 0.0625, 0.045, 10],
            ["lambda", 0.065, 0.04, 25],
            ["lambda", 0.065, 0.035, 25],
            ["mu", 0.065, 0.055, 25],
            ["mu", 0.065, 0.05, 25],
            ["nu", 0.065, 0.03, 10],
            ["nu", 0.0675, 0.05, 10],
            ["nu", 0.0675, 0.045, 10],
            ["nu", 0.0675, 0.04, 10],
            ["nu", 0.0675, 0.035, 10],
            ["l", 0.04, 0.02, 6],
            ["l", 0.045, 0.024, 5],
            ["l", 0.05, 0.03, 5],
            ["l", 0.055, 0.035, 5],
            ["l", 0.0575, 0.045, 5],
            ["l", 0.0585, 0.05, 6],
            ["l", 0.06, 0.06, 6],
            ["l", 0.06, 0.055, 6],
            ["l", 0.06, 0.0525, 6],
            ["h", 0.04, 0.01, 5],
            ["h", 0.04, 0.005, 5],
            ["h", 0.045, 0.012, 5],
            ["h", 0.045, 0.004, 5],
            ["h", 0.05, 0.005, 5],
            ["h", 0.055, 0.01, 5],
            ["h", 0.059, 0.015, 5],
            ["h", 0.065, 0.025, 5],
            ["h", 0.065, 0.02, 5],
            ["h", 0.0675, 0.055, 5],
        ]
    elif data_type == "train2" or data_type == "val2":
        parameter_list = [
            ["alpha", 0.05, 0.015, 50],
            ["alpha", 0.0425, 0.00525, 5],
            ["alpha", 0.045, 0.009, 5],
            ["alpha", 0.046875, 0.009, 5],
            ["alpha", 0.046875, 0.0165, 5],
            ["alpha", 0.0475, 0.01275, 5],
            ["alpha", 0.048125, 0.0165, 5],
            ["alpha", 0.049375, 0.014, 5],
            ["alpha", 0.05, 0.01775, 5],
            ["alpha", 0.050625, 0.02025, 5],
            ["alpha", 0.04375, 0.0065, 5],
            ["beta", 0.05, 0.025, 50],
            ["beta", 0.040625, 0.014, 50],
            ["beta", 0.0425, 0.01525, 5],
            ["beta", 0.04375, 0.01775, 5],
            ["beta", 0.046875, 0.0215, 5],
            ["beta", 0.048125, 0.02275, 5],
            ["beta", 0.050625, 0.02525, 5],
            ["beta", 0.052500, 0.02775, 5],
            ["beta", 0.02875, 0.00775, 5],
            ["beta", 0.035, 0.0115, 5],
            ["beta", 0.029375, 0.00775, 5],
            ["gamma", 0.055, 0.025, 50],
            ["gamma", 0.05125, 0.0215, 5],
            ["gamma", 0.051875, 0.02275, 5],
            ["gamma", 0.0525, 0.02275, 5],
            ["gamma", 0.053125, 0.02275, 5],
            ["gamma", 0.05375, 0.02275, 5],
            ["gamma", 0.05375, 0.024, 5],
            ["gamma", 0.054375, 0.02525, 5],
            ["gamma", 0.055, 0.024, 5],
            ["gamma", 0.05625, 0.02525, 5],
            ["gamma", 0.055625, 0.02525, 5],
            ["delta", 0.055, 0.03, 50],
            ["delta", 0.053125, 0.0265, 5],
            ["delta", 0.054375, 0.029, 5],
            ["delta", 0.055, 0.03025, 5],
            ["delta", 0.05625, 0.03275, 5],
            ["delta", 0.05625, 0.034, 5],
            ["delta", 0.056875, 0.03525, 5],
            ["delta", 0.0575, 0.0365, 5],
            ["delta", 0.05875, 0.0415, 5],
            ["delta", 0.058125, 0.039, 5],
            ["delta", 0.059375, 0.044, 5],
            ["epsilon", 0.055, 0.015, 50],
            ["epsilon", 0.053125, 0.014, 5],
            ["epsilon", 0.0525, 0.01275, 5],
            ["epsilon", 0.054375, 0.01275, 5],
            ["epsilon", 0.055, 0.0165, 5],
            ["epsilon", 0.055625, 0.01775, 5],
            ["epsilon", 0.058125, 0.01775, 5],
            ["epsilon", 0.05875, 0.02025, 5],
            ["epsilon", 0.055625, 0.0165, 5],
            ["epsilon", 0.06, 0.02025, 5],
            ["epsilon", 0.055625, 0.019, 5],
            ["zeta", 0.06, 0.025, 50],
            ["zeta", 0.05625, 0.0215, 5],
            ["zeta", 0.0575, 0.02275, 5],
            ["zeta", 0.058125, 0.02525, 5],
            ["zeta", 0.05875, 0.0265, 5],
            ["zeta", 0.059375, 0.024, 5],
            ["zeta", 0.059375, 0.02525, 5],
            ["zeta", 0.06, 0.02275, 5],
            ["zeta", 0.06, 0.024, 5],
            ["zeta", 0.06, 0.02525, 5],
            ["zeta", 0.060625, 0.02275, 5],
            ["eta", 0.06, 0.03, 50],
            ["eta", 0.058125, 0.02775, 5],
            ["eta", 0.059375, 0.029, 5],
            ["eta", 0.06, 0.03025, 5],
            ["eta", 0.06125, 0.03275, 5],
            ["eta", 0.06125, 0.034, 5],
            ["eta", 0.0625, 0.03525, 5],
            ["eta", 0.0625, 0.03775, 5],
            ["eta", 0.064375, 0.0415, 5],
            ["eta", 0.06375, 0.039, 5],
            ["eta", 0.063125, 0.03775, 5],
            ["theta", 0.0565, 0.03, 50],
            ["theta", 0.05625, 0.029, 5],
            ["theta", 0.056875, 0.03025, 5],
            ["theta", 0.05875, 0.03275, 5],
            ["theta", 0.06, 0.0365, 5],
            ["theta", 0.059375, 0.03775, 5],
            ["theta", 0.060625, 0.03775, 5],
            ["theta", 0.06125, 0.04275, 5],
            ["theta", 0.061875, 0.044, 5],
            ["theta", 0.061875, 0.05025, 5],
            ["theta", 0.06125, 0.064, 5],
            ["iota", 0.06, 0.05, 50],
            ["iota", 0.0575, 0.03775, 5],
            ["iota", 0.05875, 0.04275, 5],
            ["iota", 0.06, 0.049, 5],
            ["iota", 0.06, 0.05025, 5],
            ["iota", 0.060625, 0.05525, 5],
            ["iota", 0.060625, 0.0565, 5],
            ["iota", 0.060625, 0.05775, 5],
            ["iota", 0.060625, 0.059, 5],
            ["iota", 0.060625, 0.064, 5],
            ["iota", 0.059375, 0.04525, 5],
            ["kappa", 0.063, 0.046, 50],
            ["kappa", 0.061875, 0.04275, 5],
            ["kappa", 0.0625, 0.049, 5],
            ["kappa", 0.0625, 0.0565, 5],
            ["kappa", 0.0625, 0.06025, 5],
            ["kappa", 0.063125, 0.05525, 5],
            ["kappa", 0.0625, 0.06525, 5],
            ["kappa", 0.061875, 0.064, 5],
            ["kappa", 0.063125, 0.05775, 5],
            ["kappa", 0.0625, 0.05525, 5],
            ["kappa", 0.061875, 0.0665, 5],
            ["lambda", 0.065, 0.04, 50],
            ["lambda", 0.06125, 0.02275, 5],
            ["lambda", 0.0625, 0.03025, 5],
            ["lambda", 0.06375, 0.03025, 5],
            ["lambda", 0.065, 0.039, 5],
            ["lambda", 0.065625, 0.04525, 5],
            ["lambda", 0.063750, 0.026500, 5],
            ["lambda", 0.065625, 0.035250, 5],
            ["lambda", 0.066250, 0.046500, 5],
            ["lambda", 0.066250, 0.044000, 5],
            ["lambda", 0.066250, 0.040250, 5],
            ["mu", 0.065, 0.05, 50],
            ["mu", 0.063750, 0.056500, 5],
            ["mu", 0.06375, 0.059, 5],
            ["mu", 0.064375, 0.05525, 5],
            ["mu", 0.064375, 0.05025, 5],
            ["mu", 0.065, 0.0515, 5],
            ["mu", 0.065625, 0.05025, 5],
            ["mu", 0.065, 0.0565, 5],
            ["mu", 0.065625, 0.054, 5],
            ["mu", 0.065625, 0.0515, 5],
            ["mu", 0.065625, 0.0540, 5],
            ["vu", 0.0675, 0.05, 50],
            ["vu", 0.065000, 0.029, 5],
            ["vu", 0.065625, 0.0315, 5],
            ["vu", 0.066250, 0.0515, 5],
            ["vu", 0.066875, 0.03275, 5],
            ["vu", 0.067500, 0.04025, 5],
            ["vu", 0.067500, 0.03025, 5],
            ["vu", 0.068125, 0.0465, 5],
            ["vu", 0.068750, 0.03775, 5],
            ["vu", 0.066875, 0.044, 5],
            ["vu", 0.068750, 0.04025, 5],
            ["pi", 0.060625, 0.069, 25],
            ["pi", 0.060625, 0.06775, 25],
            ["pi", 0.060625, 0.0665, 25],
            ["pi", 0.060625, 0.07025, 25],
            ["l", 0.055, 0.035, 100],
            ["h", 0.05, 0.005, 100],
        ]
    nType = range(len(parameter_list))
    for iType in nType:
        k = parameter_list[iType][1]
        f = parameter_list[iType][2]
        for iRun in range(parameter_list[iType][3]):
            parameter_vector.append([k, f])
            label_vector.append(parameter_list[iType][0])

    return parameter_vector, label_vector


# %% calculate and return saliency map for an input image from a certain class
# given a certain model
def get_saliency_map(model, image_in, class_idx):
    image = tf.convert_to_tensor(image_in)
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)

        loss = predictions[:, class_idx]

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, image)

    # take maximum across channels
    gradient = tf.reduce_max(gradient, axis=-1)

    # convert to numpy
    gradient = gradient.numpy()

    # normaliz between 0 and 1
    min_val, max_val = np.min(gradient), np.max(gradient)
    smap = (gradient - min_val) / (max_val - min_val + tf.keras.backend.epsilon())

    return smap


# %% define some helpger functions
def intLabels(label_char):
    """
    Convert labels to integer numbers based on default categories

    Parameters
    ----------
    labl : array of strings/char-arrays
        Array that contains string class labels for each datapoint.

    Returns
    -------
    labl_int : integer array
        Array with numeric integer class labels.

    """
    label_int = convert_labels_to_integer(label_char, pd.unique(label_char))

    return label_int


# %% Initialize the pattern for pearson patterns with added noise
def initPearsonNoise(width, xshift, yshift, noise):
    """
    Helper function to calculate initial state with added noise

    Parameters
    ----------
    width : integer
        System width.
    xshift : float
        Shift of 2nd perturbation in x direction. Value is relative to width.
    yshift : float
        Shift of 2nd perturbation in y direction. Value is relative to width.
    noise : float
        Level of noise to be added to the initial conditions.

    Returns
    -------
    t : float
        Initial time (0).
    U : 2D array
        Initial state for U.
    V : 2D array
        Initial state for V.

    """

    t = 0
    U = np.ones([width, width])
    V = np.zeros([width, width])

    U[width // 2 - 9 : width // 2 + 11, width // 2 - 9 : width // 2 + 11] = 0.5
    V[width // 2 - 9 : width // 2 + 11, width // 2 - 9 : width // 2 + 11] = 0.25

    U[
        int(np.round(width * xshift)) - 9 : int(np.round(width * xshift)) + 11,
        int(np.round(width * yshift)) - 9 : int(np.round(width * yshift)) + 11,
    ] = 0.5
    V[
        int(np.round(width * xshift)) - 9 : int(np.round(width * xshift)) + 11,
        int(np.round(width * yshift)) - 9 : int(np.round(width * yshift)) + 11,
    ] = 0.25

    U[
        int(np.round(width * xshift)) - 9 + 6 : int(np.round(width * xshift)) + 11 + 6,
        int(np.round(width * yshift)) - 9 + 9 : int(np.round(width * yshift)) + 11 + 9,
    ] = 0.99
    V[
        int(np.round(width * xshift)) - 9 + 6 : int(np.round(width * xshift)) + 11 + 6,
        int(np.round(width * yshift)) - 9 + 9 : int(np.round(width * yshift)) + 11 + 9,
    ] = 0.01

    U = U * (1 + noise * np.random.randn(width, width))
    V = V * (1 + noise * np.random.randn(width, width))

    return t, U, V


# %% laplace calculates the laplace operator with nearest and next nearest neighbor weighting and periodic boundary conditions
#    for reference see http://www.physics.mcgill.ca/~provatas/papers/Phase_Field_Methods_text.pdf
def laplace(M):
    """
    Calculate central space discrete laplace of an input matrix

    Parameters
    ----------
    M : 2D array
        A 2D matrix on which the laplace is to be calculated.

    Returns
    -------
    2D array
        The laplace of the input matrix.

    """
    return (
        -3 * M
        + 0.5
        * (
            np.roll(M, (1, 0), axis=(0, 1))
            + np.roll(M, (0, 1), axis=(0, 1))
            + np.roll(M, (-1, 0), axis=(0, 1))
            + np.roll(M, (0, -1), axis=(0, 1))
        )
        + 0.25
        * (
            np.roll(M, (1, 1), axis=(0, 1))
            + np.roll(M, (1, -1), axis=(0, 1))
            + np.roll(M, (-1, 1), axis=(0, 1))
            + np.roll(M, (-1, -1), axis=(0, 1))
        )
    )


# %% Calculate non-linear term
def nLI(U, V):
    """
    Helper function to calculate the nonlinear interaction term of the Gray-Scott Model

    Parameters
    ----------
    U : 2D array
        Discreate concentration field of U.
    V : 2D array
        Discrete concentration field of V.

    Returns
    -------
    nonlinearInteraction : TYPE
        DESCRIPTION.

    """
    nonlinearInteraction = U * V * V
    return nonlinearInteraction


# %%
def randscale(img):
    """
    Rescale input img randomly by +- 5%

    Parameters
    ----------
    img : array
        Usually input image.

    Returns
    -------
    same as img
        Rescaled image.

    """
    return img * (1 + 0.1 * (np.random.rand() - 0.5))


# %%
def runSimulation(
    seed_local,
    diff_u,
    diff_v,
    rate_k,
    rate_f,
    sys_width,
    timestep,
    stoptime,
    savetime,
    save_timestep,
):
    """
    Run a single iteration of the simulation, solving the Gray-Scott model
    using forward-time central-space discretization

    Parameters
    ----------
    seed_local : integer
        Local seed for reproducible random number generation.
    diff_u : float
        Diffusion coefficient of reactant u (a).
    diff_v : float
        Diffusion coefficient of reactant v (b).
    rate_k : float
        Reaction rate k.
    rate_f : float
        Reaction rate f.
    sys_width : integer
        Length of quadratic grid. Total size is sys_width x sys_width
    timestep : float
        time-step for numerical integration.
    stoptime : float
        Total simulation time.
    savetime : float
        Time from which patterns will be saved to dataset. Useually a multiple
        of save_timestep.
    save_timestep : float
        Timestep after which a pattern will be added to the output dataset.

    Returns
    -------
    dataset_to_predict : Numpy array [sys_width, sys_width, N]
        Output of the simulation. N is equal to the number of saved patterns
        and is equal to np.ceil(savetime/save_timestep+1).

    """
    # setup initial values and initial state
    np.random.seed(seed_local)
    savetime_local = savetime
    dataset_to_predict = np.ones(
        [sys_width, sys_width, int(np.ceil(savetime / save_timestep + 1))]
    )
    iSave = 0
    rate_sum = rate_k + rate_f

    shift_x = 0.5
    shift_y = 0.5
    while not check(chessboarddistance(shift_x, shift_y, 0.5, 0.5)):
        shift_x = np.random.rand() * 0.8 + 0.1  # 0.3
        shift_y = np.random.rand() * 0.8 + 0.1  # 0.2

    # construct initial state made of two perturbations of a homogeneous
    # background at random distance within a certain range(see check).
    timer_local, U_local, V_local = initPearsonNoise(sys_width, shift_x, shift_y, 0)
    # random rotation
    angle = np.random.randint(0, 89)
    rot_A = ndimage.rotate(U_local, angle, mode="wrap", prefilter=False, reshape=False)
    rot_B = ndimage.rotate(V_local, angle, mode="wrap", prefilter=False, reshape=False)

    # random shift in x and y
    x_rand = int(np.random.rand() * sys_width)
    y_rand = int(np.random.rand() * sys_width)
    U_local = np.roll(rot_A, x_rand, axis=1)
    V_local = np.roll(rot_B, x_rand, axis=1)
    U_local = np.roll(rot_A, y_rand, axis=0)
    V_local = np.roll(rot_B, y_rand, axis=0)

    # Main integration loop
    while timer_local < stoptime:
        nonlinearInteraction = nLI(U_local, V_local)
        U_local_new = (
            U_local
            + (
                diff_u * laplace(U_local)
                - nonlinearInteraction
                + rate_f * (1 - U_local)
            )
            * timestep
        )
        V_local_new = (
            V_local
            + (diff_v * laplace(V_local) + nonlinearInteraction - rate_sum * V_local)
            * timestep
        )
        U_local = U_local_new
        V_local = V_local_new
        timer_local = timer_local + timestep

        if timer_local >= stoptime - savetime_local:
            dataset_to_predict[:, :, iSave] = U_local
            iSave = iSave + 1
            savetime_local = savetime_local - save_timestep

    return dataset_to_predict
