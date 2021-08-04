#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot montage of patterns as classified by the CNN given as input. Clicking on an image returns the corresponding parameters (k,f)

__author__     = "Christian Scholz and Sandy Scholz"
__copyright__  = "Copyright 2021, authors"
__credits__    = ["Christian Scholz", "Sandy Scholz"]
__license__    = "GPLv3"
__version__    = "1.0"
__maintainer__ = "Christian Scholz and Sandy Scholz"
__email__      = "coscholz1984@gmail.com"
__status__     = "Development"
__summary__    = "Plot montage of patterns as classified by the CNN given as input. Clicking on an image returns the corresponding parameters (k,f)"

Usage
--------

python Predict_Class_Montage.py [model path] [Class No.]

"""

# %% Dependencies
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys

# %% check or set default input arguments, here which model is used for
# visualisation
#
if len(sys.argv) == 1:
    input_path = "model_CNN_2D"
    iClass = 0
    print(
        "No input given, using 'model_CNN_2D' case and display class 0, for other models, "
        + "specify input path and class (No. 0 - 14)"
    )
elif len(sys.argv) == 3:
    if sys.argv[1] == "model_CNN_2D" or sys.argv[1] == "model_CNN_2D_2nd":
        input_path = sys.argv[1]
        print("Model " + input_path + " used")
    else:
        input_path = sys.argv[1]
        print("Unknown model path given, using " + input_path + ".")
    if all(char.isdigit() for char in sys.argv[2]):
        if (input_path == "model_CNN_2D") and (int(sys.argv[2]) in range(15)):
            iClass = int(sys.argv[2])
            print("Plot class: " + sys.argv[2])
        elif (input_path == "model_CNN_2D_2nd") and (int(sys.argv[2]) in range(16)):
            iClass = int(sys.argv[2])
            print("Plot class: " + sys.argv[2])
        else:
            iClass = sys.argv[2]
            print("Unknown model, setting class to " + iClass + ".")
    else:
        iClass = 0
        print("Unknown class, displaying class 0 instead.")
else:
    input_path = sys.argv[1]
    iClass = 0
    raise NameError("Too many inputs given, using 'model_CNN_2D'. Display class 0.")


# %% Font and color settings
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 16  # set font size for axes titles
plt.rcParams["legend.fontsize"] = 18
all_colors = plt.cm.tab20((np.arange(20)).astype(int))
colors_16 = np.zeros((16, 4))
colors_16[:14] = all_colors[:14]
colors_16[14:] = all_colors[18:]
colors_15 = np.zeros((15, 4))
colors_15[:13] = colors_16[:13]
colors_15[13:] = colors_16[14:]
colors_15[1] = all_colors[16]
cmap_15 = ListedColormap(colors_15, name="c15")
cmap_16 = ListedColormap(colors_16, name="c16")

# %% Load datasets and parameters
PREFIX = "Dataset_Parameter_Scan_"
# 1
filepat_1 = PREFIX + "1.p"
filepar_1 = PREFIX + "Parameters_1.p"
results_1 = pickle.load(open(filepat_1, "rb"))
dataset_parameters_1 = pickle.load(open(filepar_1, "rb"))

# 2
filepat_2 = PREFIX + "2.p"
filepar_2 = PREFIX + "Parameters_2.p"
results_2 = pickle.load(open(filepat_2, "rb"))
dataset_parameters_2 = pickle.load(open(filepar_2, "rb"))

# 3
filepat_3 = PREFIX + "3.p"
filepar_3 = PREFIX + "Parameters_3.p"
results_3 = pickle.load(open(filepat_3, "rb"))
dataset_parameters_3 = pickle.load(open(filepar_3, "rb"))

dataset_parameters = dataset_parameters_1 + dataset_parameters_2 + dataset_parameters_3
results_all = results_1 + results_2 + results_3
np_res_all = np.array(results_all)

# %% load generated and trained model and predict classes
model = tf.keras.models.load_model(input_path)
if iClass >= model.output_shape[1]:
    raise NameError(
        "Class out of range for given model: Class "
        + str(iClass)
        + " for model output "
        + str(model.output_shape[1] - 1)
        + "."
    )
labels_all = model.predict(np_res_all)
lmx = labels_all.argmax(axis=1)

x_params = np.array(dataset_parameters)
y_labels = np.array(lmx)


# %% draw a grid of images in each class (only heterogeneous patterns)
digit_size = 128
scale = 1.0
n = np.ceil(np.sqrt(np.sum((y_labels == iClass)))).astype(int)
patterns = np_res_all[y_labels == iClass, :, :, :]
vpar = np.array(dataset_parameters)[y_labels == iClass]
vparstring = [["" for c in range(n)] for r in range(n)]
figure = np.zeros((digit_size * n, digit_size * n, 3))
# linearly spaced coordinates corresponding to the 2D plot
# of digit classes in the latent space
grid_x = np.linspace(-scale, scale, n)
grid_y = np.linspace(-scale, scale, n)[::-1]

iCount = 0
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        if iCount < np.sum((y_labels == iClass)):
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
                :,
            ] = patterns[iCount, :, :, :]
            vparstring[i][j] = (
                "("
                + "{:.6f}".format(vpar[iCount][0])
                + ","
                + "{:.6f}".format(vpar[iCount][1])
                + ")"
            )
            iCount = iCount + 1

# plt.figure()
def onclick(event):
    print(
        "%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f, (k,f)=%s"
        % (
            "double" if event.dblclick else "single",
            event.button,
            event.x,
            event.y,
            event.xdata,
            event.ydata,
            vparstring[event.ydata.astype(int) // 128][event.xdata.astype(int) // 128],
        )
    )


fig, ax = plt.subplots()
ax.imshow(figure)
cid = fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()
