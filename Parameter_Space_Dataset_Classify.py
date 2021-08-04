#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Here we plot the predictions of classes made by the neural network across the parameter-space

__author__     = "Christian Scholz and Sandy Scholz"
__copyright__  = "Copyright 2021, authors"
__credits__    = ["Christian Scholz", "Sandy Scholz"]
__license__    = "GPLv3"
__version__    = "1.0"
__maintainer__ = "Christian Scholz and Sandy Scholz"
__email__      = "coscholz1984@gmail.com"
__status__     = "Development"
__summary__    = "Plot the parameter space (k,f) of the Gray-Scott model with class predictions made by the first CNN"

Usage
--------

python Parameter_Space_Dataset_Classify.py [model path]

"""

# %% Dependencies
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import tensorflow as tf
import GStools as gst
import numpy as np
import pickle
import sys

# %% check or set default input arguments, here which model is used for
# visualisation
#
if len(sys.argv) == 1:
    input_path = "model_CNN_2D"
    print(
        "No input given, plotting for 'model_CNN_2D' case, for other models, "
        + "specify input path"
    )
elif len(sys.argv) == 2:
    if sys.argv[1] == "model_CNN_2D" or sys.argv[1] == "model_CNN_2D_2nd":
        input_path = sys.argv[1]
        print("Model " + input_path + " used")
    else:
        input_path = "model_CNN_2D"
        print("Unknown model path given, plotting for 'model_CNN_2D'.")
else:
    input_path = sys.argv[1]
    raise NameError("Too many inputs given, plotting for 'model_CNN_2D'.")


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

labels_all = model.predict(np_res_all)
lmx = labels_all.argmax(axis=1)

x_params = np.array(dataset_parameters)
y_labels = np.array(lmx)

# %% Convert result to image and plot in parameter space
idx1 = np.round((x_params[:, 0] - 0.02375) / (0.0025 / 4)) + 1
idx1 = idx1.astype(int)
idx2 = np.round((x_params[:, 1] - 0.0015) / (0.005 / 4)) + 1
idx2 = idx2.astype(int)

ticks = gst.get_greek_labels(model.output_shape[1])
if model.output_shape[1] == 15:
    color = colors_15
    cmap = cmap_15
elif model.output_shape[1] == 16:
    color = colors_16
    cmap = cmap_16

img = np.empty(
    (
        95,
        87,
    )
)
img[:] = np.nan

img[idx2 - 1, idx1 - 1] = y_labels
fig, ax = plt.subplots(figsize=(6, 8))
im = ax.imshow(
    img,
    interpolation=None,
    cmap=cmap,
    origin="lower",
    extent=[0.02375, 0.0775, 0.0015, 0.12],
)
for n in range(model.output_shape[1]):
    plt.plot(0, 0, "s", color=color[n], label=ticks[n])

plt.xlabel("k")
plt.ylabel("f")
plt.xlim(0.02, 0.08)
plt.ylim(0, 0.12)
plt.gca().set_aspect("equal", adjustable="box")
plt.legend(loc="upper right", bbox_to_anchor=(1.5, 1.025))
plt.show()
