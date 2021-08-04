#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script is used to generate a dataset from numerical solution of the 
Gray-Scott reaction diffusion system at various parameter pairs (k,f). The 
patterns are saved as 3 consecutive frames at the end of each simulation.

The script can run several simulations in parallel if your CPU can execute 
multiple threads in parallel. Adept NTHREADS to a reasonable number of threads 
that your CPU supports (default 8).

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

Options
--------
The script supports two optinal parameters
python Repro_generate.py [data_type] [dimension]

data_type: 'train', 'val' or 'test'
dimension: '2D' or '3D'

Output
--------
A file with training, validation or test datasets (patterns)
and a file with corresponding class labels

Output filenames are given by:

FILEPREFIX + data_type + ".p"
FILEPREFIX + data_type + "_label.p"

Usage
--------
The following command in the comand promt will generate the training data for 
the 3D use case:

python Repro_generate.py train 3D 

"""
# %% Import libraries

from joblib import Parallel, delayed
import GStools as gst
import pickle
import time
import sys

# %% check or set default input arguments, i.e. which type of dataset
# (training, validation, test) and whether for 2D or 3D convolutional network
if len(sys.argv) == 1:
    data_type = "train"
    data_size = "2D"
    print(
        "No input given, generating train data set for 2D, for other data "
        + "types use 'val', 'test' or 'train' as input. For different data"
        + "convolution target use '3D' as second input"
    )
elif len(sys.argv) == 3:
    if (
        sys.argv[1] == "test"
        or sys.argv[1] == "train"
        or sys.argv[1] == "val"
        or sys.argv[1] == "train2"
        or sys.argv[1] == "val2"
    ):
        data_type = sys.argv[1]
        print("Data type set to " + data_type)
    else:
        data_type = "train"
        print("Data type set to train use 'val' or 'test' for other types")
    if sys.argv[2] == "2D" or sys.argv[2] == "3D":
        data_size = sys.argv[2]
        print("Data convolution target set to " + data_size)
    else:
        data_size = "2D"
        print("Convolution target set to 2D, use '3D' for other case")
else:
    data_type = "train"
    data_size = "2D"
    raise NameError(
        "Wrong input given, generating train data set for 2D, for other data "
        + "types use val, test, or train as input for other data convolution "
        + "target use '3D'"
    )

# %% Define global variables

# Constants
SIZE = 128
TIMESTEP = 0.25
STOPTIME = 15000
if data_size == "2D":
    TIMESTEP_SAVE = (
        30  # save three frames at STOPTIME-2*TIMESTEP_SAVE, TIMESTEP_SAVE, 0
    )
    SAVETIME = 2 * TIMESTEP_SAVE
    FILEPREFIX = "Dataset_2D_"
elif data_size == "3D":
    TIMESTEP_SAVE = 400  # save ten frames at STOPTIME-2*TIMESTEP_SAVE, TIMESTEP_SAVE, 0
    SAVETIME = 9 * TIMESTEP_SAVE
    FILEPREFIX = "Dataset_3D_"
NTHREADS = 8

# %% Parameter of the Gray-Scott model
"""
# Test constants
# Parameters for the Gray Scott model,
# see also https://groups.csail.mit.edu/mac/projects/amorphous/GrayScott/
# classes for the Gray Scott model, 'B' and 'R' define the blue and red
# homogeniuous states, in the paper these are labeled High and Low
# Parameters =[class, k, f, number of petterns to create] correspond mostly to
# to pearsons results, with some minor adaptations due to slightly different
# inital conditions
"""
dataset_parameters, dataset_labels = gst.get_dataset_parameter(data_type)

# Make sure rng for train, val and test data uses different seeds
seed_multiplier = 0
if data_type == "train" or data_type == "train2":
    seed_multiplier = 2  # 2 = Train, 1 = Val, 3 = Test
elif data_type == "val" or data_type == "val2":
    seed_multiplier = 1
elif data_type == "test":
    seed_multiplier = 3

seed_offset = round(seed_multiplier * len(dataset_parameters)) + 1

# wrapper function that can be executed in parallel for loop
def wrap_Simulation(seed_input):
    # Diffusion constants
    DU = 0.2
    DV = 0.1
    savetime_local = SAVETIME
    print(
        "Running simulation "
        + str(seed_input + 1)
        + "/"
        + str(len(dataset_parameters))
        + " (seed: "
        + str(seed_input + seed_offset)
        + ")"
    )
    dataset_output = gst.runSimulation(
        seed_input + seed_offset,
        DU,
        DV,
        dataset_parameters[seed_input][0],
        dataset_parameters[seed_input][1],
        SIZE,
        TIMESTEP,
        STOPTIME,
        savetime_local,
        TIMESTEP_SAVE,
    )
    print(
        "Finished simulation "
        + str(seed_input + 1)
        + "/"
        + str(len(dataset_parameters))
    )
    return dataset_output


# %% run simulations

print(
    "Starting data generation for "
    + data_type
    + " data"
    + "\n"
    + "Attempting parallel execution of up to "
    + str(NTHREADS)
    + " threads"
)

tic = time.perf_counter()
results = Parallel(n_jobs=NTHREADS)(
    delayed(wrap_Simulation)(i) for i in range(len(dataset_parameters))
)

print(
    "Simulations finished after "
    + str(time.perf_counter() - tic)
    + " seconds"
    + "\n"
    + "Saving results to "
    + FILEPREFIX
    + data_type
    + ".p"
)

# %% save results to files

pickle.dump(results, open(FILEPREFIX + data_type + ".p", "wb"))
pickle.dump(dataset_labels, open(FILEPREFIX + data_type + "_label.p", "wb"))
