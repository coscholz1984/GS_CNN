#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Here we generate patterns in a fine scan throughout the parameter space

__author__     = "Christian Scholz and Sandy Scholz"
__copyright__  = "Copyright 2021, authors"
__credits__    = ["Christian Scholz", "Sandy Scholz"]
__license__    = "GPLv3"
__version__    = "1.0"
__maintainer__ = "Christian Scholz and Sandy Scholz"
__email__      = "coscholz1984@gmail.com"
__status__     = "Development"
__summary__    = "Fine-scan the parameter space and generate patterns for 
                  classification"

Usage
--------

python Parameter_Space_Dataset_Generate.py

"""

# %% Dependencies
from joblib import Parallel, delayed
import GStools as gst
import numpy as np
import pickle

# %% Define global variables
SIZE = 128
TIMESTEP = 0.25
STOPTIME = 15000
TIMESTEP_SAVE = 30
SAVETIME = 2 * TIMESTEP_SAVE
PREFIX = "Dataset_Parameter_Scan_"

# 1st block of parameters
dataset_parameters_1 = []
for i_k in np.arange(0.04, 0.0675, 0.0025 / 4):
    for i_f in np.arange(0.004, 0.0625, 0.005 / 4):
        dataset_parameters_1.append([i_k, i_f])

# 2nd block of parameters
dataset_parameters_2 = []
for i_k in np.arange(0.04, 0.0675, 0.0025 / 4):
    for i_f in np.arange(0.06275, 0.12, 0.005 / 4):  # (.004,.0625,.005/4)
        dataset_parameters_2.append([i_k, i_f])

# 3rd block of parameters
dataset_parameters_3 = []
for i_k in np.arange(0.068125, 0.07750000000000001, 0.0025 / 4):
    for i_f in np.arange(0.004, 0.0625, 0.005 / 4):
        dataset_parameters_3.append([i_k, i_f])

for i_k in np.arange(0.0375, 0.045, 0.0025 / 4):
    for i_f in np.arange(0.0015, 0.00900001, 0.005 / 4):
        dataset_parameters_3.append([i_k, i_f])

for i_k in np.arange(0.0375, 0.045, 0.0025 / 4):
    for i_f in np.arange(0.01025, 0.0215, 0.005 / 4):
        dataset_parameters_3.append([i_k, i_f])

for i_k in np.arange(0.02375, 0.036876, 0.0025 / 4):
    for i_f in np.arange(0.0015, 0.0215, 0.005 / 4):
        dataset_parameters_3.append([i_k, i_f])

dataset_parameters_list = [
    dataset_parameters_1,
    dataset_parameters_2,
    dataset_parameters_3,
]

# Diffusion constants
da = 0.2
db = 0.1

# %% Run data generation loop over three blocks to keep file size smaller
for iIteration, dataset_parameters in enumerate(dataset_parameters_list):

    seed_multiplier = iIteration
    filepat = PREFIX + str(iIteration) + ".p"
    filepar = PREFIX + "Parameters_" + str(iIteration) + ".p"
    seed_offset = round(seed_multiplier * len(dataset_parameters)) + 1

    print("Starting simulations for dataset block " + str(iIteration + 1))

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

    results = Parallel(n_jobs=8)(
        delayed(wrap_Simulation)(i) for i in range(len(dataset_parameters))
    )

    print("Saving results to " + filepat)
    pickle.dump(results, open(filepat, "wb"))
    pickle.dump(dataset_parameters, open(filepar, "wb"))
