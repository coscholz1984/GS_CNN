# Exploring pattern formation in the Gray-Scott model with convolutional neural networks

---

## 1. Introduction

A neural network approach to classify patterns in a reaction-diffusion system (here: the Gray-Scott model).

The goal of this project is to identify patterns in the Gray-Scott reaction-diffusion model and possibly others using neural networks.

The Gray-Scott model is a system of two coupled reation-diffusion equations, that is well know for the variety of patterns that it can display based on the input parameters and initial conditions. Several types of patterns have been identified in the literature based on manual searches and subsequent definitions of spatial, temporal and statistical properties of these patterns. It is difficult however, to automize the search for these patters in the parameter-space, since there is no simple method to identify the exact regions, where a specific type of pattern is found.
Instead pattern types are often identified manually by researches by the naked eye and then later categorized.

This project is aimed at improving this approach using convolutional neural networks. We start from a set of well know patterns that have been identified in the literature and then fine-scan the parameter-space using a neural network that has been training to recognize the already known pattern types.

For an in-depth explanation of the system and results, see the following publication:

	C. Scholz and S. Scholz, "Exploring complex pattern formation with convolutional neural networks" (2021)

To cite this work, please cite the above publication, or, in case of specific code, please cite
	
	C. Scholz and S. Scholz, https://github.com/coscholz1984/GS_CNN

---

## 2. Requirements

The scripts published here require the following libraries, incl. versions

- python 3.8.3
- keras 2.3.1 with TensorFlow backend
- numba 0.49.1
- numpy 1.18.4
- pandas 1.0.3
- pickle 4.0
- sklearn 0.23.1

All scripts have been tested under the WinPython 3.8.3 distribution. Scripts that use parallelization must be executed from command line.

---

## 3. Instructions:

### 3.1. Display pattern formation in the Gray-Scott model

Run a single simulation for a specific set of input parameters [seed] [D_u] [f] [k]. For example, display a solution from the alpha class:

    python gray_scott_2D.py 2 0.2 0.009 0.045

Varying the seed will change the initial conditions randomly. In GStools.py the function get_dataset_parameter() returns a set of parameters for each class. 

### 3.2. Generate Training, Validation and Test Data for 2D convolution CNN case:

Execute the following commands to generate training, validation and test datasets. Depending on your CPU, you can adapt the constant NTHREADS (default 8) to speed parallel execution.

    python Dataset_Generate.py train 2D
    python Dataset_Generate.py val 2D
    python Dataset_Generate.py test 2D

Each script runs ~2 hrs on an Intel core i7 9th gen processor with 6 cores. All raw data files are also available for download from a OSF repository at https://osf.io/byrzm/

### 3.3. Neural networks training

#### Train neural network:

When you have the datasets available, run the following script to train the model.

    python CNN_Train_2D.py
	
The script runs for ~1.4 hrs on an Intel core i7 9th gen processor with 6 cores. Pre-trained model weights are also stored in this repository.

#### Evaluate results:

Evaluate training history and accuracy.

    python CNN_Evaluate.py model_CNN_2D

#### Display saliency maps:

Display the saliency maps for selected patterns from training dataset.

    python CNN_Saliency.py

### 3.4. Parameter space fine-scan

#### Generate fine-scan of parameter space:

This generates a fine scan of the parameter space. This script takes about 24 hrs to run. The raw data is also available for download from a OSF repository at https://osf.io/byrzm/

    python Parameter_Space_Dataset_Generate.py

#### Classify patterns in parameter space:

To classify the results using the 2D convolutional CNN, run the following.

    python Parameter_Space_Dataset_Classify.py model_CNN_2D

#### Plot classification predictions per class. Click on a pattern to display parameters (k,f)

This displays the patterns associated with a certain class.

    python Predicted_Class_Montage.py model_CNN_2D 0

To specify the displayed class set the second input parameter to any number from 0 to 14.

#### Generate Training and Validation data for 2nd training pass

To generate training data with additional parameters run.

    python Dataset_Generate.py train2 2D
    python Dataset_Generate.py val2 2D

Both scripts run for ~5 hrs. The raw data is also available for download from a OSF repository at https://osf.io/byrzm/

#### Perform 2nd training iteration

Run the second iteration of the training.

    python CNN_Train_2D_2nd.py
	
The script runs for ~1.4 hrs on an Intel core i7 9th gen processor with 6 cores. Pre-trained weights are also stored in this repository.

#### Display results for classification:

Display results of the new model.

    python Parameter_Space_Dataset_Classify.py model_CNN_2D_2nd

---

## 4. Bonus: 3D Convolutional network

### 4.1. Neural network with 3D convolutions

#### Generate Training, Validation and Test data for 3D CNN

    python Dataset_Generate.py train 3D
    python Dataset_Generate.py val 3D
    python Dataset_Generate.py test 3D

Each script runs ~2 hrs on an Intel core i7 9th gen processor with 6 cores. The raw data is also available for download from a OSF repository at https://osf.io/byrzm/

#### Training 3D convolution

    python CNN_Train_3D.py

This script runs for ~14 hrs. Pre-trained weights are also stored in this repository.

#### Evaluate training results

    python CNN_Evaluate.py model_CNN_3D
