FFA for ONNs 

This repository implements a multi-layer opto-electronic neural network designed for image classification. The network is composed of both optical layers and post-processing digital layers, and it supports training using the FFA for updating parameters. The provided code uses the MNIST dataset and Fashion MNIST for training and testing in the manuscript.

Setup

To get started with the project, you need to clone the repository and install the necessary dependencies. Make sure Python 3.6+ installed, along with PyTorch 1.13.1 and Numpy.

File Structure

dataloader_train.py: Contains the DatasetFromFilenames class, which loads positive and negative image paths for training, applies transformations, and prepares the data for training.
dataloader_test.py: Defines the DatasetFromFilenames_test class for loading test data.
Layer.py: Contains definitions for the Optical_Layer, Postprocessing_Layer, and Layer classes, which define the architecture of the neural network, including the optical and post-processing layers.
main.py: The main entry point for the code, which handles data loading, model setup, and training execution.
train.py: Defines the train function for training the model and the test function for evaluating model performance.

Usage

Prepare the MNIST dataset: The paths in the code if needed (e.g., train_pos_filenames, train_neg_filenames, test_filenames). Then run the training and test the model.

Training Parameters

Batch size: 256
Learning rate: 0.05 for the first 500 epochs and 0.008 for the last 500 epochs
Theta:100
