# Image Classification Model Training
=====================================

## Table of Contents
---------------

* [Getting Started](#getting-started)
* [Command Line Arguments](#command-line-arguments)
* [Model Architecture](#model-architecture)
* [Training](#training)
* [Testing](#testing)
* [Saving the Model](#saving-the-model)
* [Dataset](#dataset)
* [Example Usage](#example-usage)

## Getting Started
---------------

This project trains an image classification model using PyTorch and torchvision. The model is trained on a dataset of images, and the goal is to classify each image into one of 102 categories.

To run the model, navigate to the project directory and run the training script using the following command:

```bash
python train.py <data_dir> --save_dir <save_dir> --arch <arch> --learning_rate <learning_rate> --hidden_units <hidden_units> --epochs <epochs> --gpu

Command Line Arguments

Available Arguments
data_dir: The path to the dataset directory (required)
save_dir: The path to the directory where the model checkpoint will be saved (default: ./)
arch: The architecture of the model (default: densenet121, options: densenet121, vgg13)
learning_rate: The learning rate for the optimizer (default: 0.001)
hidden_units: The number of hidden units in the classifier (default: 512)
epochs: The number of epochs to train the model (default: 20)
gpu: Whether to use CUDA (default: False)

Model Architecture

The model architecture is based on a pre-trained convolutional neural network (CNN) from torchvision. The CNN is fine-tuned for the image classification task by adding a custom classifier on top of the pre-trained model. The classifier consists of multiple layers, including a dropout layer.

Training

The model is trained using the Adam optimizer and cross-entropy loss. The training process is divided into epochs, and the model is validated on a separate validation dataset after each epoch. The best model checkpoint is saved based on the validation accuracy.

Testing

The model is tested on a separate test dataset, and the accuracy is calculated.

Saving the Model

The model checkpoint is saved to a file named checkpoint.pth in the specified save_dir. The checkpoint contains the model architecture, model state dictionary, and the class-to-index mapping.

Dataset

The dataset is expected to be stored in a directory with the following structure:
data_dir
train
valid
test

Each subdirectory contains images of the corresponding category.

Example Usage

Training with Default Settings
python train.py data --gpu

Training with Custom Architecture and Hyperparameters

python train.py data --arch vgg13 --learning_rate 0.01 --hidden_units 512 --epochs 10 --save_dir ./ --gpu
undefined