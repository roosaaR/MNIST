# MNIST Classification with KNN, Naive Bayes, and Full Bayes

This repository contains Python scripts that classify the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digit dataset using different machine learning algorithms: **1-Nearest Neighbor (KNN)**, **Naive Bayes**, and **Full Bayes**.

## Overview

The MNIST dataset is a collection of 28x28 pixel grayscale images of handwritten digits (0-9). It is widely used as a benchmark for image classification algorithms.

In this project, we implement and evaluate three classification methods:
1. **1-Nearest Neighbor (1-NN)** — Classifies based on the closest training example.
2. **Naive Bayes** — Assumes independence between pixel values and uses probabilistic modeling for classification.
3. **Full Bayes** — Models the entire joint probability distribution of the pixel values.
4. **Multilayer Perceptron (MLP)** — A fully connected neural network that uses backpropagation to learn from the data.

### Files Included
- **`mnist_nn.py`**: Implementation of the 1-Nearest Neighbor classifier for MNIST data.
- **`mnist_naive_bayes.py`**: Implementation of Naive Bayes classification.
- **`mnist_full_bayes.py`**: Implementation of Full Bayes classification.
- **`mnist_mlp.py`**: Implementation of the Multilayer Perceptron (MLP) classifier using Keras.
- **`utils.py`**: Contains utility functions such as data preprocessing and accuracy calculation.
- **`README.md`**: This documentation file.

## 1-Nearest Neighbor Classifier (mnist_nn.py)

The **`mnist_nn.py`** file implements a 1-Nearest Neighbor (KNN) classifier for the MNIST dataset. The key steps in the process are:

### Functions

- **`class_acc(pred, gt)`**: 
  - Computes the classification accuracy by comparing the predicted labels (`pred`) to the ground truth labels (`gt`).
  
- **`preprocess_data(x_train, x_test)`**: 
  - Flattens the MNIST images (28x28) into a 1D array and normalizes pixel values to the range [0, 1].
  
- **`train_data(x_train, y_train)`**: 
  - Trains a 1-Nearest Neighbor classifier on the training data.
  
- **`print_accuracy(knn, x_test, y_test)`**: 
  - Predicts the labels for the test set and prints the classification accuracy.


## Naive Bayes Classifier (mnist_naive_bayes.py)

The **`mnist_naive_bayes.py`** file implements the Naive Bayes classifier for the MNIST dataset. This classifier assumes independence between pixel values, which simplifies the calculation of class probabilities.

### Key Features:
- **Gaussian Naive Bayes**: The algorithm assumes that the pixel values follow a Gaussian (normal) distribution for each class.
- **Noise Addition**: Before training, a small amount of Gaussian noise is added to the training data to simulate variations in the data and improve the robustness of the classifier.

### Functions

- **`add_noise(x_train)`**:
  - Adds white noise (Gaussian noise) to the training data to make the model more robust.
  
- **`calculate_means(x_train, classes)`**:
  - Computes the mean pixel values for each class (0-9) in the training dataset.

- **`calculate_variances(x_train, classes)`**:
  - Computes the variance of pixel values for each class (0-9), which is necessary for the Gaussian Naive Bayes likelihood calculations.

- **`log_likelihood(x, mean, var)`**:
  - Calculates the log-likelihood of a test sample belonging to a particular class, based on the pixel-wise means and variances of that class.

- **`predict_class(x, means, variances)`**:
  - Predicts the class label for a given input by comparing the log-likelihoods for each class.


## Full Bayes Classifier (mnist_full_bayes.py)

The **`mnist_full_bayes.py`** file implements the Full Bayes classifier for the MNIST dataset. Unlike Naive Bayes, which assumes pixel independence, the Full Bayes classifier models the **joint distribution** of pixel values, allowing for more complex relationships between the pixels.

### Key Features:
- **Full Joint Distribution**: The classifier calculates the full covariance matrix for each class, which allows it to model the joint relationships between all pixel values.
- **Multivariate Gaussian Distribution**: Each class's pixels are modeled as following a multivariate normal distribution, with a different mean and covariance matrix for each class.
- **Noise Addition**: Before training, white noise is added to the training data to make the classifier more robust.

### Functions

- **`preprocess_data(x_train, x_test)`**:
  - Flattens the MNIST images from 2D (28x28) to 1D arrays and normalizes pixel values to the range [0, 1].

- **`add_noise(x_train)`**:
  - Adds zero-mean white noise to the training data to improve the model's generalization.

- **`calculate_means(x_train, classes)`**:
  - Calculates the mean pixel values for each class (0-9) in the training dataset.

- **`calculate_covariance(x_train, classes)`**:
  - Calculates the covariance matrices for each class. The covariance matrix models the relationships between pixel values. A small regularization term is added to ensure numerical stability when inverting the covariance matrix.

- **`predict_class(x_test, means, covariances)`**:
  - Predicts the class label for each input test sample by calculating the log-likelihood for each class and returning the class with the highest log-likelihood.


## Multilayer Perceptron (MLP) Classifier (mnist_mlp.py)

The **`mnist_mlp.py`** file implements a Multilayer Perceptron (MLP) classifier for the MNIST dataset using Keras. The MLP is a fully connected feedforward neural network trained with backpropagation.

### Key Features:
- **Two Hidden Layers**:
  - The first hidden layer has 64 neurons with the ReLU activation function.
  - The second hidden layer has 32 neurons, also with ReLU activation.
  
- **Output Layer**: The output layer has 10 neurons with a softmax activation function to represent the 10 possible digit classes (0-9).
  
- **SGD Optimizer**: The model is trained using Stochastic Gradient Descent (SGD) with a learning rate of 0.01.

### Functions

- **`preprocess_data(x_train, x_test, y_train, y_test)`**:
  - Flattens the 28x28 pixel MNIST images into 784-dimensional vectors, normalizes pixel values, and one-hot encodes the class labels.

- **`create_model()`**:
  - Creates the MLP model with two hidden layers and an output layer using Keras.

- **`plot_data(tr_hist)`**:
  - Plots the training loss curve over time to visualize how well the model learns during training.
