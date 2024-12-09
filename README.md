# Hand Sign Image Classification Project

## Project Overview

The **Hand Sign Image Classification Project** aims to classify images of hand signs using various machine learning models. The dataset used in this project is the **ASL Hand Sign Dataset** (grayscale and thresholded) from [furkanakdeniz/asl-handsign-dataset-grayscaled-thresholded](https://github.com/furkanakdeniz/asl-handsign-dataset-grayscaled-thresholded). The dataset consists of grayscale and thresholded images of American Sign Language (ASL) hand gestures, with a resolution of 128x128 pixels. 

The project implements and evaluates four different models to classify these hand signs:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Convolutional Neural Network (CNN)**

The models are evaluated based on their accuracy and ability to recognize hand gestures.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models](#models)
  - [Logistic Regression](#logistic-regression)
  - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
  - [Support Vector Machine (SVM)](#support-vector-machine-svm)
  - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
- [Installation](#installation)




## Dataset

The dataset used in this project is the **ASL Hand Sign Dataset** from [furkanakdeniz/asl-handsign-dataset-grayscaled-thresholded](https://github.com/furkanakdeniz/asl-handsign-dataset-grayscaled-thresholded). It contains grayscale and thresholded images of hand gestures representing different letters of the alphabet (A-Z) in American Sign Language (ASL). The dataset does not include hand signs for **J** and **Z**, as these letters require motion to be represented correctly.

### Data Breakdown

- **Image size**: 128x128 pixels (grayscale and thresholded)
- **Number of classes**: 26 (one for each letter of the alphabet, excluding J and Z)
- **Train Data**:
  - Grayscale: 22,880 images
  - Thresholded: 30,050 images
- **Validation Data**:
  - Grayscale: 4,053 images
  - Thresholded: 7,523 images

The dataset is pre-processed to grayscale images, and the pixel values are thresholded to improve model performance.

## Models

### Logistic Regression
Logistic Regression is a linear model used for multi-class classification tasks. This model predicts the probabilities of each class and assigns the class with the highest probability to each image.

- **Implementation**: Sklearn's `LogisticRegression`
- **Key Features**:
  - Linear model
  - Multinomial loss function for multi-class classification

### K-Nearest Neighbors (KNN)
KNN is a simple, non-parametric algorithm that classifies data based on the majority class among the 'K' nearest neighbors of the point.

- **Implementation**: Sklearn's `KNeighborsClassifier`
- **Key Features**:
  - Non-linear model
  - Easy to implement
  - Sensitive to the choice of 'K' and the distance metric

### Support Vector Machine (SVM)
SVM is a powerful model that constructs a hyperplane or set of hyperplanes in a high-dimensional space, which is used for classification. It works well in high-dimensional spaces and is effective for text and image classification.

- **Implementation**: Sklearn's `SVC`
- **Key Features**:
  - Linear and non-linear separation via kernel functions
  - One-vs-rest approach for multi-class classification

### Convolutional Neural Network (CNN)
CNNs are deep learning models designed specifically for processing structured grid data like images. This project implements a simple CNN with convolutional, pooling, and fully connected layers to classify the hand signs.

- **Implementation**: TensorFlow/Keras
- **Key Features**:
  - Convolutional layers for spatial feature extraction
  - Pooling layers for downsampling
  - Fully connected layers for classification

## Installation

To run the project, you need Python 3.x and the required libraries. Follow these steps to set up the environment:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/hand-sign-images-classification-project.git
   cd hand-sign-images-classification-project

## Authors
* **NEDDAY ANAS**
* **BELHOCINE MEHDI**
* **BOUDRA Ayoub**
* **KENZEDDINE Mohamed Amine**
 