# Digit-Recognizer-Project-With-CNN

# Kaggle - Digit Recognizer Competition Solution

This repository contains a deep learning solution for the [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) competition on Kaggle, which is a classic "Getting Started" challenge. The project's goal is to build a **Convolutional Neural Network (CNN)** model to accurately identify handwritten digits from the MNIST dataset.

## Project Overview

The project aims to correctly classify grayscale images of handwritten digits (0-9), each of size 28x28 pixels. To achieve this, a comprehensive machine learning workflow was implemented, including data preprocessing, model design, training, evaluation, and prediction.

## Dataset

The dataset is provided by Kaggle and consists of two main files:
* `train.csv`: Contains 42,000 labeled images used for training the model. Each row represents an image, with the first column being the digit's `label` and the subsequent 784 columns (28x28) being the pixel values.
* `test.csv`: Contains 28,000 unlabeled images for testing the model's performance.

## Project Workflow and Architecture

The project was structured and executed in the following steps:

1.  **Data Loading and Exploration:** The `train.csv` and `test.csv` files were loaded as Pandas DataFrames. An initial analysis was performed to understand the data's structure.

2.  **Data Preprocessing:** This critical phase prepared the data for our model:
    * **Normalization:** Pixel values were scaled from the original [0, 255] range to a [0, 1] range to ensure faster and more stable model training.
    * **Reshaping:** The flat pixel vectors (1x784) were reshaped into a 3D format (28x28x1) suitable for the CNN architecture, preserving the spatial structure of the images.
    * **One-Hot Encoding:** The numerical labels (e.g., `5`) were converted into categorical vectors (e.g., `[0,0,0,0,0,1,0,0,0,0]`), which is required for the model's final classification layer and loss function.

3.  **Data Augmentation:** To prevent overfitting and improve the model's ability to generalize, random transformations (rotation, zoom, shifts) were applied to the training images using `ImageDataGenerator`.

4.  **Model Development (CNN Architecture):** A Convolutional Neural Network was built using the Keras Sequential API. The architecture is as follows:
    * **Convolutional Block 1:** Two `Conv2D` layers (32 filters, 5x5 kernel), followed by `MaxPool2D` and `Dropout(0.25)`.
    * **Convolutional Block 2:** Two `Conv2D` layers (64 filters, 3x3 kernel), followed by `MaxPool2D` and `Dropout(0.25)`.
    * **Fully Connected Block:** A `Flatten` layer, a `Dense` layer with 256 neurons, `Dropout(0.5)`, and the final `Dense` output layer with 10 neurons and `softmax` activation.
    * **Compilation:** The model was compiled with the `adam` optimizer and `categorical_crossentropy` loss function.

5.  **Model Training:** The model was trained for a set number of epochs on the augmented image data. Its performance was monitored on a separate validation set at the end of each epoch.

6.  **Prediction and Submission:** The trained model was used to predict the digits for the test dataset. The results were then formatted and saved into a `submission.csv` file as required by the competition.

## Technologies and Libraries Used
* **Python 3**
* **TensorFlow & Keras:** For building, compiling, and training the deep learning model.
* **Pandas:** For loading and managing the datasets.
* **NumPy:** For high-performance numerical computations and data manipulation.
* **Scikit-learn:** For splitting the data into training and validation sets.
* **Matplotlib & Seaborn:** For visualizing the data and the model's training history.

## Results and Performance
The developed model achieved a high accuracy score on the validation set, demonstrating the effectiveness of the CNN architecture and data augmentation techniques for image classification tasks.

* **Version 1 (Trained for 20 Epochs):**
    * **Validation Accuracy:** Approximately **99.428%**
    * **Kaggle Public Leaderboard Score/Rank:** [** 0.99428  Score/ 159 Rank **]

Further improvements could be achieved by training for more epochs, fine-tuning hyperparameters, or using a more complex architecture.
