#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 14/04/2023 19:57
@author: hheise

You can also use deep learning-based approaches to match cells across images. For example, you can train a convolutional
neural network (CNN) to learn feature representations of cells from images and use these representations to match cells
across different images. Siamese networks (SNN), which are specifically designed for image matching tasks, can be used
for this purpose. You can implement this approach using deep learning libraries such as TensorFlow or PyTorch in Python.

In this example, a Siamese neural network is used to predict the similarity between cells in two images based on their
extracted features. The network is trained on a ground-truth dataset of manually confirmed matched cells, and the
trained model is then used to predict similarity scores for all cell pairs. A threshold is set to determine if a pair
of cells is considered as a match based on the predicted similarity score.

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Model
from keras.layers import Input, Dense, concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras.backend as K
import tensorflow as tf

n_cells = 100
n_features = 5

# Hyperparameters (might need tuning)


# Assume you have extracted features for each cell in two images, stored in numpy arrays
# features_img1 and features_img2, where each row represents the features for a single cell
features_img1 = pd.read_csv('.\\custom scripts\\chatgpt_suggestions\\reference_cell_features.csv')
features_img2 = pd.read_csv('.\\custom scripts\\chatgpt_suggestions\\target_cell_features.csv')

# Assume you have a ground-truth dataset of manually confirmed matched cells for training, stored in a numpy array
# ground_truth, where each row represents a pair of matched cells as (img1_idx, img2_idx) indices
ground_truth = pd.read_csv('.\\custom scripts\\chatgpt_suggestions\\ground_truth.csv')

"""
Our ground truth consists of manually confirmed matches. However, the SNN outputs similarity scores from 0 to 1.
To bridge this, we have to do two things:
1.  We generate positive and negative examples from the ground truth matches. Positive examples are confirmed matches,
    which will get a label of 1. Negative examples are confirmed non-matches, which will get a label of 0. The SNN 
    should learn to assign a high similarity score to positive, and a low score to negative examples. The dataset should
    contain a similar amount of positive and negative examples to avoid bias towards negative examples (which are much
    more abundant in the dataset). While negative examples are randomly downsampled, it might be a good idea to bias the
    selection towards cells within close proximity, as these are the most difficult pairs to evaluate. It might also be
    worth it to completely ignore pairs with a distance above a certain threshold, given that the FOVs are reasonably 
    well aligned and cells cannot travel more than a few microns between images.
2.  To let the SNN output a float similarity score based on a binary input, we have to use a custom "contrastive loss"
    function. This function computes the loss for positive and negative pairs separately and punishes negative pairs
    for having a similarity score larger than 0, and positive pairs for having a similarity score less than "margin",
    a similarity threshold below above which a pair can be considered "similar".
"""

def generate_pairs(X, y):
    pairs = []
    labels = []
    num_pairs = len(y)
    for i in range(num_pairs):
        pairs.append([X[i, 0], X[i, 1]])
        labels.append(y[i])

        # Todo: implement bias for negative examples with close distance
        # Generate negative examples by randomly selecting pairs
        for _ in range(2):  # adjust number of negative examples per positive example
            random_idx = np.random.randint(num_pairs)
            pairs.append([X[i, 0], X[random_idx, 1]])
            labels.append(0)  # label as non-match
    return np.array(pairs), np.array(labels)


# Custom contrastive loss function for Siamese Network
def contrastive_loss(y_true: tf.Tensor, y_pred: tf.Tensor, margin: float = 1) -> float:
    """
    Contrastive loss function for Siamese Neural Network.

    Args:
        y_true: A tensor of true labels (0 or 1) for positive or negative pairs, shape: (batch_size,).
        y_pred: A tensor of predicted similarity scores (or distances?) (float values between 0 and 1), shape: (batch_size,).
        margin: Margin value to control the dissimilarity threshold between positive and negative pairs.

    Returns:
        A scalar representing the average contrastive loss of the current batch.
    """

    # Unify true labels to same datatype as prediction values and reshape for broadcasting
    y_true = K.cast(y_true, dtype=y_pred.dtype)
    y_true = K.reshape(y_true, [-1])

    # Todo: check how to interpret SNN output (similarity or distance)
    ### If SNN output is interpreted as similarity (similar pairs have high value):
    # # Punish negative pairs (where y_true == 0) for having a similarity score > 0
    # neg_loss = (1 - y_true) * K.square(y_pred)
    # # Punish positive pairs (where y_true == 1) for having a similarity score < margin
    # pos_loss = y_true * K.square(K.maximum(margin - y_pred, 0))

    ### If SNN output is interpreted as distance (similar pairs have low value):
    # Punish negative pairs (where y_true == 0) for having a similarity score > 0
    neg_loss = y_true * K.square(y_pred)
    # Punish positive pairs (where y_true == 1) for having a similarity score < margin
    pos_loss = (1 - y_true) * K.square(K.maximum(margin - y_pred, 0))

    # Sum to get the loss of all pairs, and return the average contrastive loss of the current batch
    return K.mean(neg_loss + pos_loss)


# Normalize the features
scaler = StandardScaler()
features_img1_norm = scaler.fit_transform(features_img1)
features_img2_norm = scaler.transform(features_img2)

# Create training and validation datasets
X_train, X_val, y_train, y_val = train_test_split(np.concatenate((features_img1_norm, features_img2_norm), axis=1),
                                                  ground_truth, test_size=0.2, random_state=42)

# Todo: Check typical architecture of Siamese networks
# Define Siamese neural network architecture
input_dim = features_img1_norm.shape[1]
input_img1 = Input(shape=(input_dim,))
input_img2 = Input(shape=(input_dim,))
shared_layer = Dense(64, activation='relu')
x1 = shared_layer(input_img1)
x2 = shared_layer(input_img2)
merged = concatenate([x1, x2])
output = Dense(1, activation='sigmoid')(merged)
model = Model(inputs=[input_img1, input_img2], outputs=output)

# Compile the model
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss=contrastive_loss, metrics=['accuracy'])

# Set up early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Train the Siamese neural network
model.fit([X_train[:, :input_dim], X_train[:, input_dim:]], y_train, batch_size=32, epochs=100,
          validation_data=([X_val[:, :input_dim], X_val[:, input_dim:]], y_val), callbacks=[early_stop])

# Predict the similarity scores between cells in two images
predictions = model.predict([features_img1_norm, features_img2_norm])

# Set a threshold for matching based on the similarity scores
# Todo: evaluate threshold for matches, possibly hyperparameter-tune
threshold = 0.5  # This threshold measures similarity, so a higher threshold is more strict

# Initialize an array to store matched cell indices
matched_indices = []

# Iterate through the similarity scores
for i in range(predictions.shape[0]):
    if predictions[i] > threshold:
        # If the similarity score is above the threshold, consider it as a match
        matched_indices.append((i, np.argmax(predictions[i])))

# Print the matched cell indices
print("Matched Cell Indices: ", matched_indices)

"""
The Siamese Network outputs a similarity score for all cell pairs. A threshold is needed to determine whether a 
suggested match is actually a match. For this we have to tune the threshold by evaluating the model performance using
several metrics:

1.  Precision: Precision is the proportion of true positive matches (correctly matched pairs) out of the total number of
    positive matches (the sum of true positives and false positives). It measures the accuracy of positive predictions 
    made by the algorithm. The formula for precision is:
    Precision = True Positives / (True Positives + False Positives)

2.  Recall (Sensitivity, True Positive Rate): Recall is the proportion of true positive matches out of the total number 
    of actual positive matches (the sum of true positives and false negatives). It measures the ability of the algorithm 
    to correctly identify all the positive matches. The formula for recall is:
    Recall = True Positives / (True Positives + False Negatives)

3.  F1-score: The F1-score is the harmonic mean of precision and recall, and it provides a balanced measure of precision 
    and recall. It is commonly used when both precision and recall are important. The formula for F1-score is:
    F1-score = 2 * (Precision * Recall) / (Precision + Recall)

4.  Accuracy: Accuracy is the proportion of correct matches (both true positives and true negatives) out of the total 
    number of matches. It measures the overall correctness of the algorithm's predictions. The formula for accuracy is:
    Accuracy = (True Positives + True Negatives) / (True Positives + False Positives + False Negatives + True Negatives)
"""
