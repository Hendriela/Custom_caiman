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

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Model
from keras import layers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras.backend as K
import tensorflow as tf

#%% Custom parameters of network (might not need tuning)
# Names of the features that should be used as input
FEATURES = ['rois_nearby', 'closest_roi_angle', 'com_x', 'com_y', 'quad_ul', 'quad_ur', 'quad_ll', 'quad_lr']
generate_new_pairs = False      # If True, new pairs are generated. Otherwise, previously generated pairs are loaded.
RANDOM_STATE = None             # Random state of the train_test_splitter for reproducibility across function calls. If None, purely random splitter.
DATA_DIR = '.\\custom scripts\\chatgpt_suggestions'

#%% Hyperparameters (might need tuning)

# Fraction of the total dataset that should be kept for the validation and testing datasets
VAL_TEST_SIZE = 0.2

# Fraction of the validation-testing dataset that should be used for validation
VAL_SIZE = 0.5

# The number of negative examples for each positive example (4:1 seems a good ratio for text, see doi_10.18653/v1/W16-1617)
N_NEGATIVE = 4

# Distance below which negative pairs are pushed further apart, the "acceptable" range around a sample in which a pair
# is classified as "similar". A smaller margin makes the network stricter and more sensitive.
MARGIN = 1.0    # A margin of 1 seems to be standard for contrastive loss

# Number of nodes/neurons in the first dense hidden layer
DENSE_SIZE = 4

BATCH_SIZE = 128
EPOCHS = 100


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


#%% Functions
def load_features():
    # Assume you have extracted features for each cell in two sets, stored in numpy arrays features_set1 (reference
    # cells) and features_set2 (target cells), where each row represents the features for a single cell.
    feat_set1 = pd.read_csv(os.path.join(DATA_DIR, 'reference_cell_features.csv'), index_col=0)
    feat_set2 = pd.read_csv(os.path.join(DATA_DIR, 'target_cell_features.csv'), index_col=0)

    # Assume you have a ground-truth dataset of manually confirmed matched cells for training, stored in a numpy array
    # ground_truth, where each row represents a pair of matched cells as (set1_idx, set2_idx) indices.
    truth = pd.read_csv(os.path.join(DATA_DIR, 'ground_truth.csv'), index_col=0)

    return feat_set1, feat_set2, truth


def save_pairs(new_pairs, new_labels):
    p_1 = new_pairs[:, 0]
    p_2 = new_pairs[:, 1]
    np.savetxt(os.path.join(DATA_DIR, 'pairs_ref.csv'), p_1, delimiter=',', fmt='%.8f')
    np.savetxt(os.path.join(DATA_DIR, 'pairs_tar.csv'), p_2, delimiter=',', fmt='%.8f')
    np.savetxt(os.path.join(DATA_DIR, 'labels.csv'), new_labels, delimiter=',', fmt='%.8f')


def load_pairs():
    """
    Load previously generated pairs.

    Returns:
        Pairs, shape (n_pairs, 2, n_features); and labels, shape (n_pairs,)
    """
    p_1 = np.loadtxt(os.path.join(DATA_DIR, 'pairs_ref.csv'), delimiter=',')
    p_2 = np.loadtxt(os.path.join(DATA_DIR, 'pairs_tar.csv'), delimiter=',')
    pair_label = np.loadtxt(os.path.join(DATA_DIR, 'labels.csv'), delimiter=',')
    pair_stack = np.stack((p_1, p_2), axis=1)
    return pair_stack, pair_label


def generate_pairs(set1: pd.DataFrame, set2: pd.DataFrame, y: pd.DataFrame):
    """

    Args:
        set1: DataFrame of set 1 (reference cells) and their features. -> features_set1
        set2: DataFrame of set 2 (target cells) and their features. -> features_set2
        y: DataFrame with two columns, containing confirmed matches of the global_mask_idx of set1 cells to set2 cells.

    Returns:

    """

    pair = []
    lab = []

    for _, curr_match in y.iterrows():

        # For each matching cells, select the corresponding row from the sets, keep relevant features, and transform
        # into a numpy array for further processing
        set1_cell = np.array(set1[set1['global_mask_id'] == curr_match['mask_id']][FEATURES]).squeeze()
        set2_cell = np.array(set2[set2['global_mask_id'] == curr_match['matched_id']][FEATURES]).squeeze()

        # Append the two cell-features together with the label 1, since these are confirmed matches
        pair.append([set1_cell, set2_cell])
        lab.append(1)

        # Todo: implement bias for negative examples with close distance
        # Generate negative examples by randomly selecting non-matching pairs
        for _ in range(N_NEGATIVE):

            # Get the index of a random set1 cell
            random_1 = np.random.choice(set1['global_mask_id'])

            # Get the index of a random set2 cell, excluding accepted matches for this cell
            blacklist = np.array(y[y['mask_id'] == random_1])[:, 1]
            non_matches = set2.loc[~set2['global_mask_id'].isin(blacklist)]
            random_2 = np.random.choice(non_matches['global_mask_id'])

            # Get features for this non-matching pair of cells
            set1_cell = np.array(set1[set1['global_mask_id'] == random_1][FEATURES]).squeeze()
            set2_cell = np.array(set2[set2['global_mask_id'] == random_2][FEATURES]).squeeze()

            # Append the pair and label as 0, non-match
            pair.append([set1_cell, set2_cell])
            lab.append(0)

    return np.array(pair), np.array(lab)


def generate_datasets(dataset, label):
    """
    Generate training, validation and testing datasets from a large dataset of pairs and standardizes after splitting.

    Args:
        dataset: 3D array of total dataset, shape (n_pairs, 2, n_features). -> from generate_pairs()
        label: 1D array of labels, shape (n_pairs). 0 for no match, 1 for matching pair.

    Returns:
        Tuple of all training, validation and testing datasets and labels
    """

    # Create training, validation and testing datasets
    x_train, x_val_test, y_training, y_val_test = train_test_split(dataset, label, test_size=VAL_TEST_SIZE,
                                                                   random_state=RANDOM_STATE)
    val_mask = np.random.rand(len(x_val_test)) < VAL_SIZE  # Use 50% of the val_test set as validation (10% of dataset)
    x_val = x_val_test[val_mask]
    x_test = x_val_test[~val_mask]
    y_valid = y_val_test[val_mask]
    y_testing = y_val_test[~val_mask]

    # Split up pairs again (Scaler needs 2D array)
    x_training_1 = x_train[:, 0]
    x_training_2 = x_train[:, 1]
    x_valid_1 = x_val[:, 0]
    x_valid_2 = x_val[:, 1]
    x_testing_1 = x_test[:, 0]
    x_testing_2 = x_test[:, 1]

    # Normalize the features (only fit on training set, apply to all)
    scaler = StandardScaler()
    scaler.fit(np.concatenate((x_training_1, x_training_2)))
    x_train_1_norm = scaler.transform(x_training_1)
    x_train_2_norm = scaler.transform(x_training_2)
    x_val_1_norm = scaler.transform(x_valid_1)
    x_val_2_norm = scaler.transform(x_valid_2)
    x_test_1_norm = scaler.transform(x_testing_1)
    x_test_2_norm = scaler.transform(x_testing_2)

    return (x_train_1_norm, x_train_2_norm, y_training, x_val_1_norm, x_val_2_norm, y_valid, x_test_1_norm,
            x_test_2_norm, y_testing)


def euclidean_distance(vectors):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vectors: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """
    # Todo: Possibly Euclidean distance is also not useful, but angular distance is needed for high-dimensional space

    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sum_squared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))


def contrastive_loss(y_true: tf.Tensor, y_pred: tf.Tensor, margin: float = MARGIN) -> float:
    """
    Contrastive loss function for Siamese Neural Network.

    Args:
        y_true: A tensor of true labels (0 or 1) for positive or negative pairs, shape: (batch_size,).
        y_pred: A tensor of predicted distances (float values between 0 and 1), shape: (batch_size,).
        margin: Margin value to control the dissimilarity threshold between positive and negative pairs.

    Returns:
        A scalar representing the average contrastive loss of the current batch.
    """

    # Unify true labels to same datatype as prediction values and reshape for broadcasting
    y_true = K.cast(y_true, dtype=y_pred.dtype)
    y_true = K.reshape(y_true, [-1])

    ### If SNN output is interpreted as similarity (similar pairs have high value):
    # # Punish negative pairs (where y_true == 0) for having a similarity score > 0
    # neg_loss = (1 - y_true) * K.square(y_pred)
    # # Punish positive pairs (where y_true == 1) for having a similarity score < margin
    # pos_loss = y_true * K.square(K.maximum(margin - y_pred, 0))

    ### If SNN output is interpreted as distance (similar pairs have low value):
    # Punish positive pairs (where y_true == 1) for having a distance > 0, downscaled by sampling ratio of pos/neg
    pos_loss = y_true * K.square(y_pred) * (1/N_NEGATIVE)

    # Punish negative pairs (where y_true == 0) for having a distance < margin -> hinge loss
    # (negative pairs with a high distance are already well separated and don't have to be punished
    neg_loss = (1 - y_true) * K.square(K.maximum(margin - y_pred, 0))

    # Sum to get the loss of all pairs, and return the average contrastive loss of the current batch
    return K.mean(neg_loss + pos_loss)


def build_model(n_features):
    """
    Define Siamese neural network architecture. Adapted from https://keras.io/examples/vision/siamese_contrastive/.

    Args:
        n_features: Number of features per cell, determines input shape

    Returns:

    """

    # Single tower
    input_layer = layers.Input(shape=(n_features,))   # Input layer (gets features of a single cell)
    x = layers.BatchNormalization()(input_layer)      # Batch Normalization might not be necessary, but should not be harmful
    x = layers.Dense(DENSE_SIZE, activation='relu')(x)  # Dense layer that processes the input features
    embedding_network = Model(input_layer, x)         # This is one tower, embedding the cell features into an abstract space

    # Create two instances of the same embedding network, creating one tower for each cell in the matched pairs
    # This allows the Siamese Network to share weights between both towers (they are trained together)
    input_1 = layers.Input(shape=(n_features,))
    input_2 = layers.Input(shape=(n_features,))
    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    # Create a merge layer, which takes the embedding of both towers and computes the Euclidean distance between them
    merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])

    # Another normalization, then the output layer with 1 output, and sigmoid activation for binary output type
    normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
    output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)

    # Create the model with two input layers and our single output layer, which gets the data flowing through the model
    siamese = Model(inputs=[input_1, input_2], outputs=output_layer)

    return siamese


def plt_metric(hist, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'.

    Arguments:
        hist: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.

    Returns:
        None.
    """
    plt.plot(hist[metric])
    if has_valid:
        plt.plot(hist["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()


#%% Setting up and running the model

# Load data and generate or load pairs
if generate_new_pairs:

    features_set1, features_set2, ground_truth = load_features()
    # Generate positive and negative examples
    pairs, labels = generate_pairs(features_set1, features_set2, ground_truth)
    # Save pairs and labels to save time, generating pairs takes a few minutes
    save_pairs(pairs, labels)

else:
    pairs, labels = load_pairs()

# Split dataset
x_train_1, x_train_2, y_train, x_val_1, x_val_2, y_val, x_test_1, x_test_2, y_test = generate_datasets(dataset=pairs, label=labels)

# Build the model architecture
model = build_model(n_features=x_train_1.shape[1])

# Compile the model
optimizer = Adam(learning_rate=0.001, epsilon=0.001)
model.compile(optimizer=optimizer, loss=contrastive_loss, metrics=['accuracy'])

# Set up early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Train the Siamese neural network
history = model.fit([x_train_1, x_train_2],
                    y_train,
                    validation_data=([x_val_1, x_val_2], y_val),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=[early_stop])

# Plot the accuracy
plt_metric(hist=history.history, metric="accuracy", title="Model accuracy")

# Plot the contrastive loss
plt_metric(hist=history.history, metric="loss", title="Contrastive Loss")

# Predict the similarity scores between cells in two images
predictions = model.predict([x_test_1, x_test_2])

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
