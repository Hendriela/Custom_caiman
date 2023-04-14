#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 14/04/2023 19:17
@author: hheise

This is a simple and commonly used approach where you match objects based on their closest neighbors in another image.
You can calculate distances or similarities between features of cells (e.g., Euclidean distance, cosine similarity) and
establish a threshold for matching. Cells with feature distances/similarities below the threshold are considered as
matches. You can implement this approach using libraries such as NumPy or Scipy in Python.

In this example, features_img1 and features_img2 are assumed to be numpy arrays that represent the extracted features
for cells in two images. The script calculates the pairwise distances/similarities between the features of cells in the
two images and compares them with a threshold to determine the matched cells. The matched cell indices are stored in the
matched_indices list. You can further process these matched cell pairs or use them for subsequent analysis as needed.

Note: This is a basic example of Nearest Neighbor Matching and may not be suitable for all cases. You may need to adjust
the distance/similarity metric, threshold, and other parameters based on the characteristics of your dataset and the
desired level of accuracy. Additionally, it's important to carefully validate and verify the results of any automated
cell matching method to ensure their accuracy and reliability for your specific application.
"""

import numpy as np

# Assume you have extracted features for each cell in two images, stored in numpy arrays
# features_img1 and features_img2, where each row represents the features for a single cell
features_img1 = np.zeros(1)
features_img2 = np.zeros(1)

# Calculate pairwise distances/similarities between the features of cells in two images
# You can choose the appropriate distance/similarity metric based on the characteristics of your features
# For example, you can use Euclidean distance for center of mass and spatial footprint, and cosine similarity for angles
distances = np.linalg.norm(features_img1 - features_img2, axis=1)  # Euclidean distance
similarities = np.dot(features_img1, features_img2.T) / (np.linalg.norm(features_img1, axis=1) * np.linalg.norm(
    features_img2, axis=1))[:, np.newaxis]  # Cosine similarity

# Set a threshold for matching based on the distance/similarity
threshold = 0.2  # You can adjust this threshold based on your specific dataset (low threshold is more strict)

# Initialize an array to store matched cell indices
matched_indices = []

# Iterate through cells in the first image
for i in range(features_img1.shape[0]):
    # Find the index of the closest cell in the second image
    closest_index = np.argmin(distances[i])

    # Check if the distance/similarity between the closest cell and the current cell is below the threshold
    if distances[i, closest_index] < threshold:
        # If yes, consider it as a match and store the indices of the matched cells
        matched_indices.append((i, closest_index))

# Print the matched cell indices
print("Matched Cell Indices: ", matched_indices)
