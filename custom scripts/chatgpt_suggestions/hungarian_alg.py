#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 14/04/2023 19:33
@author: hheise

The Hungarian algorithm, also known as the Munkres algorithm, is an optimal assignment algorithm that can be used for
object matching. It finds the optimal assignment of objects across two images based on the cost matrix, which represents
the pairwise distances or similarities between the features of cells. You can implement this algorithm using libraries
such as SciPy in Python.

In this example, features_img1 and features_img2 are assumed to be numpy arrays that represent the extracted features
for cells in two images. The script calculates the pairwise distances/similarities between the features of cells in the
two images and creates a cost matrix for the Hungarian Algorithm. The Hungarian Algorithm is then used to find the
optimal matching between cells based on the cost matrix. The matched cell indices are stored in the matched_indices
list. You can further process these matched cell pairs or use them for subsequent analysis as needed.

Note: The Hungarian Algorithm is a more sophisticated method for cell matching that can handle cases where there are
multiple cells in one image that need to be matched with multiple cells in another image. However, it may have higher
computational complexity compared to Nearest Neighbor Matching, so it may not be suitable for very large datasets. As
with any cell matching method, it's important to carefully validate and verify the results to ensure their accuracy and
reliability for your specific application.

Hendrik: This suggestion only makes use of one feature, e.g. Euclidean distance between cells. To use a combination of
features for this approach, check out the code in graph_hungarian.py.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

# Assume you have extracted features for each cell in two images, stored in numpy arrays
# features_img1 and features_img2, where each row represents the features for a single cell
features_img1 = np.zeros(1)
features_img2 = np.zeros(1)

# Calculate pairwise distances/similarities between the features of cells in two images
# You can choose the appropriate distance/similarity metric based on the characteristics of your features
# For example, you can use Euclidean distance for center of mass and spatial footprint, and cosine similarity for angles
distances = np.linalg.norm(features_img1 - features_img2, axis=1)  # Euclidean distance
similarities = np.dot(features_img1, features_img2.T) / (np.linalg.norm(features_img1, axis=1) * np.linalg.norm(features_img2, axis=1))[:, np.newaxis]  # Cosine similarity

# Set a threshold for matching based on the distance/similarity
threshold = 0.2  # You can adjust this threshold based on your specific dataset

# Create a cost matrix for the Hungarian Algorithm
# Set the cost to a large value for cell pairs whose distance/similarity is above the threshold
# and set it to the distance/similarity value for cell pairs whose distance/similarity is below the threshold
cost_matrix = np.where(distances < threshold, distances, np.inf)

# Use the Hungarian Algorithm to find the optimal matching between cells
row_indices, col_indices = linear_sum_assignment(cost_matrix)

# Initialize an array to store matched cell indices
matched_indices = []

# Iterate through the matched cell indices returned by the Hungarian Algorithm
for row_idx, col_idx in zip(row_indices, col_indices):
    # Check if the cost for the matched cell pair is below the threshold
    if cost_matrix[row_idx, col_idx] < np.inf:
        # If yes, consider it as a match and store the indices of the matched cells
        matched_indices.append((row_idx, col_idx))

# Print the matched cell indices
print("Matched Cell Indices: ", matched_indices)
